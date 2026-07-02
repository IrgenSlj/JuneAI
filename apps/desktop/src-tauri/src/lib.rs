// Tauri entry point for the June desktop shell.
//
// Phase 1 scaffold opened the SvelteKit build inside a native window.
// Phase 2 added the typed capability layer (notify is wired end-to-end).
// Phase 3 registered Ollama supervision commands.
// Phase 4 (this file's current state) wires native affordances:
//   tauri-plugin-window-state    — remember size/position across launches
//   tauri-plugin-autostart       — launch June on login (opt-in via /settings)
//   tauri-plugin-global-shortcut — Cmd+Shift+J toggles window visibility
//   tray icon                    — menu-bar entry with Open/Quit + click-to-toggle
// Phase 5 (sidecar): auto-starts the bundled june-api brain on setup and
//   terminates it on exit. The window opens immediately; the API starts in
//   the background (the UI already shows a runtime badge while it warms up).

mod native;
mod ollama;
mod sidecar;

use native::{get_autostart, set_autostart};
use ollama::{
    bootstrap_ollama, is_model_pulled, is_ollama_installed, pull_model, start_ollama, OllamaState,
};
use sidecar::{get_api_token, start_api, stop_api, SidecarState, TokenState};
use std::sync::atomic::Ordering;
use std::time::Duration;
use tauri::Manager;
use tauri_plugin_autostart::MacosLauncher;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            // No CLI args on autostart; the shell opens to whatever route the
            // window-state plugin restored.
            Some(vec![]),
        ))
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| native::handle_hotkey(app, event.state))
                .build(),
        )
        .manage(OllamaState::default())
        .manage(SidecarState::default())
        .setup(|app| {
            let handle = app.handle();
            native::install_tray(handle)?;
            native::register_hotkey(handle)?;

            // Load the persistent loopback token once, before spawning the
            // sidecar, and register it in Tauri state. start_api sets it on the
            // sidecar's JUNE_API_TOKEN env and the get_api_token command hands
            // the same value to the webview. The token is stored 0600 in the
            // app-config dir and reused across launches so that if a june-api
            // from a previous launch is still listening, the webview's token
            // still matches it (see load_or_create_token).
            app.manage(TokenState(sidecar::load_or_create_token(handle)));

            // Start the june-api sidecar in the background so the window opens
            // immediately without blocking. The UI runtime badge reflects
            // readiness; the sidecar logs outcome to stderr.
            let api_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                match start_api(api_handle).await {
                    Ok(()) => eprintln!("[june-desktop] june-api sidecar is ready"),
                    Err(e) => eprintln!("[june-desktop] june-api sidecar error: {e}"),
                }
            });

            // Watchdog: polls every 5 s and restarts the sidecar if /healthz is
            // unreachable AFTER it was healthy at least once (a genuine crash).
            // Reaps the dead child handle first (no zombie). Backs off
            // exponentially on repeated failures (5s * 2^n, capped at 60 s) but
            // never gives up permanently — self-heals when the cause clears.
            // Terminates cleanly when shutting_down is set so app quit is clean.
            let watchdog_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut interval = Duration::from_secs(5);
                let mut failures: u32 = 0;
                // Only restart once the sidecar has been reachable at least once.
                // During the initial cold start the first start_api is still
                // health-waiting, and we must not reap/restart the child it just
                // spawned — that would kill and respawn the sidecar on every launch.
                let mut ever_healthy = false;
                loop {
                    tokio::time::sleep(interval).await;
                    // Check before doing any I/O.
                    if watchdog_handle
                        .state::<SidecarState>()
                        .shutting_down
                        .load(Ordering::SeqCst)
                    {
                        break;
                    }
                    if sidecar::api_reachable(sidecar::API_PORT).await {
                        ever_healthy = true;
                        failures = 0;
                        interval = Duration::from_secs(5);
                        continue;
                    }
                    // Sidecar is down — re-check the flag before restarting to
                    // avoid racing a quit that happened while we probed.
                    if watchdog_handle
                        .state::<SidecarState>()
                        .shutting_down
                        .load(Ordering::SeqCst)
                    {
                        break;
                    }
                    // Don't fight the initial cold start: only restart a sidecar
                    // that was reachable at least once (a genuine crash), never
                    // one that simply hasn't finished starting yet.
                    if !ever_healthy {
                        continue;
                    }
                    sidecar::reap_dead_child(&watchdog_handle).await;
                    match start_api(watchdog_handle.clone()).await {
                        Ok(()) => {
                            failures = 0;
                            interval = Duration::from_secs(5);
                            eprintln!("[june-desktop] watchdog: june-api restarted successfully");
                        }
                        Err(e) => {
                            failures += 1;
                            // 5s * 2^failures, capped at 60 s.
                            let exp = failures.min(10) as u64;
                            let secs = (5u64 * (1u64 << exp)).min(60);
                            interval = Duration::from_secs(secs);
                            eprintln!(
                                "[june-desktop] watchdog: restart failed ({e}); \
                                 retrying in {secs}s (failure #{failures})"
                            );
                        }
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            is_ollama_installed,
            start_ollama,
            is_model_pulled,
            pull_model,
            bootstrap_ollama,
            get_autostart,
            set_autostart,
            stop_api,
            get_api_token,
        ])
        .build(tauri::generate_context!())
        .expect("error building June desktop application");

    // Run the event loop. On exit, terminate the supervised june-api child so
    // no orphaned process survives the app quit. SidecarState uses
    // std::sync::Mutex (never held across an await point), so locking here is
    // a plain sync call; start_kill() sends SIGKILL without awaiting exit.
    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            let state = app_handle.state::<SidecarState>();
            // Signal the watchdog to stop BEFORE killing the child so it can
            // never respawn during quit. SeqCst ensures the store is visible to
            // the watchdog task on all platforms.
            state.shutting_down.store(true, Ordering::SeqCst);
            // Semicolon makes this a statement so the Result<MutexGuard, _>
            // temporary is dropped before `state` goes out of scope.
            //
            // SIGKILL (start_kill) rather than a graceful SIGTERM is acceptable
            // here: june-api runs SQLite in WAL mode, which is crash-safe — a
            // hard kill mid-write leaves at most an uncheckpointed WAL that is
            // replayed on next open, never a corrupt DB. The startup corrupt-DB
            // recovery is a further backstop. A graceful shutdown from this sync
            // exit handler would require awaiting the child, which we cannot do
            // here without blocking quit.
            if let Ok(mut guard) = state.child.lock() {
                if let Some(ref mut child) = *guard {
                    let _ = child.start_kill();
                }
            };
        }
    });
}
