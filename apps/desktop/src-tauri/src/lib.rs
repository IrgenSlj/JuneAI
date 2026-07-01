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
use sidecar::{start_api, stop_api, SidecarState};
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
            // Semicolon makes this a statement so the Result<MutexGuard, _>
            // temporary is dropped before `state` goes out of scope.
            if let Ok(mut guard) = state.child.lock() {
                if let Some(ref mut child) = *guard {
                    let _ = child.start_kill();
                }
            };
        }
    });
}
