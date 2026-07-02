// June API sidecar supervision.
//
// Mirrors the start_ollama pattern in ollama.rs: a state struct holding the
// child handle, an api_reachable() health probe, spawn on setup, health-wait
// up to 60 s (PyInstaller frozen cold start is 13–30 s per the spike), and a
// kill on exit.
//
// Fixed port: 8000. The SvelteKit static bundle bakes PUBLIC_JUNE_API_URL at
// desktop build time; the default (no env override) is http://localhost:8000
// (see apps/web/src/lib/api.ts line 4). Port 8000 keeps dev and packaged
// builds consistent without needing a runtime port-injection mechanism.
//
// TODO(Ship-2b): Add the june-api PyInstaller onedir directory to
// tauri.conf.json as a `resources` entry so it lands at
// June.app/Contents/Resources/june-api/ and the path resolution below
// succeeds in a packaged build. Also wire JUNE_BUILD_SHA into the spawn env
// (baked by the packaging CI step). Until Ship-2b lands, the early-return in
// start_api() handles the dev case where june-api is already running
// independently.

use std::env;
use std::process::Stdio;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};
use std::time::Duration;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Manager};
use tokio::process::{Child, Command};

/// Fixed loopback port for the june-api sidecar. Must match the default
/// PUBLIC_JUNE_API_URL baked into the SvelteKit desktop bundle:
/// http://localhost:8000 (apps/web/src/lib/api.ts).
pub const API_PORT: u16 = 8000;

/// Poll budget for the health wait: 120 × 500 ms = 60 s total.
/// The PyInstaller frozen cold start is 13–30 s (spike findings); 60 s gives
/// a 2× margin. Do NOT reuse Ollama's 10 s budget.
const HEALTH_POLL_ITERS: u32 = 120;
const HEALTH_POLL_MS: u64 = 500;

#[derive(Default)]
pub struct SidecarState {
    pub child: Mutex<Option<Child>>,
    /// Set to true in the RunEvent::Exit handler before killing the child so
    /// the watchdog never respawns during quit.
    pub shutting_down: AtomicBool,
}

/// Loopback shared-secret. Loaded once at startup (lib.rs setup), set on the
/// sidecar's `JUNE_API_TOKEN` env, and handed to the webview through the
/// `get_api_token` command so it can send `X-June-Token` on every request.
pub struct TokenState(pub String);

/// Load the persistent per-install loopback token, or create it on first run.
///
/// Stored 0600 in the Tauri app-config dir so it is STABLE across launches.
/// A per-launch random token would break the adopt-existing-server path in
/// `start_api`: if a june-api from a previous launch is still listening on
/// `API_PORT`, this instance early-returns without re-spawning, but a fresh
/// per-launch token would be one the surviving server never saw, so every
/// webview request would 401. A stable token means an adopted server shares
/// the same secret.
///
/// Degrades to a fresh random token (this launch only) if the config dir can't
/// be read or written, so startup never fails on a filesystem hiccup.
pub fn load_or_create_token(app: &AppHandle) -> String {
    let fresh = || uuid::Uuid::new_v4().to_string();
    let Ok(dir) = app.path().app_config_dir() else {
        return fresh();
    };
    let token_path = dir.join("api-token");
    if let Ok(existing) = std::fs::read_to_string(&token_path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let token = fresh();
    if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&token_path, &token).is_ok() {
        // Best-effort: tighten to owner-only so other users can't read the
        // shared secret. The process env still exposes it to same-user
        // processes (accepted for a single-user local app).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ =
                std::fs::set_permissions(&token_path, std::fs::Permissions::from_mode(0o600));
        }
    }
    token
}

/// Returns true if the june-api is already answering on `port`.
/// Probes /healthz — the one path the loopback-token middleware always exempts,
/// so the health-wait still succeeds once JUNE_API_TOKEN enforcement is active
/// (a token-less probe of /system would 401 and the wait would never resolve).
pub async fn api_reachable(port: u16) -> bool {
    reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/healthz"))
        .timeout(Duration::from_secs(1))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Return the per-launch loopback token so the webview can send it as
/// `X-June-Token`. Registered in lib.rs `generate_handler!`.
#[tauri::command]
pub fn get_api_token(state: tauri::State<'_, TokenState>) -> String {
    state.0.clone()
}

/// Start the bundled june-api sidecar. If the API is already reachable (e.g.
/// a dev `june-api` process is running separately), returns Ok(()) immediately
/// without spawning — same early-return pattern as start_ollama.
pub async fn start_api(app: AppHandle) -> Result<(), String> {
    if api_reachable(API_PORT).await {
        return Ok(());
    }

    // Resolve the bundled PyInstaller onedir executable path.
    // The onedir ships as a resource directory; the entry-point executable is
    // named "june-api" inside the "june-api/" subdirectory.
    // TODO(Ship-2b): the resource directory only exists after the onedir is
    // added to tauri.conf.json `resources`. Path resolution will return an
    // error until then (handled gracefully below).
    let binary_path = app
        .path()
        .resolve("june-api/june-api", BaseDirectory::Resource)
        .map_err(|e| format!("Could not resolve june-api resource path: {e}"))?;

    // The per-launch loopback token registered in lib.rs setup. Setting it here
    // turns on the sidecar's X-June-Token enforcement; the webview reads the
    // same value via get_api_token and sends it on every request.
    let token = app.state::<TokenState>().0.clone();

    let mut cmd = Command::new(&binary_path);
    cmd.env("JUNE_API_HOST", "127.0.0.1")
        .env("JUNE_API_PORT", API_PORT.to_string())
        .env("JUNE_API_TOKEN", &token)
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // Forward JUNE_DATA_DIR from the parent environment if set.
    if let Ok(data_dir) = env::var("JUNE_DATA_DIR") {
        cmd.env("JUNE_DATA_DIR", data_dir);
    }

    let mut child = cmd.spawn().map_err(|e| {
        format!(
            "Failed to spawn june-api at {}: {e}",
            binary_path.display()
        )
    })?;

    {
        let state = app.state::<SidecarState>();
        let mut guard = state.child.lock().unwrap();
        // A quit may have begun between the watchdog's pre-spawn shutting_down
        // check and this spawn. Re-check under the child lock: if we're shutting
        // down, the RunEvent::Exit handler has already run (it found child=None)
        // or is blocked on this very lock. Either way, do NOT store the child —
        // kill it now so it cannot be orphaned past app quit. start_kill() is
        // synchronous (SIGKILL, no await), so it is safe under the lock.
        if state.shutting_down.load(Ordering::SeqCst) {
            let _ = child.start_kill();
            return Err("shutdown in progress; killed just-spawned june-api".to_string());
        }
        *guard = Some(child);
    }

    // Poll until reachable or the budget is exhausted.
    for _ in 0..HEALTH_POLL_ITERS {
        if api_reachable(API_PORT).await {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(HEALTH_POLL_MS)).await;
    }

    Err(format!(
        "june-api started but did not become reachable on port {API_PORT} within {}s.",
        (HEALTH_POLL_ITERS as u64 * HEALTH_POLL_MS) / 1000
    ))
}

/// Take and reap the dead child handle before a watchdog respawn.
///
/// The std MutexGuard is dropped inside the scoped block; `child.wait().await`
/// is called after the guard is gone so we never hold a !Send guard across an
/// await point (same pattern as `stop_api`).
pub async fn reap_dead_child(app: &AppHandle) {
    let old = {
        let state = app.state::<SidecarState>();
        let mut guard = state.child.lock().unwrap();
        guard.take()
    }; // guard dropped here — safe to await below
    if let Some(mut child) = old {
        let _ = child.start_kill(); // no-op if already dead
        let _ = child.wait().await; // reap so no zombie lingers
    }
}

/// Kill the supervised june-api child, if one was spawned.
/// Exposed as a Tauri command so the frontend can request a clean shutdown;
/// the RunEvent::Exit handler in lib.rs also terminates the child on app quit.
#[tauri::command]
pub async fn stop_api(state: tauri::State<'_, SidecarState>) -> Result<(), String> {
    // Extract the child inside a sync block so the MutexGuard is dropped
    // before the .await below — std::sync::MutexGuard is not Send.
    let child = {
        let mut guard = state.child.lock().unwrap();
        guard.take()
    };
    if let Some(mut child) = child {
        child
            .kill()
            .await
            .map_err(|e| format!("Failed to kill june-api: {e}"))?;
    }
    Ok(())
}
