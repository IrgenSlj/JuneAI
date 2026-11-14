// Runs the June daemon (`june serve`, Node) as a child of the app.
//
// - A fresh token per launch, passed in the environment (never argv, which
//   other local users can read). Nothing is written to disk.
// - The daemon picks a free port and reports it on its first stdout line:
//   `JUNE_READY {"port":53121}`. No fixed port, no adopting whatever answers.
// - We hold the daemon's stdin open and it runs with --exit-with-stdin, so if
//   this app dies in any way, the pipe closes and the daemon exits with it.
//
// Development builds run the `node` on PATH against the repo's built files.
// Release builds will run a bundled official Node binary; that packaging is
// gate work (PLAN.md, "Daemon rules"), with signing and notarisation.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};

pub struct Daemon {
    pub port: u16,
    pub token: String,
}

/// Keeps the child and its stdin alive for the life of the app. Dropping the
/// stdin handle would close the pipe and stop the daemon.
#[derive(Default)]
pub struct DaemonState {
    pub child: Mutex<Option<(Child, ChildStdin)>>,
}

struct Paths {
    node: PathBuf,
    server: PathBuf,
    web: PathBuf,
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn paths() -> Result<Paths, String> {
    #[cfg(debug_assertions)]
    {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        Ok(Paths {
            node: env_path("JUNE_NODE").unwrap_or_else(|| PathBuf::from("node")),
            server: env_path("JUNE_SERVER_BUNDLE").unwrap_or_else(|| repo.join("apps/server/dist/server.mjs")),
            web: env_path("JUNE_WEB_DIR").unwrap_or_else(|| repo.join("apps/web/build")),
        })
    }
    #[cfg(not(debug_assertions))]
    {
        Err("release packaging of the Node sidecar is not built yet (gate work)".into())
    }
}

/// A 244-bit random token from two v4 UUIDs, hex without dashes.
fn new_token() -> String {
    format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple())
}

pub async fn start(state: &DaemonState, config_dir: PathBuf) -> Result<Daemon, String> {
    let paths = paths()?;
    let token = new_token();
    let mut child = Command::new(&paths.node)
        .arg(&paths.server)
        .arg("serve")
        .arg("--exit-with-stdin")
        .arg("--static")
        .arg(&paths.web)
        .arg("--config-dir")
        .arg(&config_dir)
        .env("JUNE_TOKEN", &token)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("could not start {} {}: {e}", paths.node.display(), paths.server.display()))?;

    let stdin = child.stdin.take().ok_or("daemon has no stdin")?;
    let stdout = child.stdout.take().ok_or("daemon has no stdout")?;
    let mut lines = BufReader::new(stdout).lines();

    let port = tokio::time::timeout(Duration::from_secs(20), async {
        while let Some(line) = lines.next_line().await.map_err(|e| e.to_string())? {
            if let Some(json) = line.strip_prefix("JUNE_READY ") {
                let value: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
                let port = value["port"].as_u64().ok_or("JUNE_READY without a port")?;
                return u16::try_from(port).map_err(|e| e.to_string());
            }
        }
        Err("daemon exited before it was ready".to_string())
    })
    .await
    .map_err(|_| "daemon did not report ready within 20 s".to_string())??;

    // Keep reading stdout so the pipe never fills and blocks the daemon.
    tauri::async_runtime::spawn(async move {
        while let Ok(Some(line)) = lines.next_line().await {
            eprintln!("[june] {line}");
        }
    });

    *state.child.lock().map_err(|e| e.to_string())? = Some((child, stdin));
    Ok(Daemon { port, token })
}

/// Stop the daemon on quit: closing stdin asks it to exit; the kill makes sure.
pub fn stop(state: &DaemonState) {
    if let Ok(mut guard) = state.child.lock() {
        if let Some((mut child, stdin)) = guard.take() {
            drop(stdin);
            let _ = child.start_kill();
        }
    }
}
