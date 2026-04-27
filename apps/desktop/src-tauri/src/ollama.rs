// Ollama supervision — Phase 3 of the desktop shell plan.
//
// Five Tauri commands fulfil the Platform interface defined in
// packages/ui/src/platform/types.ts:
//   - is_ollama_installed     : binary present on PATH?
//   - start_ollama            : spawn `ollama serve` and wait for it to answer
//   - is_model_pulled         : ask Ollama's /api/tags
//   - pull_model              : stream /api/pull progress as Tauri events
//   - bootstrap_ollama        : open the official installer download URL
//
// The supervised child handle lives in `OllamaState` so that we can terminate
// it on app exit. Bootstrap stays "open the download page" in this phase
// because the official installers want to run with their own UX (Gatekeeper
// dialog on macOS, UAC prompt on Windows). Phase 3.5 may revisit and download
// + extract in-process if real users hit friction with the OS hand-off.

use futures_util::StreamExt;
use serde::Serialize;
use std::process::Stdio;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

const OLLAMA_BASE: &str = "http://127.0.0.1:11434";
const PULL_PROGRESS_EVENT: &str = "ollama-pull-progress";
const INSTALL_PROGRESS_EVENT: &str = "ollama-install-progress";

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub fraction: Option<f64>,
    pub status: String,
}

#[derive(Default)]
pub struct OllamaState {
    pub child: Mutex<Option<Child>>,
}

#[tauri::command]
pub async fn is_ollama_installed() -> Result<bool, String> {
    Ok(which::which("ollama").is_ok())
}

#[tauri::command]
pub async fn start_ollama(app: AppHandle) -> Result<(), String> {
    if reachable().await {
        return Ok(());
    }
    let bin = which::which("ollama").map_err(|_| {
        "Ollama isn't installed. Click \"Install Ollama\" first.".to_string()
    })?;

    let mut cmd = Command::new(bin);
    cmd.arg("serve").stdout(Stdio::null()).stderr(Stdio::null());
    let child = cmd.spawn().map_err(|e| format!("Couldn't spawn ollama: {e}"))?;

    let state = app.state::<OllamaState>();
    {
        let mut guard = state.child.lock().await;
        *guard = Some(child);
    }

    // Wait up to ~10s for the server to answer; this matches the brain's
    // tolerance for an Ollama cold start on a fresh machine.
    for _ in 0..40 {
        if reachable().await {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    Err("Ollama process started but did not become reachable within 10 seconds.".into())
}

#[tauri::command]
pub async fn is_model_pulled(tag: String) -> Result<bool, String> {
    let url = format!("{OLLAMA_BASE}/api/tags");
    let res = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map_err(|e| format!("Couldn't reach Ollama: {e}"))?;
    let json: serde_json::Value = res
        .json()
        .await
        .map_err(|e| format!("Couldn't parse /api/tags: {e}"))?;
    let models = json
        .get("models")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();
    Ok(models.iter().any(|m| {
        m.get("name")
            .and_then(|n| n.as_str())
            .map(|n| n == tag)
            .unwrap_or(false)
    }))
}

#[tauri::command]
pub async fn pull_model(app: AppHandle, tag: String) -> Result<(), String> {
    let url = format!("{OLLAMA_BASE}/api/pull");
    let body = serde_json::json!({ "name": tag, "stream": true });

    let res = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Couldn't reach Ollama: {e}"))?;

    if !res.status().is_success() {
        return Err(format!(
            "Ollama refused the pull request: HTTP {}",
            res.status()
        ));
    }

    let mut stream = res.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut last_status = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Pull stream error: {e}"))?;
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let trimmed = trim_newline(&line);
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(trimmed) {
                if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                    return Err(err.to_string());
                }
                let status = v
                    .get("status")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();
                let total = v.get("total").and_then(|t| t.as_u64());
                let completed = v.get("completed").and_then(|t| t.as_u64());
                let fraction = match (total, completed) {
                    (Some(t), Some(c)) if t > 0 => Some((c as f64 / t as f64).clamp(0.0, 1.0)),
                    _ => None,
                };
                last_status = status.clone();
                let _ = app.emit(
                    PULL_PROGRESS_EVENT,
                    Progress { fraction, status },
                );
            }
        }
    }

    // Ollama emits a final {"status":"success"} when the pull completes.
    if last_status != "success" {
        return Err(format!(
            "Pull stream ended without success (last status: {last_status})"
        ));
    }
    Ok(())
}

#[tauri::command]
pub async fn bootstrap_ollama(app: AppHandle) -> Result<(), String> {
    use tauri_plugin_shell::ShellExt;

    let url = if cfg!(target_os = "macos") {
        "https://ollama.com/download/Ollama-darwin.zip"
    } else if cfg!(target_os = "windows") {
        "https://ollama.com/download/OllamaSetup.exe"
    } else {
        "https://ollama.com/download/linux"
    };

    let _ = app.emit(
        INSTALL_PROGRESS_EVENT,
        Progress {
            fraction: None,
            status: "Opening installer download in your browser…".to_string(),
        },
    );

    app.shell()
        .open(url, None)
        .map_err(|e| format!("Couldn't open installer URL: {e}"))?;

    let _ = app.emit(
        INSTALL_PROGRESS_EVENT,
        Progress {
            fraction: None,
            status: "Run the installer, then return here and click \"Start Ollama\".".to_string(),
        },
    );
    Ok(())
}

async fn reachable() -> bool {
    reqwest::Client::new()
        .get(format!("{OLLAMA_BASE}/api/tags"))
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn trim_newline(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    while end > 0 && (line[end - 1] == b'\n' || line[end - 1] == b'\r') {
        end -= 1;
    }
    &line[..end]
}
