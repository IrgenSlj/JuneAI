// The June Mac app: one window around the June daemon.
//
// On launch the window shows a small boot page, the daemon starts (see
// sidecar.rs), and the window navigates to the daemon's own URL with the
// per-launch token in the fragment. From then on the window is just a browser
// pointed at http://127.0.0.1:<port>, exactly like the browser mode, so both
// are tested by the same Playwright suite. The only native features the page
// may use are the folder picker and opening links (capabilities/default.json).

mod sidecar;

use sidecar::DaemonState;
use tauri::{Manager, Url};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .manage(DaemonState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            let config_dir = app.path().app_config_dir()?;
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<DaemonState>();
                let window = handle.get_webview_window("main").expect("main window");
                match sidecar::start(&state, config_dir).await {
                    Ok(daemon) => {
                        let url = format!("http://127.0.0.1:{}/#token={}", daemon.port, daemon.token);
                        if let Err(e) = Url::parse(&url).map_err(|e| e.to_string()).and_then(|u| window.navigate(u).map_err(|e| e.to_string())) {
                            eprintln!("[june-desktop] could not open the UI: {e}");
                        }
                    }
                    Err(e) => {
                        eprintln!("[june-desktop] daemon failed to start: {e}");
                        let message = serde_json::to_string(&format!("June could not start: {e}")).unwrap_or_default();
                        let _ = window.eval(&format!("document.body.textContent = {message}"));
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error building June");

    app.run(|handle, event| {
        if let tauri::RunEvent::Exit = event {
            sidecar::stop(&handle.state::<DaemonState>());
        }
    });
}
