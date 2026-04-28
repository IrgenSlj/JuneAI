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

mod native;
mod ollama;

use native::{get_autostart, set_autostart};
use ollama::{
    bootstrap_ollama, is_model_pulled, is_ollama_installed, pull_model, start_ollama, OllamaState,
};
use tauri_plugin_autostart::MacosLauncher;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
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
        .setup(|app| {
            let handle = app.handle();
            native::install_tray(handle)?;
            native::register_hotkey(handle)?;
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running June desktop application");
}
