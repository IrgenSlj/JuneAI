// Tauri entry point for the June desktop shell.
//
// Phase 1 scaffold opened the SvelteKit build inside a native window.
// Phase 2 added the typed capability layer (notify is wired end-to-end).
// Phase 3 (this file's current state) registers the Ollama supervision
// commands that close the gap /help/ollama leaves on the web.

mod ollama;

use ollama::{
    bootstrap_ollama, is_model_pulled, is_ollama_installed, pull_model, start_ollama, OllamaState,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .manage(OllamaState::default())
        .invoke_handler(tauri::generate_handler![
            is_ollama_installed,
            start_ollama,
            is_model_pulled,
            pull_model,
            bootstrap_ollama,
        ])
        .run(tauri::generate_context!())
        .expect("error while running June desktop application");
}
