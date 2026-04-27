// Tauri entry point for the June desktop shell.
//
// Phase 1 scaffold: this only opens the existing SvelteKit build inside a
// native window. No Rust commands are exposed yet. Phase 2 introduces the
// capability layer in `packages/ui/src/platform.ts` and registers the first
// `invoke` handlers here. Phase 3 adds Ollama supervision (see ADR 0008).

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .run(tauri::generate_context!())
        .expect("error while running June desktop application");
}
