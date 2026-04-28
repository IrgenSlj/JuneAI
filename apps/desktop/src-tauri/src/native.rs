// Native affordances — Phase 4 of the desktop shell plan.
//
// Three Rust-side responsibilities live here:
//   - tray:     menu-bar icon with Open/Quit menu, click toggles the window
//   - hotkey:   global Cmd+Shift+J that toggles window visibility from anywhere
//   - autostart: thin wrappers around tauri-plugin-autostart so the UI can
//                expose a single toggle in /settings
//
// Window state persistence is handled entirely by tauri-plugin-window-state
// in lib.rs and needs nothing from this module.

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager, Runtime,
};
use tauri_plugin_autostart::ManagerExt;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

/// Hotkey is hard-coded for v1. Phase 4.5 may make it configurable in /settings.
pub const HOTKEY_LABEL: &str = "Cmd+Shift+J (Ctrl+Shift+J on Windows/Linux)";

pub fn build_hotkey() -> Shortcut {
    Shortcut::new(Some(Modifiers::SUPER | Modifiers::SHIFT), Code::KeyJ)
}

pub fn install_tray<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    let open_item = MenuItem::with_id(app, "tray-open", "Open June", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "tray-quit", "Quit June", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open_item, &quit_item])?;

    let icon = app
        .default_window_icon()
        .cloned()
        .ok_or_else(|| tauri::Error::AssetNotFound("default window icon".into()))?;

    TrayIconBuilder::with_id("main-tray")
        .icon(icon)
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "tray-open" => focus_main_window(app),
            "tray-quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                focus_main_window(tray.app_handle());
            }
        })
        .build(app)?;
    Ok(())
}

pub fn register_hotkey<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    app.global_shortcut().register(build_hotkey())?;
    Ok(())
}

pub fn handle_hotkey<R: Runtime>(app: &AppHandle<R>, state: ShortcutState) {
    if state == ShortcutState::Pressed {
        toggle_main_window(app);
    }
}

fn focus_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn toggle_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        match window.is_visible() {
            Ok(true) => {
                let _ = window.hide();
            }
            _ => focus_main_window(app),
        }
    }
}

#[tauri::command]
pub async fn get_autostart<R: Runtime>(app: AppHandle<R>) -> Result<bool, String> {
    app.autolaunch()
        .is_enabled()
        .map_err(|e| format!("Couldn't read autostart state: {e}"))
}

#[tauri::command]
pub async fn set_autostart<R: Runtime>(app: AppHandle<R>, enabled: bool) -> Result<(), String> {
    let manager = app.autolaunch();
    if enabled {
        manager
            .enable()
            .map_err(|e| format!("Couldn't enable autostart: {e}"))
    } else {
        manager
            .disable()
            .map_err(|e| format!("Couldn't disable autostart: {e}"))
    }
}
