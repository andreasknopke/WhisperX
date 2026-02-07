import gradio as gr
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
import shutil
import gc
import sys
import subprocess
from dotenv import load_dotenv
from src.transcriber import transcribe_audio, LANGUAGE_OPTIONS
from src.model_manager import MODELS

# --- UMGEBUNGSVARIABLEN LADEN ---
load_dotenv()
# Wir nutzen die exakten Namen, die deine Web-App im process.env sucht
AUTH_USER = os.getenv("WHISPER_AUTH_USERNAME", "admin")
AUTH_PASS = os.getenv("WHISPER_AUTH_PASSWORD", "password123")



# --- CUDA KOEXISTENZ OPTIMIERUNG ---
# Verhindert, dass Torch den VRAM komplett reserviert (wichtig für V100 + LM-Studio)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_MODULE_LOADING"] = "LAZY"

def cleanup_temp():
    """Löscht Gradio-Temp-Dateien beim Start."""
    temp_dir = os.path.join(os.environ.get('TEMP', ''), 'gradio')
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"--- System: Gradio Temp-Ordner bereinigt ---")
        except: pass

def remote_restart():
    """Startet die App neu."""
    try:
        print("--- SYSTEM: Neustart via API/UI ausgelöst ---")
        os.startfile(sys.argv[0])
        os._exit(0)
    except Exception as e: return f"Fehler: {str(e)}"

def clear_gpu_memory():
    """Erzwingt VRAM Bereinigung."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            return "VRAM/Cache erfolgreich bereinigt."
        return "Nur RAM bereinigt (CPU-Mode)."
    except Exception as e: return f"Fehler: {str(e)}"

def kill_python_zombies():
    """Beendet hängende Python-Prozesse (außer diesen)."""
    try:
        current_pid = os.getpid()
        # Taskkill filtert den aktuellen Prozess aus, um Selbstmord zu verhindern
        cmd = f'taskkill /F /IM python.exe /FI "PID ne {current_pid}"'
        subprocess.run(cmd, shell=True, capture_output=True)
        return "Andere Python-Instanzen wurden beendet."
    except Exception as e: return f"Fehler: {str(e)}"

def build_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="WhisperX Backend") as interface:
        gr.Markdown("# 🎙️ WhisperX Schreibdienst Backend")
        
        # Status-Anzeige oben
        status_color = "green" if torch.cuda.is_available() else "red"
        gr.Markdown(f"Status: <span style='color:{status_color}'>{'CUDA Aktiv' if torch.cuda.is_available() else 'CPU Modus'}</span>")

        with gr.Tabs():
            # TAB 1: TRANSKRIPTION
            with gr.TabItem("Transkription"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        gr.Markdown("### ⚙️ Konfiguration")
                        file_input = gr.File(label="Audio/Video Datei")
                        language_dropdown = gr.Dropdown(choices=list(LANGUAGE_OPTIONS.keys()), label="Sprache", value="Identify")
                        model_dropdown = gr.Dropdown(choices=MODELS, label="Modell", value="large-v3", allow_custom_value=True)
                        device_dropdown = gr.Dropdown(choices=["cuda", "cpu"], label="Gerät", value="cuda" if torch.cuda.is_available() else "cpu")
                        initial_prompt_input = gr.Textbox(label="Initial Prompt", lines=3)
                        speed_mode_input = gr.Checkbox(label="Speed Mode", value=False, visible=False)
                        transcribe_button = gr.Button("▶️ Transkription Starten", variant="primary")

                    with gr.Column(variant="panel"):
                        gr.Markdown("### 📝 Auswertung")
                        time_output = gr.Textbox(label="Bearbeitungszeit / Status")
                        text_output = gr.TextArea(label="Text-Ergebnis", lines=12, show_copy_button=True)
                        json_output = gr.TextArea(label="JSON Daten", visible=False)

            # TAB 2: SYSTEM ADMIN (Für API-Heilung)
            with gr.TabItem("🔧 System Admin"):
                gr.Markdown("### 🛠️ Hardware-Management (Tesla V100)")
                with gr.Row():
                    btn_vram = gr.Button("🧹 VRAM leeren")
                    btn_zombie = gr.Button("💀 Zombies killen")
                    btn_reboot = gr.Button("🔄 Server Neustart")
                
                admin_status = gr.Textbox(label="Aktion-Ergebnis", interactive=False)

                # API-Endpunkte für deine Web-App
                btn_vram.click(fn=clear_gpu_memory, outputs=admin_status, api_name="system_cleanup")
                btn_zombie.click(fn=kill_python_zombies, outputs=admin_status, api_name="system_kill_zombies")
                btn_reboot.click(fn=remote_restart, outputs=admin_status, api_name="system_reboot")

        # Event-Handler Transkription
        transcribe_button.click(
            fn=transcribe_audio,
            inputs=[file_input, language_dropdown, model_dropdown, device_dropdown, initial_prompt_input, speed_mode_input],
            outputs=[text_output, json_output, time_output],
            api_name="start_process"
        )

    return interface

if __name__ == "__main__":
    cleanup_temp()
    
    # CUDA Vor-Check
    if torch.cuda.is_available():
        torch.cuda.init()

    interface = build_interface()
    
    # Start mit Login-Schutz
    # Die Web-App nutzt den POST auf /login mit username/password
    interface.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        auth=(AUTH_USER, AUTH_PASS),
        auth_message="WhisperX Backend Login"
    )