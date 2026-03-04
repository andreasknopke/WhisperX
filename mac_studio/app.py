"""
WhisperX Backend – Mac Studio M3 Ultra Edition
===============================================
Optimiert für Apple Silicon (MPS), 30 gleichzeitige Verbindungen,
und optionale LLM-Nachkontrolle via Ollama/MLX.
"""

import gradio as gr
import torch
import os
import shutil
import gc
import sys
import signal
import subprocess
from dotenv import load_dotenv
from src.transcriber import transcribe_audio, LANGUAGE_OPTIONS
from src.model_manager import MODELS, model_pool, DEFAULT_POOL_SIZE
from src.llm_client import LLMClient

# --- UMGEBUNGSVARIABLEN LADEN ---
load_dotenv()
AUTH_USER = os.getenv("WHISPER_AUTH_USERNAME", "admin")
AUTH_PASS = os.getenv("WHISPER_AUTH_PASSWORD", "password123")

# --- APPLE SILICON OPTIMIERUNGEN ---
# Aktiviert MPS Fallback für nicht-unterstützte Ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Optimiert Metal Shader Compilation
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"

# LLM Client initialisieren (optional)
llm_client = LLMClient()


def get_device_info() -> str:
    """Erkennt das verfügbare Compute-Backend."""
    if torch.backends.mps.is_available():
        return "mps", "🍎 Apple Silicon (MPS) Aktiv"
    elif torch.cuda.is_available():
        return "cuda", "🟢 CUDA Aktiv"
    else:
        return "cpu", "🔴 CPU Modus"


def cleanup_temp():
    """Löscht Gradio-Temp-Dateien beim Start."""
    temp_candidates = [
        os.path.join(os.environ.get('TMPDIR', '/tmp'), 'gradio'),
        os.path.join('/tmp', 'gradio'),
    ]
    for temp_dir in temp_candidates:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"--- System: Gradio Temp-Ordner bereinigt: {temp_dir} ---")
            except Exception:
                pass


def remote_restart():
    """Startet die App neu (macOS-kompatibel)."""
    try:
        print("--- SYSTEM: Neustart via API/UI ausgelöst ---")
        # macOS: Neuen Prozess spawnen und aktuellen beenden
        subprocess.Popen([sys.executable] + sys.argv)
        os._exit(0)
    except Exception as e:
        return f"Fehler: {str(e)}"


def clear_gpu_memory():
    """Erzwingt Speicherbereinigung (Unified Memory)."""
    try:
        gc.collect()
        if torch.backends.mps.is_available():
            # MPS Cache leeren
            torch.mps.empty_cache()
            return "MPS Cache + RAM erfolgreich bereinigt."
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            return "VRAM/Cache erfolgreich bereinigt."
        return "Nur RAM bereinigt (CPU-Mode)."
    except Exception as e:
        return f"Fehler: {str(e)}"


def kill_python_zombies():
    """Beendet hängende Python-Prozesse (macOS-kompatibel)."""
    try:
        current_pid = os.getpid()
        # macOS: pgrep + kill statt taskkill
        result = subprocess.run(
            ["pgrep", "-f", "python"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.strip().split('\n') if p.strip()]
        killed = 0
        for pid in pids:
            if pid != current_pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed += 1
                except ProcessLookupError:
                    pass
        return f"{killed} andere Python-Instanz(en) beendet."
    except Exception as e:
        return f"Fehler: {str(e)}"


def get_pool_status():
    """Zeigt den Status des Worker-Pools."""
    available = model_pool.get_available_count()
    total = model_pool.pool_size
    queue_size = model_pool.get_queue_size()
    return (
        f"Worker: {available}/{total} verfügbar | "
        f"Queue: {queue_size} wartend | "
        f"LLM: {'✅ Verbunden' if llm_client.is_available() else '❌ Nicht verfügbar'}"
    )


def transcribe_with_llm_review(file_path, language, model_name, device,
                                initial_prompt_user="", speed_mode=False,
                                llm_review=False):
    """Transkription mit optionaler LLM-Nachkontrolle."""
    # Reguläre Transkription
    text, json_data, timing = transcribe_audio(
        file_path, language, model_name, device,
        initial_prompt_user, speed_mode
    )

    # Optionale LLM-Nachkontrolle
    if llm_review and text and llm_client.is_available():
        try:
            reviewed_text = llm_client.review_transcription(
                text, language=language
            )
            timing_parts = timing.rsplit(" in ", 1)
            timing = f"{timing_parts[0]} + LLM-Review in {timing_parts[1]}" if len(timing_parts) == 2 else timing + " + LLM-Review"
            return [reviewed_text, json_data, timing]
        except Exception as e:
            print(f"--- LLM Review fehlgeschlagen: {e} ---")
            return [text, json_data, timing + " (LLM-Review fehlgeschlagen)"]

    return [text, json_data, timing]


def build_interface():
    default_device, device_label = get_device_info()

    with gr.Blocks(theme=gr.themes.Soft(), title="WhisperX Backend – Mac Studio") as interface:
        gr.Markdown("# 🎙️ WhisperX Schreibdienst Backend – Mac Studio Edition")
        gr.Markdown(
            f"Status: {device_label} | "
            f"Pool: {model_pool.pool_size} Worker | "
            f"Max Queue: {model_pool.pool_size * 5} | "
            f"Device-Remap: cuda→{default_device}"
        )

        with gr.Tabs():
            # TAB 1: TRANSKRIPTION
            with gr.TabItem("Transkription"):
                with gr.Row():
                    with gr.Column(variant="panel"):
                        gr.Markdown("### ⚙️ Konfiguration")
                        file_input = gr.File(label="Audio/Video Datei")
                        language_dropdown = gr.Dropdown(
                            choices=list(LANGUAGE_OPTIONS.keys()),
                            label="Sprache", value="Identify"
                        )
                        model_dropdown = gr.Dropdown(
                            choices=MODELS, label="Modell",
                            value="large-v3", allow_custom_value=True
                        )
                        device_dropdown = gr.Dropdown(
                            choices=["mps", "cpu"] if torch.backends.mps.is_available()
                            else ["cuda", "cpu"] if torch.cuda.is_available()
                            else ["cpu"],
                            label="Gerät", value=default_device
                        )
                        initial_prompt_input = gr.Textbox(
                            label="Initial Prompt", lines=3
                        )
                        speed_mode_input = gr.Checkbox(
                            label="Speed Mode", value=False, visible=False
                        )
                        llm_review_input = gr.Checkbox(
                            label="🤖 LLM-Nachkontrolle (Mistral)",
                            value=False,
                            info="Transkription wird durch Mistral Small gegengelesen"
                        )
                        transcribe_button = gr.Button(
                            "▶️ Transkription Starten", variant="primary"
                        )

                    with gr.Column(variant="panel"):
                        gr.Markdown("### 📝 Auswertung")
                        time_output = gr.Textbox(label="Bearbeitungszeit / Status")
                        text_output = gr.TextArea(
                            label="Text-Ergebnis", lines=12,
                            show_copy_button=True
                        )
                        json_output = gr.TextArea(
                            label="JSON Daten", visible=False
                        )

            # TAB 2: SYSTEM ADMIN
            with gr.TabItem("🔧 System Admin"):
                gr.Markdown("### 🛠️ Hardware-Management (Apple Silicon M3 Ultra)")
                with gr.Row():
                    btn_vram = gr.Button("🧹 Memory leeren")
                    btn_zombie = gr.Button("💀 Zombies killen")
                    btn_reboot = gr.Button("🔄 Server Neustart")
                    btn_status = gr.Button("📊 Pool Status")

                admin_status = gr.Textbox(label="Aktion-Ergebnis", interactive=False)

                btn_vram.click(
                    fn=clear_gpu_memory, outputs=admin_status,
                    api_name="system_cleanup"
                )
                btn_zombie.click(
                    fn=kill_python_zombies, outputs=admin_status,
                    api_name="system_kill_zombies"
                )
                btn_reboot.click(
                    fn=remote_restart, outputs=admin_status,
                    api_name="system_reboot"
                )
                btn_status.click(
                    fn=get_pool_status, outputs=admin_status,
                    api_name="system_pool_status"
                )

        # Event-Handler Transkription
        transcribe_button.click(
            fn=transcribe_with_llm_review,
            inputs=[
                file_input, language_dropdown, model_dropdown,
                device_dropdown, initial_prompt_input, speed_mode_input,
                llm_review_input
            ],
            outputs=[text_output, json_output, time_output],
            api_name="start_process",
            concurrency_limit=model_pool.pool_size * 5,  # Queue-Tiefe = 5× Worker-Anzahl
        )

    return interface


if __name__ == "__main__":
    cleanup_temp()

    # Backend-Info
    device, label = get_device_info()
    print(f"--- SYSTEM: {label} ---")
    print(f"--- SYSTEM: Worker Pool: {model_pool.pool_size} Instanzen ---")
    print(f"--- SYSTEM: LLM Client: {'Aktiv' if llm_client.is_available() else 'Inaktiv'} ---")

    # MPS Vor-Check
    if torch.backends.mps.is_available():
        # Warm-up: kleinen Tensor auf MPS allozieren um Backend zu initialisieren
        _ = torch.zeros(1, device="mps")
        print("--- SYSTEM: MPS Backend initialisiert ---")

    interface = build_interface()

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=(AUTH_USER, AUTH_PASS),
        auth_message="WhisperX Backend Login",
        max_threads=model_pool.pool_size * 7,  # ~7 Threads pro Worker (Headroom)
    )
