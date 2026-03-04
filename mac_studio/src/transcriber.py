"""
WhisperX Transcriber – Apple Silicon (MPS) Edition
===================================================
Angepasst für MPS-Backend mit Worker-Pool-Unterstützung
und optionaler LLM-Nachkontrolle.
"""

import whisperx
import torch
import os
import gc
import time
import json
from src.model_manager import model_pool
from src.utils import validate_multimedia_file, convert_to_wav, save_transcription, cleanup

# Standard-Instruktion für Whisper
FORMAT_PROMPT = "Klammern (so wie diese) und Satzzeichen wie Punkt, Komma, Doppelpunkt und Semikolon sind wichtig."

LANGUAGE_OPTIONS = {
    "Identify": None,
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it"
}


def clear_memory(device):
    """Speicher freigeben – MPS/CUDA/CPU-kompatibel."""
    gc.collect()
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def transcribe_audio(file_path, language, model_name, device,
                     initial_prompt_user="", speed_mode=False):
    """
    Transkribiert Audio/Video-Dateien mit WhisperX.

    Verwendet den Model-Pool für Thread-sichere parallele Verarbeitung.
    Identische API-Rückgabe wie die Windows-Version:
      [full_text, segments_json_string, timing_string]

    HINWEIS: Der Schreibdienst-Client sendet hardcoded device="cuda".
    Auf Apple Silicon wird dies transparent auf "mps" gemappt.
    """
    worker = None
    try:
        start_time = time.time()

        # Device-Remapping: Schreibdienst sendet "cuda", wir brauchen "mps"
        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
                print(f"--- SYSTEM: Device 'cuda' → 'mps' (Apple Silicon) ---")
            else:
                device = "cpu"
                print(f"--- SYSTEM: Device 'cuda' → 'cpu' (kein GPU-Backend) ---")

        combined_prompt = f"{FORMAT_PROMPT} {initial_prompt_user}".strip()

        # 1. Worker aus dem Pool holen (blockiert bis einer frei wird)
        worker = model_pool.acquire(model_name, device, timeout=120)
        model = worker.model

        # 2. KONFIGURATION
        b_size = 1 if speed_mode else 5

        if hasattr(model, "options"):
            try:
                if hasattr(model.options, "_replace"):
                    model.options = model.options._replace(
                        initial_prompt=combined_prompt,
                        beam_size=b_size,
                        best_of=b_size,
                        temperatures=[0.0] if speed_mode else [0.0, 0.2, 0.4],
                        condition_on_previous_text=True,
                        prompt_reset_on_temperature=0.5
                    )
                print(f"--- WORKER {worker.worker_id}: "
                      f"{'Speed-Mode' if speed_mode else 'Deep-Context'} "
                      f"(Beam: {b_size}) ---")
            except Exception as e:
                print(f"--- WORKER {worker.worker_id}: Optionen teilweise: {e} ---")

        if hasattr(model, "asr_options"):
            model.asr_options["initial_prompt"] = combined_prompt

        # 3. Audio vorbereiten
        validated_file_path = validate_multimedia_file(file_path)
        if not validated_file_path.lower().endswith(
            (".wav", ".mp3", ".flac", ".ogg", ".aac")
        ):
            validated_file_path = convert_to_wav(validated_file_path)

        audio = whisperx.load_audio(validated_file_path)

        # 4. TRANSKRIPTION
        t_batch_size = 16 if speed_mode else 1
        transcribe_options = {"batch_size": t_batch_size}

        if language != "Identify":
            transcribe_options['language'] = LANGUAGE_OPTIONS.get(language, "de")

        print(f"--- WORKER {worker.worker_id}: Starte "
              f"{'SPEED' if speed_mode else 'PRÄZISIONS'}-Transkription "
              f"für {os.path.basename(file_path)} ---")

        result = model.transcribe(audio, **transcribe_options)
        final_segments = result["segments"]
        detected_lang = result.get("language", "de")

        # 5. ALIGNMENT
        if not speed_mode:
            print(f"--- WORKER {worker.worker_id}: Transkription fertig "
                  f"({detected_lang}). Starte Alignment... ---")

            align_model_id = (
                "jonatasgrosman/wav2vec2-large-xlsr-53-german"
                if detected_lang == "de" else None
            )

            # Alignment-Modell auf demselben Device laden
            align_device = device
            # MPS-Fallback: Einige Alignment-Ops unterstützen MPS nicht voll
            # In dem Fall auf CPU ausweichen
            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=align_device,
                    model_name=align_model_id
                )
            except RuntimeError as e:
                if "MPS" in str(e) or "mps" in str(e):
                    print(f"--- WORKER {worker.worker_id}: MPS-Fallback "
                          f"für Alignment auf CPU ---")
                    align_device = "cpu"
                    align_model, metadata = whisperx.load_align_model(
                        language_code=detected_lang,
                        device=align_device,
                        model_name=align_model_id
                    )
                else:
                    raise

            aligned_result = whisperx.align(
                final_segments,
                align_model,
                metadata,
                audio,
                align_device,
                return_char_alignments=False
            )
            final_segments = aligned_result["segments"]

            del align_model
            clear_memory(align_device)
        else:
            print(f"--- WORKER {worker.worker_id}: Speed-Mode aktiv "
                  f"- Alignment übersprungen ---")

        # 6. DATEN AUFBEREITEN (identisch zur Windows-Version)
        full_text = " ".join(
            [seg["text"] for seg in final_segments]
        ).strip()
        segments_json_string = json.dumps(final_segments)

        mode_str = "Speed-Mode" if speed_mode else "Präzisions-Modus"
        elapsed = time.time() - start_time
        return [
            full_text,
            segments_json_string,
            f"{mode_str} fertig in {elapsed:.2f}s (Worker {worker.worker_id})"
        ]

    except Exception as e:
        import gradio as gr
        print(f"SCHWERER FEHLER (Worker {worker.worker_id if worker else '?'}): "
              f"{str(e)}")
        if not speed_mode:
            clear_memory(device)
        raise gr.Error(f"Backend-Fehler: {str(e)}")
    finally:
        # Worker zurück in den Pool geben
        if worker is not None:
            model_pool.release(worker)
        cleanup(device, file_path)
        clear_memory(device)
