import whisperx
import torch
import os
import gc
import time
import json
from src.model_manager import model_manager
from src.utils import validate_multimedia_file, convert_to_wav, save_transcription, cleanup

# TF32 aktivieren für schnellere GPU-Berechnungen
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    gc.collect()
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def transcribe_audio(file_path, language, model_name, device, initial_prompt_user="", speed_mode=False):
    try:
        start_time = time.time()
        combined_prompt = f"{FORMAT_PROMPT} {initial_prompt_user}".strip()
        
        # 1. Modell laden
        model = model_manager.load_model(model_name, device)
        
        # 2. KONFIGURATION - b_size VOR dem try-Block definieren!
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
                print(f"--- SYSTEM: {'Speed-Mode' if speed_mode else 'Deep-Context'} aktiviert (Beam: {b_size}) ---")
            except Exception as e:
                print(f"--- SYSTEM: Hinweis - Optionen konnten nur teilweise gesetzt werden: {e} ---")
        
        if hasattr(model, "asr_options"):
            model.asr_options["initial_prompt"] = combined_prompt

        # 3. Audio vorbereiten
        validated_file_path = validate_multimedia_file(file_path)
        if not validated_file_path.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".aac")):
            validated_file_path = convert_to_wav(validated_file_path)
        
        audio = whisperx.load_audio(validated_file_path)

        # 4. TRANSKRIPTION
        t_batch_size = 16 if speed_mode else 1
        transcribe_options = {"batch_size": t_batch_size} 
        
        # Sprache setzen (vermeidet Language Detection)
        if language != "Identify":
            transcribe_options['language'] = LANGUAGE_OPTIONS.get(language, "de")
        
        print(f"--- SYSTEM: Starte {'SPEED' if speed_mode else 'PRÄZISIONS'}-Transkription für {os.path.basename(file_path)} ---")
        
        result = model.transcribe(audio, **transcribe_options)
        final_segments = result["segments"]
        detected_lang = result.get("language", "de")
        
        # 5. ALIGNMENT (Wird im Speed-Mode übersprungen)
        if not speed_mode:
            print(f"--- SYSTEM: Transkription fertig ({detected_lang}). Starte Alignment... ---")
            
            align_model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-german" if detected_lang == "de" else None

            align_model, metadata = whisperx.load_align_model(
                language_code=detected_lang, 
                device=device,
                model_name=align_model_id
            )

            aligned_result = whisperx.align(
                final_segments, 
                align_model, 
                metadata, 
                audio, 
                device, 
                return_char_alignments=False
            )
            final_segments = aligned_result["segments"]
            
            del align_model
            clear_memory(device)
        else:
            print("--- SYSTEM: Speed-Mode aktiv - Alignment übersprungen ---")

        # 6. DATEN AUFBEREITEN
        full_text = " ".join([seg["text"] for seg in final_segments]).strip()
        segments_json_string = json.dumps(final_segments)

        mode_str = "Speed-Mode" if speed_mode else "Präzisions-Modus"
        return [full_text, segments_json_string, f"{mode_str} fertig in {time.time()-start_time:.2f}s"]

    except Exception as e:
        import gradio as gr
        print(f"SCHWERER FEHLER: {str(e)}")
        if not speed_mode:
            clear_memory(device)
        raise gr.Error(f"Backend-Fehler: {str(e)}")
    finally:
        cleanup(device, file_path)
        clear_memory(device)