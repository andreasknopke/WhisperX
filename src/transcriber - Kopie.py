import whisperx
import torch
import os
import gc
import time
import json
from src.model_manager import model_manager
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
    gc.collect()
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def transcribe_audio(file_path, language, model_name, device, initial_prompt_user=""):
    try:
        start_time = time.time()
        combined_prompt = f"{FORMAT_PROMPT} {initial_prompt_user}".strip()
        
        # 1. Modell laden
        model = model_manager.load_model(model_name, device)
        
        # 2. PRÄZISIONS-KONFIGURATION (Initial Prompt & Kontext)
        # Fix für neuere faster-whisper Versionen (vermeidet _replace Fehler bei dict-Typen)
        if hasattr(model, "options"):
            try:
                if hasattr(model.options, "_replace"):
                    model.options = model.options._replace(
                        initial_prompt=combined_prompt,
                        beam_size=5,
                        best_of=5,
                        temperatures=[0.0, 0.2, 0.4],
                        condition_on_previous_text=True, # Behält Prompt/Kontext aktiv
                        prompt_reset_on_temperature=0.5
                    )
                print("--- SYSTEM: Deep-Context & Beam Search (5) aktiviert ---")
            except Exception as e:
                print(f"--- SYSTEM: Hinweis - Optionen konnten nur teilweise gesetzt werden: {e} ---")
        
        # Ebene 2: asr_options (für interne WhisperX Logik)
        if hasattr(model, "asr_options"):
            model.asr_options["initial_prompt"] = combined_prompt

        # 3. Audio vorbereiten
        validated_file_path = validate_multimedia_file(file_path)
        if not validated_file_path.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".aac")):
            validated_file_path = convert_to_wav(validated_file_path)
        
        audio = whisperx.load_audio(validated_file_path)

        # 4. TRANSKRIPTION
        # batch_size=1 ist kritisch für die korrekte Nutzung des initial_prompt bei WhisperX!
        transcribe_options = {"batch_size": 1} 
        
        if language != "Identify":
            transcribe_options['language'] = LANGUAGE_OPTIONS.get(language, "de")
        
        print(f"--- SYSTEM: Starte Präzisions-Transkription für {os.path.basename(file_path)} ---")
        print(f"--- PROMPT: {initial_prompt_user[:100]}... ---")
        
        result = model.transcribe(audio, **transcribe_options)
        
        detected_lang = result.get("language", "de")
        
        # 5. ALIGNMENT (Zeitstempel für das Schreibprogramm)
        print(f"--- SYSTEM: Transkription fertig ({detected_lang}). Starte Alignment... ---")
        
        # Spezifisches deutsches Alignment-Modell für V100/WhisperX optimiert
        align_model_id = None
        if detected_lang == "de":
            align_model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
            print(f"--- SYSTEM: Nutze deutsches Alignment-Modell: {align_model_id} ---")

        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, 
            device=device,
            model_name=align_model_id
        )

        aligned_result = whisperx.align(
            result["segments"], 
            align_model, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        del align_model
        clear_memory(device)

        # 6. DATEN AUFBEREITEN
        full_text = " ".join([seg["text"] for seg in aligned_result["segments"]]).strip()
        segments_json_string = json.dumps(aligned_result["segments"])

        return [full_text, segments_json_string, f"Präzisions-Modus fertig in {time.time()-start_time:.2f}s"]

    except Exception as e:
        import gradio as gr
        print(f"SCHWERER FEHLER: {str(e)}")
        raise gr.Error(f"Backend-Fehler: {str(e)}")
    finally:
        cleanup(device, file_path)
        clear_memory(device)