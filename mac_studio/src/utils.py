"""
Utility-Funktionen – macOS/Apple Silicon Edition
=================================================
Plattform-unabhängige Hilfsfunktionen für Dateiverarbeitung.
"""

import os
import subprocess
import mimetypes
import json
import tempfile
import threading

# Constants
SUPPORTED_EXTENSIONS = (
    '.mp3', '.mp4', '.wav', '.avi', '.mov',
    '.flv', '.mkv', '.webm', '.aac', '.flac', '.ogg', '.m4a'
)
TEMP_DIR = os.path.join(os.getcwd(), 'Temp')

# Thread-lokaler Speicher für temp-Dateien (vermeidet Konflikte bei Parallelverarbeitung)
_thread_local = threading.local()


def _create_temp_directory():
    """Erstellt das Temp-Verzeichnis falls nötig."""
    os.makedirs(TEMP_DIR, exist_ok=True)


def _get_thread_temp_file() -> str:
    """Gibt einen thread-spezifischen Temp-Dateinamen zurück."""
    _create_temp_directory()
    thread_id = threading.get_ident()
    return os.path.join(TEMP_DIR, f"temp_audio_{thread_id}.wav")


def is_valid_multimedia_file(file_path):
    """Prüft ob die Datei ein unterstütztes Multimedia-Format ist."""
    normalized_path = os.path.normpath(file_path)
    mime_type, _ = mimetypes.guess_type(normalized_path)
    is_supported_mime = mime_type and (
        mime_type.startswith('audio') or mime_type.startswith('video')
    )
    return is_supported_mime or normalized_path.lower().endswith(SUPPORTED_EXTENSIONS)


def validate_multimedia_file(file_path):
    """Validiert ob die Datei ein unterstütztes Multimedia-Format ist."""
    if not is_valid_multimedia_file(file_path):
        raise ValueError(
            "The uploaded file is not a valid multimedia file. "
            "Please upload a compatible audio or video file."
        )
    return file_path


def convert_to_wav(input_file):
    """
    Konvertiert die Eingabedatei in WAV-Format.
    Thread-sicher: Jeder Thread bekommt seine eigene Temp-Datei.
    """
    temp_audio_file = _get_thread_temp_file()
    command = [
        "ffmpeg", "-i", input_file,
        "-vn", "-ac", "1", "-ar", "16000",
        "-y", temp_audio_file
    ]
    try:
        subprocess.run(
            command, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        error_message = f"Error during conversion: {e.stderr.decode()}"
        print(error_message)
        raise ValueError(
            f"Could not convert the file to WAV format. "
            f"Ensure ffmpeg is installed (brew install ffmpeg). "
            f"Details: {error_message}"
        )
    return temp_audio_file


def format_time(time_in_seconds, format_type="vtt"):
    """Formatiert Sekunden in ein lesbares Zeitformat."""
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    milliseconds = int((time_in_seconds - int(time_in_seconds)) * 1000)

    if format_type == "srt":
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    else:  # vtt
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def save_transcription(segments, file_format):
    """Speichert die Transkription im angegebenen Format."""
    _create_temp_directory()
    file_path = os.path.join(TEMP_DIR, f'transcription_output.{file_format}')
    with open(file_path, 'w', encoding='utf-8') as file:
        if file_format == "txt":
            file.writelines(
                f"{segment['text'].strip()}\n" for segment in segments
            )
        elif file_format == "vtt":
            file.write("WEBVTT\n\n")
            file.writelines(
                f"{i+1}\n"
                f"{format_time(segment['start'], 'vtt')} --> "
                f"{format_time(segment['end'], 'vtt')}\n"
                f"{segment['text'].strip()}\n\n"
                for i, segment in enumerate(segments)
            )
        elif file_format == "srt":
            file.writelines(
                f"{i+1}\n"
                f"{format_time(segment['start'], 'srt')} --> "
                f"{format_time(segment['end'], 'srt')}\n"
                f"{segment['text'].strip()}\n\n"
                for i, segment in enumerate(segments)
            )
        elif file_format == "json":
            json.dump(
                {"segments": segments}, file,
                ensure_ascii=False, indent=4
            )
    return file_path


def cleanup(device, input_file_path):
    """Bereinigt temporäre Dateien und Cache."""
    # Thread-spezifische Temp-Datei löschen
    temp_audio_file = _get_thread_temp_file()
    if os.path.exists(temp_audio_file) and input_file_path != temp_audio_file:
        try:
            os.remove(temp_audio_file)
        except OSError as e:
            print(f"Error deleting temporary file: {e}")

    # Speicher freigeben
    if device == "mps":
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    elif device == "cuda":
        import torch
        torch.cuda.empty_cache()

    import gc
    gc.collect()
