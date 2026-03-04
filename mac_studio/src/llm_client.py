"""
LLM Client – Ollama Integration für Transkriptions-Nachkontrolle
=================================================================
Verbindet sich mit einem lokalen Ollama-Server (Mistral Small)
zur Qualitätsprüfung und Korrektur von Transkriptionen.

Alternativ kann auch eine OpenAI-kompatible API genutzt werden
(z.B. MLX-LM-Server, vLLM, etc.).
"""

import os
import json
import time
import requests
from typing import Optional

# Konfiguration via Umgebungsvariablen
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral-small")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))  # Sekunden

# Alternative: OpenAI-kompatible API
OPENAI_BASE_URL = os.getenv("LLM_OPENAI_BASE_URL", "")  # z.B. http://localhost:8000/v1
OPENAI_API_KEY = os.getenv("LLM_OPENAI_API_KEY", "not-needed")
OPENAI_MODEL = os.getenv("LLM_OPENAI_MODEL", "mistral-small")


# System-Prompt für die Nachkontrolle
REVIEW_SYSTEM_PROMPT = """Du bist ein präziser Lektor für medizinische und allgemeine Transkriptionen.

Deine Aufgabe:
1. Korrigiere offensichtliche Transkriptionsfehler (falsch erkannte Wörter)
2. Korrigiere Grammatik und Interpunktion
3. Behalte den exakten Inhalt und Stil bei – KEINE Umformulierungen
4. Medizinische Fachbegriffe müssen korrekt geschrieben sein
5. Eigennamen und Abkürzungen beibehalten

WICHTIG: Gib NUR den korrigierten Text zurück, ohne Erklärungen oder Kommentare.
Wenn der Text bereits korrekt ist, gib ihn unverändert zurück."""

REVIEW_USER_TEMPLATE = """Bitte prüfe und korrigiere diese Transkription ({language}):

---
{text}
---

Korrigierter Text:"""


class LLMClient:
    """Client für LLM-basierte Transkriptions-Nachkontrolle."""

    def __init__(self):
        self._backend = self._detect_backend()
        if self._backend:
            print(f"--- LLM Client: Backend '{self._backend}' konfiguriert ---")
        else:
            print("--- LLM Client: Kein Backend verfügbar (optional) ---")

    def _detect_backend(self) -> Optional[str]:
        """Erkennt das verfügbare LLM-Backend."""
        # 1. Prüfe OpenAI-kompatible API
        if OPENAI_BASE_URL:
            try:
                resp = requests.get(
                    f"{OPENAI_BASE_URL}/models",
                    timeout=5,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
                )
                if resp.status_code == 200:
                    return "openai"
            except Exception:
                pass

        # 2. Prüfe Ollama
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if any(OLLAMA_MODEL in name for name in model_names):
                    return "ollama"
                else:
                    print(f"--- LLM Client: Ollama läuft, aber Modell "
                          f"'{OLLAMA_MODEL}' nicht gefunden. "
                          f"Verfügbar: {model_names} ---")
                    return "ollama"  # Wird beim Pull-Versuch laden
        except Exception:
            pass

        return None

    def is_available(self) -> bool:
        """Prüft ob ein LLM-Backend verfügbar ist."""
        return self._backend is not None

    def review_transcription(self, text: str, language: str = "German") -> str:
        """
        Sendet den Transkriptionstext zur Nachkontrolle an das LLM.

        Args:
            text: Der zu prüfende Transkriptionstext
            language: Sprache des Texts

        Returns:
            Der korrigierte Text
        """
        if not self._backend:
            return text

        prompt = REVIEW_USER_TEMPLATE.format(text=text, language=language)

        start = time.time()

        if self._backend == "ollama":
            result = self._query_ollama(prompt)
        elif self._backend == "openai":
            result = self._query_openai(prompt)
        else:
            return text

        elapsed = time.time() - start
        print(f"--- LLM Review: {len(text)} → {len(result)} Zeichen "
              f"in {elapsed:.1f}s ---")

        return result

    def _query_ollama(self, prompt: str) -> str:
        """Anfrage an Ollama-Server."""
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": REVIEW_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 4096,
                    }
                },
                timeout=LLM_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"--- LLM Ollama Fehler: {e} ---")
            raise

    def _query_openai(self, prompt: str) -> str:
        """Anfrage an OpenAI-kompatible API."""
        try:
            resp = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
                timeout=LLM_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"--- LLM OpenAI-compat Fehler: {e} ---")
            raise

    def pull_model(self) -> bool:
        """Zieht das konfigurierte Modell in Ollama (einmalig)."""
        if self._backend != "ollama":
            return False

        try:
            print(f"--- LLM: Lade Modell '{OLLAMA_MODEL}' in Ollama... ---")
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": OLLAMA_MODEL, "stream": False},
                timeout=600  # 10 Min für großes Modell
            )
            resp.raise_for_status()
            print(f"--- LLM: Modell '{OLLAMA_MODEL}' bereit ---")
            return True
        except Exception as e:
            print(f"--- LLM: Modell-Download fehlgeschlagen: {e} ---")
            return False
