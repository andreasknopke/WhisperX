# 🎙️ WhisperX Backend – Mac Studio M3 Ultra Edition

> Optimiert für Apple Silicon M3 Ultra (28C CPU / 60C GPU / 256 GB Unified Memory)  
> 30 gleichzeitige Verbindungen + Mistral Small LLM-Nachkontrolle

---

## 📋 Inhaltsverzeichnis

- [Architektur-Überblick](#architektur-überblick)
- [Hardware-Anforderungen](#hardware-anforderungen)
- [Installation](#installation)
- [Konfiguration](#konfiguration)
- [API-Referenz](#api-referenz)
- [Performance-Erwartungen](#performance-erwartungen)
- [Tuning-Guide](#tuning-guide)
- [Troubleshooting](#troubleshooting)

---

## 🏗️ Architektur-Überblick

```
┌──────────────────────────────────────────────────────────┐
│                   Mac Studio M3 Ultra                    │
│                   256 GB Unified Memory                  │
│                                                          │
│  ┌──────────────────┐       ┌──────────────────────────┐ │
│  │  Gradio/FastAPI   │       │     Ollama Server        │ │
│  │  (Port 7860)      │ ───── │     (Port 11434)         │ │
│  │                   │       │     Mistral Small 3      │ │
│  │  30 Connections   │       │     ~14-48 GB            │ │
│  └───────┬───────────┘       └──────────────────────────┘ │
│          │                                                │
│  ┌───────▼──────────────────────────────────────┐        │
│  │         WhisperX Worker Pool (4-6)            │        │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │        │
│  │  │Worker 0│ │Worker 1│ │Worker 2│ │Worker 3│ │        │
│  │  │large-v3│ │large-v3│ │large-v3│ │large-v3│ │        │
│  │  │ ~6 GB  │ │ ~6 GB  │ │ ~6 GB  │ │ ~6 GB  │ │        │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ │        │
│  └──────────────────────────────────────────────┘        │
│                                                          │
│  Memory: 48 GB LLM + 6×6 GB Whisper = ~84 GB            │
│  Verbleibend: ~172 GB für Audio, Alignment, OS           │
└──────────────────────────────────────────────────────────┘
```

### Schlüssel-Unterschiede zur Windows/CUDA-Version

| Aspekt | Windows/V100 | Mac Studio M3 Ultra |
|--------|-------------|---------------------|
| GPU Backend | CUDA | **MPS** (Metal Performance Shaders) |
| Memory | 64 GB RAM + 32 GB VRAM (getrennt) | **256 GB Unified** (geteilt) |
| Modell-Instanzen | 1 (VRAM-Limit) | **4-6 parallel** |
| Concurrency | Single-User | **30 Verbindungen** (Worker-Pool) |
| LLM | Extern (LM-Studio) | **Integriert** (Ollama) |
| Temp-Dateien | Shared | **Thread-isoliert** |
| System-Commands | `taskkill` | `pgrep/kill` |

---

## 💻 Hardware-Anforderungen

### Minimum
- Apple Silicon Mac (M1 Pro oder höher)
- 64 GB Unified Memory
- macOS 14.0+ (Sonoma)

### Empfohlen (für 30 User + LLM)
- **Mac Studio M3 Ultra**
- **256 GB oder 512 GB Unified Memory**
- 60C oder 80C GPU
- macOS 15.0+ (Sequoia)
- 4 TB SSD (für Modelle + Audio-Buffer)

### Memory-Budget

| Konfiguration | 256 GB / 60C | 512 GB / 80C |
|--------------|-------------|-------------|
| macOS + System | ~8 GB | ~8 GB |
| Ollama Mistral Small Q4 | ~16 GB | ~16 GB |
| WhisperX Worker (à ~6 GB) | 6×6 = 36 GB | 20×6 = 120 GB |
| Alignment Models | ~12 GB | ~40 GB |
| Audio-Buffer (30 User) | ~10 GB | ~10 GB |
| **Gesamt** | **~82 GB** | **~194 GB** |
| **Frei** | **~174 GB** | **~318 GB** |
| **Sinnvolle Worker** | **6-8** | **12-20** |

---

## 🚀 Installation

### Schnell-Setup (empfohlen)

```bash
# Repository klonen / Dateien auf Mac kopieren
cd /path/to/mac_studio

# Setup-Skript ausführen
chmod +x setup_mac.sh start_server.sh monitor.sh
./setup_mac.sh
```

### Manuelles Setup

```bash
# 1. Homebrew + System-Dependencies
brew install ffmpeg python@3.12

# 2. Python Environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. PyTorch mit MPS-Support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Dependencies
pip install -r requirements/requirements_apple.txt

# 5. Environment
cp .env.example .env
# → .env anpassen

# 6. Ollama + Mistral Small
brew install ollama
ollama serve &
ollama pull mistral-small
```

### MPS verifizieren

```python
import torch
print(f"MPS verfügbar: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
# Beide sollten True sein
```

---

## ⚙️ Konfiguration

### .env Datei

```env
# Authentifizierung
WHISPER_AUTH_USERNAME=admin
WHISPER_AUTH_PASSWORD=sicheres_passwort

# Worker Pool (4-6 für M3 Ultra 256 GB)
WHISPERX_POOL_SIZE=6

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral-small
LLM_TIMEOUT=120

# MPS Tuning
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Pool-Größe Empfehlung

| Unified Memory | GPU Cores | Pool Size | Max parallele User |
|---------------|-----------|-----------|--------------------|
| 64 GB | beliebig | 2 | ~10 |
| 128 GB | beliebig | 4 | ~20 |
| 192 GB | 60C | 5 | ~25 |
| **256 GB** | **60C** | **6-8** | **~30-40** |
| 256 GB | 80C | 8-10 | ~40-50 |
| **512 GB** | **60C** | **8-15** | **~40-75** |
| **512 GB** | **80C** | **12-20** | **~60-100** |

> **GPU-Cores als Limit:** Mehr Worker als GPU-Cores/3 bringt diminishing returns,
> da sich die Worker die Metal-Compute-Units teilen.

---

## 📡 API-Referenz

Die API ist **identisch zur Windows-Version**. Alle bestehenden Clients funktionieren ohne Änderung.

### Transkription

```
POST /api/start_process
```

Parameter (identisch):
- `file_input`: Audio/Video-Datei
- `language_dropdown`: "Identify", "English", "German", "French", "Spanish", "Italian"
- `model_dropdown`: "large-v3" (default)
- `device_dropdown`: "mps" (statt "cuda")
- `initial_prompt_input`: Optional, Kontext-Prompt
- `speed_mode_input`: true/false
- `llm_review_input`: true/false (**NEU**: LLM-Nachkontrolle)

Response (identisch):
```json
{
  "data": [
    "Transkribierter Text...",
    "[{\"start\": 0.0, \"end\": 2.5, \"text\": \"...\"}]",
    "Präzisions-Modus fertig in 12.34s (Worker 2)"
  ]
}
```

### System-Endpoints

| Endpoint | Funktion |
|----------|----------|
| `POST /api/system_cleanup` | MPS Cache + RAM bereinigen |
| `POST /api/system_kill_zombies` | Hängende Python-Prozesse beenden |
| `POST /api/system_reboot` | Server neu starten |
| `POST /api/system_pool_status` | Worker-Pool Status abfragen |

---

## 📊 Performance-Erwartungen

### Vergleich: V100 vs. M3 Ultra

#### Einzelne Transkription (10 Min Audio, large-v3)

| Metrik | V100 32 GB | M3 Ultra 256 GB |
|--------|-----------|-----------------|
| Transkription | ~40s | ~50-55s |
| + Alignment | ~15s | ~18-20s |
| **Gesamt** | **~55s** | **~70-75s** |
| Relative Speed | 1.0× | **~0.75×** |

#### Durchsatz bei 30 gleichzeitigen Usern

| Metrik | V100 32 GB | M3 Ultra 256 GB |
|--------|-----------|-----------------|
| Parallele Jobs | 1 | **4-6** |
| Queue-Wartezeit (30 User) | ~25 Min | **~5-8 Min** |
| Durchsatz/Stunde | ~65 Jobs | **~200-300 Jobs** |
| **Relativer Throughput** | **1.0×** | **~3-5×** |

> **Fazit**: Einzelne Jobs ~25% langsamer, aber Gesamtdurchsatz 3-5× höher durch Parallelisierung.

#### LLM-Nachkontrolle (Mistral Small, 1000 Wörter)

| Quantisierung | Token/s | Latenz (1000 Wörter) |
|--------------|---------|---------------------|
| Q4_K_M | 35-55 | ~3-5s |
| Q8_0 | 25-40 | ~5-8s |
| FP16 | 15-25 | ~8-15s |

### Realistische Szenarien

**Szenario A: Live-Diktat (ge-chunktes Audio, 2-5s Chunks via Schreibdienst)**
- Chunk-Verarbeitung: ~0.5-1.5s pro Chunk (Speed-Mode, kein Alignment)
- 6 Worker (256GB): 6 Chunks gleichzeitig → bis zu ~30 User live
- 20 Worker (512GB/80C): 20 Chunks gleichzeitig → bis zu ~100 User live
- **Entscheidend:** Speed-Mode (`skip_alignment=true`) für minimale Latenz

**Szenario B: Offline-Diktate (Schreibdienst Worker, 2-5 Min Audio)**
- Durchschnittliche Bearbeitungszeit: ~30-40s pro Diktat
- 6 Worker: Wartezeit max ~2-3 Min bei 30 gleichzeitigen
- Mit LLM-Review: +3-5s pro Diktat

**Szenario C: Konferenzen (30-60 Min Audio)**
- Durchschnittliche Bearbeitungszeit: ~3-5 Min pro Konferenz
- Empfehlung: Weniger Worker, dafür mehr RAM pro Worker

---

## 🔧 Tuning-Guide

### MPS-Optimierung

```env
# Mehr GPU-Speicher erlauben (Default: 0.7)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85

# MPS-Fallback für unsupported Ops
PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Worker-Pool Tuning

```python
# Weniger Worker = mehr RAM pro Worker = schnellere Einzeljobs
WHISPERX_POOL_SIZE=4  # Für große Audiodateien

# Mehr Worker = mehr Durchsatz bei kurzen Dateien
WHISPERX_POOL_SIZE=8  # Nur wenn genug RAM (256 GB)
```

### Ollama Tuning

```bash
# Mehr Context-Window für lange Transkriptionen
OLLAMA_NUM_CTX=8192

# GPU Layers (alle auf Metal)
OLLAMA_NUM_GPU=999

# Parallel Requests erlauben
OLLAMA_NUM_PARALLEL=4
```

### Monitoring

```bash
# Echtzeit-Monitor starten
./monitor.sh

# Oder manuell:
# Memory
vm_stat | head -10

# Python-Prozesse
ps aux | grep python

# Ollama
curl http://localhost:11434/api/tags
```

---

## 🐛 Troubleshooting

### "MPS backend out of memory"

```bash
# 1. Pool-Größe reduzieren
WHISPERX_POOL_SIZE=4

# 2. MPS Watermark erhöhen
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9

# 3. LLM Quantisierung auf Q4 wechseln
ollama pull mistral-small:q4_K_M
```

### "RuntimeError: MPS does not support ..."

Das Alignment-Modell nutzt einige Operationen, die MPS (noch) nicht unterstützt.
Der Transcriber fällt automatisch auf CPU zurück für Alignment-Operationen.
Dies ist normal und hat minimalen Performance-Impact.

### Ollama startet nicht

```bash
# Prüfen ob Port 11434 frei ist
lsof -i :11434

# Ollama manuell starten
ollama serve

# Modell prüfen
ollama list
ollama pull mistral-small
```

### Gradio "Too many requests"

```env
# In .env: Pool-Größe an Bedarf anpassen
WHISPERX_POOL_SIZE=6

# In app.py ist concurrency_limit=30 gesetzt
# Das erlaubt 30 gleichzeitige Verbindungen,
# die über den Worker-Pool abgearbeitet werden
```

### ctranslate2 Kompatibilität

Falls `ctranslate2` Probleme auf Apple Silicon macht:

```bash
# Option 1: Neueste Version
pip install ctranslate2 --upgrade

# Option 2: Aus Source bauen mit Metal-Support
pip install ctranslate2 --no-binary ctranslate2

# Option 3: Alternative Engine (mlx-whisper)
pip install mlx-whisper
# → Erfordert Anpassung in model_manager.py
```

---

## 📁 Dateistruktur

```
mac_studio/
├── app.py                          # Hauptanwendung (Gradio + API)
├── .env.example                    # Konfigurationstemplate
├── setup_mac.sh                    # Einrichtungsskript
├── start_server.sh                 # Start-Skript
├── monitor.sh                      # Performance-Monitor
├── requirements/
│   └── requirements_apple.txt      # Python Dependencies
├── src/
│   ├── __init__.py
│   ├── transcriber.py              # Transkription (MPS + Worker-Pool)
│   ├── model_manager.py            # Modell-Pool (Thread-sicher)
│   ├── utils.py                    # Hilfsfunktionen (Thread-sicher)
│   └── llm_client.py              # Ollama/LLM Integration
└── Temp/                           # Temporäre Dateien (auto-erstellt)
```

---

## 🔄 Migration von Windows / Schreibdienst-Kompatibilität

1. **API ist 100% identisch** – Der Schreibdienst braucht **keine Änderung**
2. **Device-Remapping:** Der Client sendet `"cuda"` → Backend mappt automatisch auf `"mps"`
3. Timestamps, JSON-Format, Segment-Struktur – alles identisch
4. **Live-Diktat** (ge-chunktes Audio) funktioniert wie bisher über Gradio API
5. **Offline-Diktate** (Worker-Queue) funktionieren wie bisher
6. Neuer optionaler Parameter `llm_review_input` (default: false)
7. Neuer Endpoint `system_pool_status` für Pool-Monitoring

### Einzige Änderung in Schreibdienst `.env.local`:

```env
# Vorher (Windows)
WHISPER_SERVICE_URL=http://windows-server:7860

# Nachher (Mac Studio)
WHISPER_SERVICE_URL=http://mac-studio:7860
# Sonst nichts ändern! Device-Remapping passiert automatisch.
```

---

## 📝 Warum NICHT vLLM?

**vLLM unterstützt Apple Silicon / MPS nicht.** Es ist ausschließlich für NVIDIA CUDA optimiert.

Alternativen für Apple Silicon:

| Framework | Performance | Ease of Use | Empfehlung |
|-----------|------------|-------------|------------|
| **Ollama** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **✅ Empfohlen** |
| MLX-LM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Für Power-User |
| llama.cpp | ⭐⭐⭐⭐ | ⭐⭐⭐ | Low-Level |
| vLLM | ❌ | – | Nicht kompatibel |

Ollama nutzt intern `llama.cpp` mit Metal-Beschleunigung und bietet eine einfache REST-API.
