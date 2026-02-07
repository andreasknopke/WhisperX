# **WhisperX Backend mit deutscher Benutzeroberfläche**

![Interface](docs/interface.png)

## **Beschreibung**

Dieses Projekt ist eine erweiterte WhisperX-Installation mit deutscher Gradio-Benutzeroberfläche, optimiert für lokale Audio- und Videotranskription. Es basiert auf [WhisperX](https://github.com/m-bain/whisperX) und [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) und bietet erweiterte Funktionen für Systemverwaltung, Authentifizierung und Leistungsoptimierung.

### **Hauptmerkmale**

- 🇩🇪 **Deutsche Benutzeroberfläche**: Vollständig lokalisierte Gradio-Oberfläche
- 🔐 **Authentifizierung**: Optionale Anmeldung via Benutzername/Passwort
- ⚡ **Speed Mode & Präzisions-Modus**: Wählbare Transkriptionsmodi
- 🛠️ **Systemverwaltung**: VRAM-Bereinigung, Prozessverwaltung, Auto-Restart
- 🎯 **Multi-Modell-Support**: Unterstützung für large-v3, large-v2, und spezialisierte deutsche Modelle
- 💻 **GPU & CPU Support**: Läuft mit CUDA oder auf CPU

---

## **Systemanforderungen**

- Python 3.10
- CUDA 12.1+ (nur für NVIDIA GPU)
- Windows (primär getestet) oder Linux

---

## **Dateiübersicht**

### **Hauptdateien**
- **`app.py`**: Hauptanwendung mit Gradio-UI (deutsche Sprache)
- **`src/transcriber.py`**: Transkriptions-Engine mit Speed/Präzisions-Modi
- **`src/model_manager.py`**: Modellverwaltung und -caching
- **`src/utils.py`**: Hilfsfunktionen für Dateivalidierung und Konvertierung

### **Setup-Dateien**
- **`setup_environment_cuda.bat`**: Installationsskript für Windows mit NVIDIA GPU (CUDA 12.1+)
- **`setup_environment_cpu.bat`**: Installationsskript für Windows ohne NVIDIA GPU
- **`run_script.bat`**: Startet die Anwendung nach der Installation
- **`start_whisperx.ps1`**: PowerShell Auto-Restart-Script mit Health-Check

### **Konfigurationsdateien**
- **`environment-cuda.yml`**: Conda-Umgebung für GPU-Systeme
- **`environment-cpu.yml`**: Conda-Umgebung für CPU-Systeme
- **`requirements/requirements_cuda.txt`**: Python-Dependencies für CUDA
- **`requirements/requirements_cpu.txt`**: Python-Dependencies für CPU
- **`.env`**: Umgebungsvariablen für Authentifizierung (nicht im Repository)

---

## **Installation**

### **Methode 1: Windows Batch-Skripte (Empfohlen)**

#### **Mit NVIDIA GPU (CUDA 12.1+)**

1. Repository herunterladen/klonen
2. `setup_environment_cuda.bat` ausführen
3. Das Skript erstellt automatisch:
   - Eine Python Virtual Environment (`venv`)
   - Installiert alle Dependencies aus `requirements/requirements_cuda.txt`
   - Installiert PyTorch mit CUDA 12.1 Support
4. Nach Abschluss: `run_script.bat` starten

#### **Ohne NVIDIA GPU (CPU-Modus)**

1. Repository herunterladen/klonen
2. `setup_environment_cpu.bat` ausführen
3. Das Skript erstellt:
   - Eine Python Virtual Environment (`venv`)
   - Installiert alle Dependencies aus `requirements/requirements_cpu.txt`
   - Installiert PyTorch ohne CUDA
4. Nach Abschluss: `run_script.bat` starten

---

### **Methode 2: Manuelle Installation (Alle Plattformen)**

```bash
# 1. Python Virtual Environment erstellen
python -m venv venv

# 2. Environment aktivieren
# Windows:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 3. Dependencies installieren
# Für GPU (CUDA 12.1+):
pip install -r requirements/requirements_cuda.txt
pip install torch==2.2.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Für CPU:
pip install -r requirements/requirements_cpu.txt
pip install torch==2.2.0 torchaudio==2.2.0
```

---

### **Methode 3: Conda Environment (Optional)**

```bash
# Für GPU:
conda env create -f environment-cuda.yml
conda activate whisperx-web-ui

# Für CPU:
conda env create -f environment-cpu.yml
conda activate whisperx-web-ui
```

---

## **Verwendung**

### **Starten der Anwendung**

**Windows (empfohlen):**
```bash
# Einfach per Batch-Skript:
run_script.bat
```

**Alle Plattformen (manuell):**
```bash
# Virtual Environment aktivieren:
venv\Scripts\activate.bat  # Windows
source venv/bin/activate   # Linux/Mac

# Anwendung starten:
python app.py
```

### **Zugriff auf die Benutzeroberfläche**

- Die Anwendung läuft auf: **http://localhost:7860**
- Bei aktivierter Authentifizierung werden Sie nach Benutzername/Passwort gefragt

### **Authentifizierung konfigurieren**

Erstellen Sie eine `.env`-Datei im Projektverzeichnis:

```env
WHISPER_AUTH_USERNAME=admin
WHISPER_AUTH_PASSWORD=IhrSicheresPasswort
```

Falls keine `.env`-Datei vorhanden ist, werden Standard-Credentials verwendet (admin/password123).

---

## **Funktionen**

### **Tab 1: Transkription**

- **Audio/Video hochladen**: Unterstützt WAV, MP3, FLAC, OGG, AAC und Videodateien
- **Sprachauswahl**: Auto-Erkennung oder manuell (Deutsch, Englisch, Französisch, Spanisch, Italienisch)
- **Modellauswahl**: 
  - `large-v3` (empfohlen für beste Qualität)
  - `large-v2`
  - `cstr/whisper-large-v3-turbo-german-int8_float32` (optimiert für Deutsch)
- **Gerät**: CUDA (GPU) oder CPU
- **Initial Prompt**: Optionale Anweisungen für bessere Kontexterkennung
- **Speed Mode**: Schnellere Transkription ohne Alignment (Trade-off: geringere Präzision)

#### **Modi im Detail:**

**Präzisions-Modus (Standard)**:
- Beam Size: 5
- Multi-Temperature Sampling: [0.0, 0.2, 0.4]
- Word-Level Alignment aktiv
- Beste Qualität für wichtige Transkriptionen

**Speed Mode**:
- Beam Size: 1
- Single Temperature: 0.0
- Alignment übersprungen
- Batch Size: 16 statt 1
- Bis zu 3x schneller, geringfügig weniger präzise

### **Tab 2: System Admin**

Erweiterte Systemverwaltung für Server-Deployments:

- **🧹 VRAM leeren**: Erzwingt GPU-Speicherbereinigung
- **💀 Zombies killen**: Beendet hängende Python-Prozesse
- **🔄 Server Neustart**: Startet die Anwendung neu (Windows)

Diese Funktionen sind auch via API verfügbar für externe Automation.

---

## **PowerShell Auto-Restart Script**

Das `start_whisperx.ps1`-Script bietet erweiterte Produktivfunktionen:

### **Features:**
- Automatischer Start und Überwachung
- Health-Check via HTTP auf Port 7860
- Auto-Restart bei Abstürzen oder Hängern
- Minimiertes Fenster für Server-Betrieb

### **Verwendung:**
```powershell
# In PowerShell ausführen:
.\start_whisperx.ps1
```

**Hinweis:** Script-Pfade müssen angepasst werden:
- `$CONDA_PATH`: Pfad zu Ihrer Conda-Installation
- `$WORKING_DIR`: Pfad zu Ihrem WhisperX-Verzeichnis
- `$ENV_NAME`: Name der Conda/venv Environment

---

## **API-Nutzung**

Die Gradio-Anwendung stellt mehrere API-Endpunkte bereit:

### **Transkription:**
```python
POST /api/start_process
{
  "data": [
    file_path,      # Datei-Upload
    "German",       # Sprache
    "large-v3",     # Modell
    "cuda",         # Gerät
    "Initial prompt text",
    false           # Speed Mode
  ]
}
```

### **System-Administration:**
```python
POST /api/system_cleanup     # VRAM leeren
POST /api/system_kill_zombies # Prozesse beenden
POST /api/system_reboot      # Neustart
```

---

## **Technische Details**

### **Optimierungen**

- **TF32 aktiviert**: Schnellere GPU-Berechnungen auf modernen NVIDIA GPUs
- **Dynamische VRAM-Verwaltung**: Speicher wird nach jeder Transkription freigegeben
- **Modell-Caching**: Wiederverwendung geladener Modelle für bessere Performance
- **Lazy CUDA Loading**: Reduzierter Speicherverbrauch beim Start

### **Unterstützte Modelle**

1. **large-v3**: Neuestes Whisper-Modell (beste Qualität)
2. **large-v2**: Vorgängerversion (stabil, bewährt)
3. **guillaumekln/faster-whisper-large-v2**: Optimierte faster-whisper-Variante
4. **cstr/whisper-large-v3-turbo-german-int8_float32**: Spezialisiert für Deutsch (schneller, int8 quantisiert)

### **Compute Types**

- **CUDA**: `float16` (Standard für beste GPU-Performance)
- **CPU**: `float32` (Standard für CPU-Berechnungen)

### **Alignment-Modelle**

Für Deutsch wird automatisch verwendet:
- `jonatasgrosman/wav2vec2-large-xlsr-53-german`

Für andere Sprachen werden die Standard-WhisperX-Alignment-Modelle genutzt.

---

## **Unterschiede zum Original WhisperX**

Dieses Repository enthält folgende Anpassungen:

### **Neue Features:**
1. ✅ Deutsche Benutzeroberfläche (Gradio)
2. ✅ Authentifizierungssystem mit .env-Konfiguration
3. ✅ Speed Mode vs. Präzisions-Modus
4. ✅ System-Admin-Tab für Server-Management
5. ✅ PowerShell Auto-Restart-Script
6. ✅ VRAM/Cache-Bereinigungsfunktionen
7. ✅ Zombie-Prozess-Killer
8. ✅ API-Endpunkte für externe Steuerung

### **Technische Änderungen:**
- Virtual Environment (venv) statt ausschließlich Conda
- Modell-Manager für besseres Caching
- Deutsche Standard-Prompts für bessere Interpunktion
- Batch-Skripte für Windows-Installation
- Requirements in separaten Dateien (CUDA/CPU)

### **Konfigurierte Optimierungen:**
- TF32 für NVIDIA Ampere+ GPUs
- Dynamisches Memory Management
- Konfigurierbare Batch-Sizes
- Multi-Temperature-Sampling (Präzisions-Modus)

---

## **Changelog**

### [Aktuelle Version]

- **Deutsche Lokalisierung**: Komplette UI auf Deutsch
- **Authentifizierung**: Login-Schutz via Umgebungsvariablen
- **System-Admin-Features**: VRAM-Bereinigung, Prozess-Management, Restart
- **Speed Mode**: Neue Option für schnellere Transkription
- **PowerShell-Script**: Auto-Restart mit Health-Check
- **Modell-Manager**: Effizientes Modell-Caching
- **Virtual Environment Support**: Installation ohne Conda möglich

### [1.1.0] - 2024-08-06 (Aus Original-Repository)

- **Modified default model selection**:
  - For CUDA-enabled devices, changed default model from "Large-v2" to "Medium"
  - For CPU devices, kept default model as "Medium"

- **Improved compute type selection**:
  - For CPU devices, now uses "int8" compute type instead of "float32"
  - For CUDA devices, kept "float16" compute type

- **Performance**: These changes aim to balance performance and resource usage across different hardware configurations

---

## **Fehlerbehebung**

### **Problem: CUDA out of memory**
- Lösung: Verwenden Sie den "VRAM leeren"-Button im System-Admin-Tab
- Alternative: Kleineres Modell wählen (z.B. `large-v2` statt `large-v3`)
- Speed Mode aktivieren (reduziert VRAM-Nutzung)

### **Problem: Langsame Transkription**
- Lösung: Speed Mode aktivieren
- Alternative: Gerät auf "cuda" stellen (falls GPU vorhanden)
- Kleineres Modell wählen

### **Problem: "Could not activate virtual environment"**
- Lösung: `setup_environment_cuda.bat` oder `setup_environment_cpu.bat` erneut ausführen
- Sicherstellen, dass Python 3.10 installiert ist

### **Problem: Server reagiert nicht mehr**
- Lösung: PowerShell-Script `start_whisperx.ps1` verwenden (automatischer Restart)
- Manuell: Prozesse mit "Zombies killen"-Button beenden

---

## **Autoren**

- **Original WhisperX Kit**: [MISTER CONTENTS](https://mistercontenidos.com/) & [Ricardo Gonzalez](https://www.linkedin.com/in/pedrocuervomkt/)
- **Deutsche Anpassung & Erweiterungen**: Andreas Knopke

---

## **Lizenz & Credits**

Dieses Projekt basiert auf:
- [WhisperX](https://github.com/m-bain/whisperX) von Max Bain
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) von SYSTRAN
- [OpenAI Whisper](https://github.com/openai/whisper) von OpenAI

---

## **Sprachen / Languages**

- [Español](docs/README_ES.md)
- [Português](docs/README_PT.md)
- English (see above for German version)
