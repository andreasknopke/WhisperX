#!/bin/bash
# =============================================================
# WhisperX Mac Studio – Einrichtungsskript
# =============================================================
# Nutzung: chmod +x setup_mac.sh && ./setup_mac.sh
# =============================================================

set -e

echo "=============================================="
echo "  WhisperX Mac Studio – Setup"
echo "  Apple Silicon (M3 Ultra) Edition"
echo "=============================================="
echo ""

# 1. Homebrew prüfen
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew nicht gefunden. Installiere..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 2. System-Dependencies
echo "📦 Installiere System-Dependencies..."
brew install ffmpeg python@3.12

# 3. Python Virtual Environment
echo "🐍 Erstelle Python-Umgebung..."
python3.12 -m venv .venv
source .venv/bin/activate

# 4. PyTorch mit MPS-Support
echo "🔥 Installiere PyTorch (MPS)..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. WhisperX Dependencies
echo "📚 Installiere WhisperX Dependencies..."
pip install -r requirements/requirements_apple.txt

# 6. .env erstellen
if [ ! -f .env ]; then
    echo "⚙️  Erstelle .env aus Template..."
    cp .env.example .env
    echo "   → Bitte .env anpassen!"
fi

# 7. Ollama installieren
echo ""
echo "=============================================="
echo "  Ollama Setup (für Mistral Small LLM)"
echo "=============================================="
if ! command -v ollama &> /dev/null; then
    echo "📥 Installiere Ollama..."
    brew install ollama
fi

echo "📥 Lade Mistral Small Modell..."
ollama pull mistral-small

echo ""
echo "=============================================="
echo "  ✅ Setup abgeschlossen!"
echo "=============================================="
echo ""
echo "  Starten mit:  ./start_server.sh"
echo ""
echo "  Oder manuell:"
echo "    1. source .venv/bin/activate"
echo "    2. ollama serve &"
echo "    3. python app.py"
echo ""
echo "  Web-UI:  http://localhost:7860"
echo "  Ollama:  http://localhost:11434"
echo ""
