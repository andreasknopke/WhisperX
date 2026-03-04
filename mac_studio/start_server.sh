#!/bin/bash
# =============================================================
# WhisperX Mac Studio – Start-Skript
# =============================================================
# Startet Ollama + WhisperX Backend
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  🎙️ WhisperX Mac Studio – Server Start"
echo "=============================================="

# .env laden
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Virtual Environment aktivieren
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Python venv aktiviert"
else
    echo "❌ .venv nicht gefunden. Bitte erst ./setup_mac.sh ausführen."
    exit 1
fi

# Ollama starten (falls nicht schon läuft)
if ! pgrep -x "ollama" > /dev/null; then
    echo "🤖 Starte Ollama Server..."
    ollama serve &
    sleep 3
    echo "✅ Ollama gestartet"
else
    echo "✅ Ollama läuft bereits"
fi

# Prüfe ob Mistral-Modell vorhanden
if ollama list 2>/dev/null | grep -q "mistral-small"; then
    echo "✅ Mistral Small Modell verfügbar"
else
    echo "⚠️  Mistral Small nicht gefunden. Lade herunter..."
    ollama pull mistral-small
fi

# System-Info
echo ""
echo "--- System Info ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS verfügbar: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
echo ""

# Pool-Größe
POOL_SIZE=${WHISPERX_POOL_SIZE:-6}
echo "--- Konfiguration ---"
echo "Worker Pool:  $POOL_SIZE Instanzen"
echo "Hinweis:      Pool-Größe ändern via WHISPERX_POOL_SIZE in .env"
echo "              256GB/60C: 6-8 | 512GB/80C: 12-20"
echo "LLM Backend:  Ollama (${OLLAMA_MODEL:-mistral-small})"
echo "Server Port:  7860"
echo "Device-Remap: cuda → mps (automatisch)"
echo ""

# WhisperX Backend starten
echo "🚀 Starte WhisperX Backend..."
echo "   Web-UI:  http://localhost:7860"
echo "   API:     http://localhost:7860/api/"
echo ""
python app.py
