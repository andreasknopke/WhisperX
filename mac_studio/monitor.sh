#!/bin/bash
# =============================================================
# Performance Monitor für Mac Studio
# =============================================================
# Zeigt Echtzeit-Statistiken für CPU, GPU, Memory, und Netzwerk
# Nutzung: ./monitor.sh
# =============================================================

echo "=============================================="
echo "  📊 Mac Studio Performance Monitor"
echo "=============================================="
echo "  Ctrl+C zum Beenden"
echo ""

while true; do
    clear
    echo "=============================================="
    echo "  📊 Mac Studio Monitor – $(date '+%H:%M:%S')"
    echo "=============================================="

    # Memory
    echo ""
    echo "--- Memory ---"
    vm_stat | awk '
        /Pages free/ { free=$3 }
        /Pages active/ { active=$3 }
        /Pages inactive/ { inactive=$3 }
        /Pages wired/ { wired=$3 }
        END {
            gsub(/\./, "", free); gsub(/\./, "", active);
            gsub(/\./, "", inactive); gsub(/\./, "", wired);
            total = (free + active + inactive + wired) * 16384 / 1073741824;
            used = (active + wired) * 16384 / 1073741824;
            printf "  Gesamt: %.1f GB | Genutzt: %.1f GB | Frei: %.1f GB\n",
                   total, used, total - used
        }'

    # CPU Last
    echo ""
    echo "--- CPU ---"
    top -l 1 -n 0 | grep "CPU usage" | awk '{print "  " $0}'

    # GPU (Metal)
    echo ""
    echo "--- GPU (Metal) ---"
    if command -v ioreg &> /dev/null; then
        echo "  Metal GPU Cores: 60 (M3 Ultra)"
        # Leider gibt macOS keine GPU-Auslastung über CLI
        echo "  (GPU-Auslastung: Activity Monitor → GPU History)"
    fi

    # Python/WhisperX Prozesse
    echo ""
    echo "--- WhisperX Prozesse ---"
    ps aux | grep -E "python.*app.py" | grep -v grep | \
        awk '{printf "  PID: %s | CPU: %s%% | MEM: %s%% | RSS: %.0f MB\n", $2, $3, $4, $6/1024}'

    # Ollama
    echo ""
    echo "--- Ollama ---"
    if pgrep -x "ollama" > /dev/null; then
        ps aux | grep "ollama" | grep -v grep | head -1 | \
            awk '{printf "  PID: %s | CPU: %s%% | MEM: %s%% | RSS: %.0f MB\n", $2, $3, $4, $6/1024}'
    else
        echo "  ❌ Nicht aktiv"
    fi

    # Netzwerk-Verbindungen auf Port 7860
    echo ""
    echo "--- Aktive Verbindungen (Port 7860) ---"
    CONN_COUNT=$(lsof -i :7860 2>/dev/null | grep ESTABLISHED | wc -l | tr -d ' ')
    echo "  Aktive Verbindungen: $CONN_COUNT"

    sleep 5
done
