# Konfiguration
$CONDA_PATH = "C:\Users\Andre.AUDIO-WS1\miniconda3\Scripts\conda.exe"
$ENV_NAME = "whisperx-web-ui"
$WORKING_DIR = "D:\GitHub\whisper\Kit-Whisperx"
$PORT = 7860
$CHECK_URL = "http://localhost:$PORT"
$WAIT_TIME = 15 # Etwas mehr Zeit lassen, da WhisperX/Gradio länger zum Laden braucht

Write-Host "--- WhisperX (Targeted Kill) Auto-Restart Script gestartet ---" -ForegroundColor Green

function Start-WhisperServer {
    Write-Host "$(Get-Date): Starte WhisperX Server (app.py)..." -ForegroundColor Cyan
    # Startet WhisperX minimiert
    Start-Process -FilePath $CONDA_PATH -ArgumentList "run", "-n", $ENV_NAME, "python", "app.py" -WorkingDirectory $WORKING_DIR -WindowStyle Minimized
}

# Erster Start
# Konfiguration
$CONDA_PATH = "C:\Users\Andre.AUDIO-WS1\miniconda3\Scripts\conda.exe"
$ENV_NAME = "whisperx-web-ui"
$WORKING_DIR = "D:\GitHub\whisper\Kit-Whisperx"
$PORT = 7860
$CHECK_URL = "http://localhost:$PORT"
$WAIT_TIME = 15

Write-Host "--- WhisperX Server & Monitor gestartet ---" -ForegroundColor Green
Write-Host "Logs werden unten angezeigt..." -ForegroundColor Gray
Write-Host "--------------------------------------------"

function Start-WhisperServer {
    Write-Host "$(Get-Date): Starte WhisperX Prozess..." -ForegroundColor Cyan
    
    # Wir nutzen hier den Aufruf über CMD /C, um die Logs direkt im aktuellen Stream zu behalten
    # Das '&' Zeichen in PowerShell startet den Befehl im selben Fenster
    & $CONDA_PATH run -n $ENV_NAME python "$WORKING_DIR\app.py"
}

while($true) {
    # Wir prüfen zuerst, ob der Port belegt ist
    $connection = Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue | Select-Object -First 1
    
    if ($null -eq $connection) {
        Write-Host "$(Get-Date): Server scheint nicht zu laufen. Initialisierung..." -ForegroundColor Yellow
        Start-WhisperServer
    }
    else {
        # Optionaler HTTP Check ob die UI reagiert
        try {
            $res = Invoke-WebRequest -Uri $CHECK_URL -Method Head -UseBasicParsing -TimeoutSec 5
            if ($res.StatusCode -ne 200) { throw "Not 200" }
        }
        catch {
            Write-Host "$(Get-Date): Port offen, aber UI reagiert nicht. Neustart PID $($connection.OwningProcess)..." -ForegroundColor Red
            Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
            Start-WhisperServer
        }
    }

    Start-Sleep -Seconds $WAIT_TIME
}