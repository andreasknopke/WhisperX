"""
WhisperX Model Manager – Pool-basiert für Apple Silicon
=======================================================
Verwaltet einen Pool von WhisperX-Modell-Instanzen für
gleichzeitige Verarbeitung von bis zu 30 Verbindungen.
"""

import whisperx
import threading
import queue
import os
import time

# Verfügbare Modelle
MODELS = [
    "large-v3",
    "guillaumekln/faster-whisper-large-v2",
    "large-v2",
    "cstr/whisper-large-v3-turbo-german-int8_float32"
]

# Pool-Konfiguration via Umgebungsvariable
# Faustregel: (RAM - LLM - 20GB) / 6GB, begrenzt durch GPU-Cores
#   256GB / 60C GPU → 6-8 Worker
#   512GB / 80C GPU → 12-20 Worker
DEFAULT_POOL_SIZE = int(os.getenv("WHISPERX_POOL_SIZE", "6"))

if DEFAULT_POOL_SIZE < 1:
    DEFAULT_POOL_SIZE = 1
    print("--- POOL: WHISPERX_POOL_SIZE muss >= 1 sein, setze auf 1 ---")

print(f"--- POOL: Konfigurierte Größe: {DEFAULT_POOL_SIZE} Worker ---")


class ModelWorker:
    """Repräsentiert einen einzelnen Whisper-Worker mit geladenem Modell."""

    def __init__(self, worker_id: int, model_name: str, device: str):
        self.worker_id = worker_id
        self.model_name = model_name
        self.device = device
        self.model = None
        self.created_at = time.time()
        self.jobs_completed = 0
        self._load_model()

    def _load_model(self):
        """Lädt das WhisperX-Modell auf dem angegebenen Device."""
        # Compute-Type basierend auf Backend
        if self.device == "mps":
            # MPS: float32 ist am stabilsten, float16 teilweise unterstützt
            # ctranslate2 auf Apple Silicon nutzt int8/float32
            compute_type = "float32"
        elif self.device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "float32"

        print(f"--- POOL: Worker {self.worker_id} lädt Modell "
              f"'{self.model_name}' auf '{self.device}' "
              f"(compute: {compute_type}) ---")

        self.model = whisperx.load_model(
            self.model_name,
            device=self.device,
            compute_type=compute_type
        )

        print(f"--- POOL: Worker {self.worker_id} bereit ---")

    def increment_jobs(self):
        self.jobs_completed += 1


class ModelPool:
    """
    Thread-sicherer Pool von WhisperX-Modell-Instanzen.

    Ermöglicht parallele Transkription mit bis zu `pool_size` gleichzeitigen
    Jobs. Weitere Anfragen werden in einer Queue gehalten.
    """

    def __init__(self, pool_size: int = DEFAULT_POOL_SIZE):
        self.pool_size = pool_size
        self._pool: queue.Queue[ModelWorker] = queue.Queue(maxsize=pool_size)
        self._workers: list[ModelWorker] = []
        self._lock = threading.Lock()
        self._initialized = False
        self._current_model_name = None
        self._current_device = None
        self._pending_requests = 0
        self._pending_lock = threading.Lock()

    def initialize(self, model_name: str = "large-v3", device: str = "mps"):
        """
        Initialisiert den Pool mit `pool_size` Modell-Instanzen.
        Wird beim ersten `acquire()` automatisch aufgerufen.
        """
        with self._lock:
            if self._initialized and model_name == self._current_model_name \
                    and device == self._current_device:
                return

            # Memory-Check
            try:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                estimated_usage_gb = self.pool_size * 6  # ~6 GB pro Worker
                print(f"\n{'='*60}")
                print(f"POOL: System RAM: {total_ram_gb:.0f} GB")
                print(f"POOL: Geschätzte Nutzung: {estimated_usage_gb} GB "
                      f"({self.pool_size} Worker à ~6 GB)")
                if estimated_usage_gb > total_ram_gb * 0.7:
                    print(f"\u26a0\ufe0f  WARNUNG: Worker-Anzahl könnte zu hoch sein! "
                          f"Empfohlen: max {int(total_ram_gb * 0.7 / 6)} Worker")
                print(f"{'='*60}")
            except ImportError:
                print(f"\n{'='*60}")
            print(f"POOL: Initialisiere {self.pool_size} Worker mit "
                  f"'{model_name}' auf '{device}'")
            print(f"{'='*60}\n")

            # Bestehende Worker aufräumen
            self._cleanup_workers()

            # Neue Worker erstellen
            for i in range(self.pool_size):
                try:
                    worker = ModelWorker(i, model_name, device)
                    self._workers.append(worker)
                    self._pool.put(worker)
                except Exception as e:
                    print(f"--- POOL: Worker {i} konnte nicht erstellt werden: {e} ---")
                    # Bei Speichermangel mit weniger Workern weitermachen
                    break

            self._current_model_name = model_name
            self._current_device = device
            self._initialized = True
            self.pool_size = len(self._workers)  # Tatsächliche Größe

            print(f"\n--- POOL: {len(self._workers)} Worker erfolgreich "
                  f"initialisiert ---\n")

    def acquire(self, model_name: str, device: str,
                timeout: float = 120) -> ModelWorker:
        """
        Holt einen freien Worker aus dem Pool.
        Blockiert bis einer verfügbar ist oder Timeout erreicht.
        """
        # Lazy initialization
        if not self._initialized or model_name != self._current_model_name \
                or device != self._current_device:
            self.initialize(model_name, device)

        with self._pending_lock:
            self._pending_requests += 1

        try:
            print(f"--- POOL: Worker angefordert "
                  f"({self.get_available_count()}/{self.pool_size} frei, "
                  f"{self.get_queue_size()} wartend) ---")
            worker = self._pool.get(timeout=timeout)
            return worker
        except queue.Empty:
            raise TimeoutError(
                f"Kein Worker innerhalb von {timeout}s verfügbar. "
                f"Alle {self.pool_size} Worker sind belegt."
            )
        finally:
            with self._pending_lock:
                self._pending_requests -= 1

    def release(self, worker: ModelWorker):
        """Gibt einen Worker zurück in den Pool."""
        worker.increment_jobs()
        self._pool.put(worker)
        print(f"--- POOL: Worker {worker.worker_id} freigegeben "
              f"(Jobs: {worker.jobs_completed}) ---")

    def get_available_count(self) -> int:
        """Anzahl der aktuell freien Worker."""
        return self._pool.qsize()

    def get_queue_size(self) -> int:
        """Anzahl der wartenden Anfragen."""
        with self._pending_lock:
            return self._pending_requests

    def _cleanup_workers(self):
        """Räumt bestehende Worker auf."""
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
        self._workers.clear()

        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


# Globale Pool-Instanz
model_pool = ModelPool(pool_size=DEFAULT_POOL_SIZE)
