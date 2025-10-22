import threading
import logging
import cv2
import time
import time
import threading
from queue import Queue

logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, rtsp_url, queue_size=3):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.connected = False
        
    def read(self):
        """Pobierz najnowszą klatkę z kolejki"""
        try:
            frame = self.frame_queue.get(timeout=1)
            return True, frame
        except:
            return False, None
        
    def stop(self):
        """Zatrzymaj strumień wideo"""
        self.stopped = True
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception as e:
                logger.error(f"Błąd przy zwalnianiu strumienia wideo: {e}")
        
    def start(self):
        """Uruchom wątek do odczytu klatek wideo"""
        self.stopped = False
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self
        
    def update(self):
        """Ciągle odczytuj klatki z kamery i umieszczaj je w kolejce"""
        attempts = 0
        max_attempts = 5
        
        while not self.stopped and attempts < max_attempts:
            if not self.connected:
                # Próba różnych backendów w kolejności
                backends = [
                    cv2.CAP_FFMPEG,
                    cv2.CAP_GSTREAMER,
                    cv2.CAP_ANY
                ]
                
                for backend in backends:
                    self.cap = cv2.VideoCapture(self.rtsp_url, backend)
                    if self.cap.isOpened():
                        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        print(f"Video dimensions: {width}x{height}")
                        logger.info(f"Połączono z kamerą RTSP używając backendu: {backend}")
                        self.connected = True
                        break
                
                if not self.connected:
                    attempts += 1
                    logger.error(f"Nie udało się połączyć z kamerą (próba {attempts}/{max_attempts})")
                    time.sleep(2)
                    continue
                
                # Optymalne parametry dla strumienia
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            
            # Odczytaj klatkę
            try:
                ret, frame = self.cap.read()
            except cv2.error as e:
                logger.error(f"Błąd OpenCV przy odczycie klatki: {e}")
                self.connected = False
                self.cap.release()
                continue

            if not ret or frame is None:
                logger.warning("Nie udało się wczytać klatki")
                self.connected = False
                self.cap.release()
                continue
                
            # Jeśli kolejka jest pełna, usuń najstarszą klatkę
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
                    
            # Dodaj klatkę do kolejki
            self.frame_queue.put(frame)
            
        self.cap.release()