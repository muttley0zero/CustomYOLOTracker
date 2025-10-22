import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import logging
import cv2
import imutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= Track =================
class Track:
    def __init__(self, detection, frame_number, track_id, embedding=None, class_id=None, 
                 bbox=None, confidence=None, ema_alpha=0.3, min_w=20, min_h=40, max_scale_change=1.5, min_hits=3):
        self.track_id = track_id
        self.positions = []
        self.first_seen = frame_number
        self.last_seen = frame_number
        self.class_id = class_id
        self.logged = False
        self.embedding = embedding.flatten() if embedding is not None else None
        self.disappeared = 0
        self.frame_number = frame_number
        self.confidence = confidence or 1.0
        self.ema_alpha = ema_alpha
        self.ema_state = None
        self.min_w = min_w
        self.min_h = min_h
        self.max_scale_change = max_scale_change
        self.min_hits = min_hits
        self.hits = 1
        self.age = 1
        self.prev_frame = None
        self.moving_bbox = None
        
        # Inicjalizacja Kalman Filter
        self.kf = self._initialize_kalman_filter(bbox if bbox is not None else detection)
        self.positions.append(self.kf.x.copy())

    def _initialize_kalman_filter(self, detection):
        if len(detection) == 4:
            x1, y1, x2, y2 = detection
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w/2, y1 + h/2
        else:
            cx, cy = detection[:2]
            w, h = 50, 100

        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8,1)
        dt = 1.0
        kf.F = np.array([
            [1,0,0,0,dt,0,0,0],
            [0,1,0,0,0,dt,0,0],
            [0,0,1,0,0,0,dt,0],
            [0,0,0,1,0,0,0,dt],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ], dtype=np.float32)
        kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ], dtype=np.float32)
        kf.P *= 100.0
        kf.R *= 5.0
        kf.Q *= 0.1
        return kf

    @property
    def position(self):
        state = self.kf.x
        cx, cy, w, h = state[0,0], state[1,0], state[2,0], state[3,0]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def predict(self):
        self.kf.predict()
        cx, cy, w, h = self.kf.x[:4, 0]
        # EMA tylko na pozycję
        if self.ema_state is None:
            self.ema_state = np.array([cx, cy])
        else:
            self.ema_state = self.ema_alpha * np.array([cx, cy]) + (1 - self.ema_alpha) * self.ema_state
        self.kf.x[0,0], self.kf.x[1,0] = self.ema_state

        # ograniczenie minimalnych wymiarów
        w = max(w, self.min_w)
        h = max(h, self.min_h)
        # ograniczenie zmiany w stosunku do poprzedniego stanu
        if self.positions:
            prev_w, prev_h = self.positions[-1][2,0], self.positions[-1][3,0]
            w = np.clip(w, prev_w / self.max_scale_change, prev_w * self.max_scale_change)
            h = np.clip(h, prev_h / self.max_scale_change, prev_h * self.max_scale_change)
        self.kf.x[2,0] = w
        self.kf.x[3,0] = h

        self.positions.append(self.kf.x.copy())
        self.age += 1
        self.disappeared += 1
        return self.position

    def update(self, detection, embedding=None, class_id=None, bbox=None, confidence=None):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w/2, y1 + h/2
        else:
            cx, cy = detection[:2]
            w, h = 50, 100
        # EMA na wymiary
        if self.positions:
            prev_w, prev_h = self.positions[-1][2,0], self.positions[-1][3,0]
            w = self.ema_alpha * w + (1 - self.ema_alpha) * prev_w
            h = self.ema_alpha * h + (1 - self.ema_alpha) * prev_h
        measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(4,1)
        self.kf.update(measurement)
        self.positions.append(self.kf.x.copy())
        self.last_seen = self.frame_number
        self.disappeared = 0
        self.hits += 1
        if embedding is not None:
            self.embedding = embedding.flatten()
        if class_id is not None:
            self.class_id = class_id
        if confidence is not None:
            self.confidence = float(confidence)
    
    def is_moving(self, frame):
        """
        Sprawdza, czy obiekt w aktualnej klatce się porusza, używając prostego detektora ruchu
        opartego na różnicy między kolejnymi klatkami.

        Parametry:
        -----------
        frame : np.ndarray
            Aktualna klatka w formacie BGR.

        Zwraca:
        -------
        moving : bool
            True, jeśli obiekt się porusza.
        bbox : tuple lub None
            Współrzędne prostokąta (x1, y1, x2, y2) obejmującego poruszający się obiekt,
            lub None jeśli ruch nie został wykryty.
        """

        if frame is None:
            # Brak klatki - nie można wykryć ruchu
            return False, None

        # 1. Konwersja do skali szarości i wygładzenie w celu redukcji szumów
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray = imutils.resize(gray, width=500)

        # 2. Jeśli brak poprzedniej klatki, ustaw aktualną jako referencyjną
        if not hasattr(self, 'prev_frame') or self.prev_frame is None:
            self.prev_frame = gray
            return False, None

        # 3. Różnica między aktualną a poprzednią klatką
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 4. Wyszukiwanie konturów
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        moving = False
        bbox = None

        for c in cnts:
            if cv2.contourArea(c) < getattr(self, 'min_area', 500):
                continue
            # Znaleziono znaczący kontur, uznajemy obiekt za poruszający się
            x, y, w, h = cv2.boundingRect(c)
            bbox = (x, y, x + w, y + h)
            moving = True
            break  # bierzemy tylko pierwszy większy kontur

        # 5. Aktualizacja poprzedniej klatki z EMA dla stabilności
        self.prev_frame = cv2.addWeighted(gray,
                                        getattr(self, 'ema_alpha', 0.3),
                                        self.prev_frame,
                                        1 - getattr(self, 'ema_alpha', 0.3),
                                        0)

        # 6. Zachowanie wykrytego prostokąta dla późniejszego wykorzystania
        self.moving_bbox = bbox

        return moving, bbox
