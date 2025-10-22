<div style="max-width: 700px; margin: 0 auto;"> 

# Dokumentacja techniczna programu do śledzenia obiektów

![main_program.png](img/main_program.png "Główne okno programu")

Autorzy: Bartosz Pawlaczyk, Paweł Roszak
___

## Wprowadzenie
Nasz program to zaawansowany system do śledzenia obiektów w czasie rzeczywistym, wykorzystujący modele głębokiego uczenia (YOLOv8, YOLOv11) oraz różne algorytmy śledzenia (BoT-SORT, ByteTrack, filtr Kalmana, algorytm ReID). System obsługuje zarówno strumienie wideo na żywo (RTSP), jak i pliki wideo, oferując funkcje liczenia obiektów, analizy ruchu i eksportu wyników w formatach benchmarkowych (KITTI, MOT16, HOTA).

Program został napisany w ramach odbywania praktyk na kierunku teleinformatyka na terenie uczelni Politechniki Poznańskiej w terminie 08.2025-09.2025.

### Spis treści:
1. Wykorzystane biblioteki
2. Struktura projektu
3. Schemat działania
4. Benchmarki i metryki
5. Podsumowanie
___

### Wykorzystane biblioteki
Główne zależności projektu:
- **OpenCV (cv2)** - przetwarzanie obrazu i wideo, rysowanie detekcji, odczyt z kamer i plików wideo
- **PyTorch** - tworzenie, trenowanie i uruchamianie sieci neuronowych, obsługa GPU i tensorów
- **TorchVision** - dodatek do PyTorch, wbudowane modele konwolucyjnych sieci neuronowych, transformacja obrazów, gotowe datasety do trenowania
- **Ultralytics** - trenowanie, detekcja, śledzenie obiektów
- **FilterPy** - implementacja filtru Kalmana(predykcja pozycji w systemach śledzenia)
- **Supervision** - narzędzia do annotacji wideo
- **SciPy** - optymalizacja i obliczenia naukowe
- **NumPy** - operacje na tablicach, macierzach i wektorach
- **tqdm** - pasek pokazujący progres ładowania danych np. ładowanie wideo
- **imutils** - narzędzia pomocnicze w rotacji, skalowaniu czy kadrowaniu obrazów z OpenCV
- **pandas** - biblioteka do analizy danych
- **MatPlotLib** - wizualizacja wyników w wykresach
- **Scikit-Learn** - klasyfikacja, regresja, analiza głównych składowych(PCA), normalizacja danych
- **Pillow** - odczyt, zapis, konwersja, czy zmiana rozmiaru obrazów
- **Gdown** - narzędzie do pobierania plików z Google Drive
- **PyCocoTools** - narzędzie do pracy z COCO dataset

Wszystkie biblioteki można importować za pomocą pliku requirements.txt używając komendy "pip install -r requirements.txt"

```python
import cv2
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
```

**[Wstępna konfiguracja środowiska](#wstępna-konfiguracja)**

___

### Struktura projektu
- **[licznik_pojazdow_i_osob_v3.py](#licznik_pojazdow_i_osob_v3py)** - główny skrypt uruchomieniowy 
- **[procesowanie.py](#procesowaniepy)** - przetwarzanie klatek i detekcja
- **[tracker.py](#trackerpy)** - customowy tracker obiektów
- **[filtr_Kalmana_i_ReID.py](#filtr_Kalmana_i_ReIDpy)** - implementacja filtru Kalmana i śledzenia
- **[yolo_tracker.py](#yolotrackerpy)** - integracja z trackerami YOLO
- **[KittiResultsWriterYOLO.py](#KittiResultsWriterYOLOpy)** - eksport wyników do formatu KITTI
- **[MOT16ResultsWriter.py](#MOT16ResultsWriterpy)** - eksport wyników do formatu MOT16

___

### Schemat działania

![schemat_działania.png](img/schemat_działania.jpg "Schemat działania")

#### Benchmarki i metryki
W celu śledzenia działania naszego trackera w połączeniu z wytrenowanym modelem detektora YOLOv8 użyty został benchmark TrackEval autorstwa JonathonLuiten, a danymi testowymi były treningowe datasety ze strony MotChallenge. Tracker generował bounding boxy dla osób i pojazdów znajdujących się na 5316 klatkach danych treningowych, te dane były później porównywane z ground-truthami(rzeczywiste współrzędne obiektów znajdujących się w danych klatkach). Przykładowe dane:

```
1,1,1376,485,37,28,0,11,1
2,1,1379,486,37,28,0,11,1
3,1,1382,487,38,29,0,11,1
4,1,1386,488,38,29,0,11,1
5,1,1389,490,38,29,0,11,1
6,1,1393,491,38,30,0,11,1
7,1,1396,492,39,30,0,11,1
8,1,1399,494,39,30,0,11,1
9,1,1403,495,39,30,0,11,1
```

Dane te są zapisywane w formacie(od lewej):
- **frame** - numer klatki
- **id** - unikalny identyfikator obiektu
- **bb_left** - lewy górny róg bounding box w pikselach
- **bb_top** - górny róg bounding box w pixelach
- **bb_width** - szerokość bounding boxa
- **bb_height** - wysokość bounding boxa
- **conf** - pewność detektora (od 0.0 do 1.0)
- **x** - pozycja obiektu w osi x
- **y** - pozycja obiektu w osi y
- **z** - pozycja obiektu w osi z(wysokość, odległość od kamery)
W naszym przypadku pominięta zostaje zmienna z, ponieważ wykorzystywaliśmy tylko bounding-boxy 2D

Nasze dane są następnie wykorzystywane przez benchmark TrackEval w celu ocenienia jakości działania algorytmu śledzenia obiektów. Oto przykładowe wyniki:

```
Evaluating MyTracker

    MotChallenge2DBox.get_raw_seq_data(MyTracker, MOT16-02)                0.2512 sec
    MotChallenge2DBox.get_preprocessed_seq_data(pedestrian)                0.1482 sec
    HOTA.eval_sequence()                                                   0.1375 sec
    CLEAR.eval_sequence()                                                  0.0266 sec
    Identity.eval_sequence()                                               0.0159 sec
    Count.eval_sequence()                                                  0.0000 sec
1 eval_sequence(MOT16-02, MyTracker)                                     0.5827 sec

All sequences for MyTracker finished in 0.58 seconds

HOTA: MyTracker-pedestrian         HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
MOT16-02                           4.8854    7.0496    3.4689    9.893     18.479    3.7316    50.221    73.791    5.7988    8.3975    49.694    4.173
COMBINED                           4.8854    7.0496    3.4689    9.893     18.479    3.7316    50.221    73.791    5.7988    8.3975    49.694    4.173

CLEAR: MyTracker-pedestrian        MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag      
MOT16-02                           -31.296   66.557    -30.387   11.574    21.619    5.5556    12.963    81.481    -35.167   2064      15769     7483      162       3         7         44        31
COMBINED                           -31.296   66.557    -30.387   11.574    21.619    5.5556    12.963    81.481    -35.167   2064      15769     7483      162       3         7         44        31

Identity: MyTracker-pedestrian     IDF1      IDR       IDP       IDTP      IDFN      IDFP
MOT16-02                           3.1848    2.4449    4.5669    436       17397     9111
COMBINED                           3.1848    2.4449    4.5669    436       17397     9111
```

### Wyjaśnienie znaczenia uzyskanych wyników

# HOTA:
HOTA - Higher Order Tracking Accuracy, główny wynik(0-100), średnia jakość tracking uwzględniająca zarówno detekcję, jak i przypisywanie ID

DetA - Detection Accuracy, jakość detekcji (czy obiekty zostały znalezione w danych klatkach)

AssA - Association Accuracy, jakość asocjacji (czy śledzone ID są poprawnie przypisane przez cały czas pomiędzy różnymi klatkami)

DetRe - Detection Recall, procent obiektów wykrytych (ile z Ground Truth zostało znalezionych)

DetPr - Detection Precision, procent detekcji, które są poprawne (jak duzo fałszywych detekcji)

AssRe/AssPr - odpowiednik recall/precision, ale dla ciągłości śledzenia (np. czy tracker potrafi utrzymać to samo ID)

LocA - Localization Accuracy, średnia dokładność lokalizacji (Intersection over Union między predykcją Ground Truth)

RHOTA - wariant HOTA kładący większy nacisk na asocjację

HOTA(0), LocA(0), HOTALocA(0) - metryki przy progu IoU=0 (czyli bez oceny dokładności lokalizacji)

# CLEAR MOT(MOTA/MOTP)
Starszy standard z MOT Challenge

MOTA - Multiple Object Tracking Accuracy, uwzględnia false positives, false negatives, ID-switches (może być ujemne, jeśli tracker popełnia więcej błędów niż poprawnych detekcji)
MOTP - Multiple Object Tracking Precision, pokazuje, jak dokładne są lokalizacje (średnie IoU dla prawidłowych dopasowań)

MODA - podobne do MOTA, ale bez uwzględniania ID-switchów

CLR_Re/CLR_Pr - recall i precision (ile obiektów znalezionych vs ile false positives)

MTR/PTR/MLR - proporcja trajektorii:
 -**MTR** - Mostly Tracked(>80% czasu śledzone poprawnie)
 -**PTR** - Partially Tracked(20-80%)
 -**MLR** - Mostly Lost(<20%)

IDSW - liczba ID Switches (ile razy tracker przypisał inne ID do tego samego obiektu)

Frag - fragmentacja (ile razy trajektoria się urywała)

# Identity
Metryki pochodzące z ID metrics (IDF1, IDP, IDR)

IDF1 - F1-score liczony na poziomie ID, łączy ID Recall i ID Precision, mówi jaka część poprawnych trajektorii jest odtworzona w całości

IDR (ID Recall) - Recall dla identyfikacji, % ground-truth detekcji, które zostały poprawnie przypisane do jakiegoś ID

IDP (ID Precision) - Precision dla identyfikacji,% predykcji, które mają poprawne ID

IDTP/IDFN/IDFP - liczby absolutne:
 -**IDTP** - Identity True Positives (ile detekcji z poprawnym ID)
 -**IDFN** - Identity False Negatives (ile GT nie miało dopasowania)
 -**IDFP** - Identity False Positives (ile predykcji miało złe ID)

# Count
Zestawienie zliczające element w danych i predykcjach

Dets - liczba detekcji, które tracker wypuścił (predykcje)

GT_Dets - liczba detekcji w ground-truth (ile obiektów naprawdę było)

IDs - liczba unikalnych identyfikatorów nadanych przez twój tracker

GT_IDs - liczba unikalnych ID w ground-truth (ile rzeczywistych trajektorii)



#### Podsumowanie



</div>

<div style="max-width: 700px; margin: 0 auto;"> 

[Powrót do strony głównej](#wprowadzenie)

### filtr_Kalmana_i_ReID\.py
Ten plik implementuje klasę Track, która wykorzystuje filtr Kalmana do śledzenia obiektów i zarządzania ich stanem.
___

#### Inicjalizacja i importy:
Import bibliotek do obliczeń numerycznych, przetwarzania obrazu, filtra Kalmana i logowania.

```python
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cosine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

#### Klasa Track - inicjalizacja:
Inicjalizuje obiekt śledzenia z unikalnym ID, przechowuje historię pozycji, informacje o czasie widzenia, klasę obiektu, embedding do re-identyfikacji i inicjalizuje filtr Kalmana.

```python
class Track:
    def __init__(self, detection, frame_number, track_id, embedding=None, class_id=None, bbox=None):
        self.track_id = track_id
        self.positions = []  # We'll store Kalman filter states instead of raw positions
        self.first_seen = frame_number
        self.last_seen = frame_number
        self.class_id = class_id
        self.logged = False
        self.embedding = embedding.flatten() if embedding is not None else None
        self.disappeared = 0
        self.frame_number = frame_number
        
        # Initialize Kalman filter
        self.kf = self._initialize_kalman_filter(bbox if bbox is not None else detection)
        
        # Store initial state
        self.positions.append(self.kf.x.copy())
```

#### Inicjalizacja filtra Kalmana:
Konwertuje detekcję na format [środek_x, środek_y, szerokość, wysokość] i inicjalizuje 8-wymiarowy stan filtra Kalmana z pozycją i prędkością.

```python
def _initialize_kalman_filter(self, detection):
    # Convert detection to [x_center, y_center, width, height] format
    if len(detection) == 4:  # It's a bbox [x1, y1, x2, y2]
        x1, y1, x2, y2 = detection
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
    else:  # It's a point detection [x, y]
        cx, cy = detection[:2]
        w, h = 50, 100  # Default size
    
    # 8-dimensional state: [cx, cy, w, h, vx, vy, vw, vh]
    kf = KalmanFilter(dim_x=8, dim_z=4)
    
    # Initial state
    kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
```

#### Macierz przejścia stanu:
Definiuje macierz przejścia stanu która modeluje ruch obiektu z uwzględnieniem prędkości.

```python
# State transition matrix - properly model movement
dt = 1.0  # Time step (assuming 1 frame)
kf.F = np.array([
    [1, 0, 0, 0, dt, 0, 0, 0],
    [0, 1, 0, 0, 0, dt, 0, 0],
    [0, 0, 1, 0, 0, 0, dt, 0],
    [0, 0, 0, 1, 0, 0, 0, dt],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
], dtype=np.float32)
```

#### Funkcja pomiaru:
Określa które zmienne stanu są mierzone - tylko pozycja i rozmiar, bez prędkości.

```python
# Measurement function - we measure [cx, cy, w, h]
kf.H = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]
], dtype=np.float32)
```

#### Macierze kowariancji:
Konfiguruje macierze kowariancji: początkową niepewność, szum pomiarowy i szum procesowy.

```python
# Covariance matrices - need proper tuning
kf.P *= 100.0  # Initial uncertainty
kf.R *= 5.0    # Measurement noise
kf.Q *= 0.1    # Process noise

return kf
```

#### Właściwość pozycji:
Konwertuje stan filtra Kalmana z formatu [środek, rozmiar] z powrotem na format bounding box [x1, y1, x2, y2].

```python
@property
def position(self):
    # Get current state from Kalman filter
    state = self.kf.x
    cx, cy, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
    
    # Convert back to [x1, y1, x2, y2] format
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    
    return [x1, y1, x2, y2]
```

#### Predykcja stanu:
Wykonuje predykcję następnego stanu na podstawie modelu ruchu i zapisuje przewidywaną pozycję.

```python
def predict(self):
    # Predict next state
    self.kf.predict()
    self.positions.append(self.kf.x.copy())
    return self.position
```

#### Aktualizacja stanu:
Aktualizuje filtr Kalmana nowym pomiarem, resetuje licznik zniknięć i aktualizuje embedding oraz klasę obiektu.

```python
  def update(self, detection, embedding=None, class_id=None, bbox=None):
        # Convert detection to measurement format [cx, cy, w, h]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w/2, y1 + h/2
        else:
            cx, cy = detection[:2]
            w, h = 50, 100  # Default size
        
        measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        
        # Update Kalman filter
        self.kf.update(measurement)
        self.positions.append(self.kf.x.copy())
        
        self.last_seen = self.frame_number
        self.disappeared = 0

        if embedding is not None:
            self.embedding = embedding.flatten()
        if class_id is not None:
            self.class_id = class_id
```



[Powrót do strony głównej](#wprowadzenie)



</div>

<div style="max-width: 700px; margin: 0 auto;"> 

### licznik_pojazdow_i_osob_v3.py

___

#### Inicjalizacja komponentów:

```python
import cv2
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
```

#### Informacje o środowisku CUDA
```python
if __name__ == "__main__":
    logger.info(f"PyTorch wersja: {torch.__version__}")
    logger.info(f"CUDA dostępne: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"Liczba GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    main()
```


#### Konfiguracja źródła wideo:

```python
# Szybkie ładowanie modelu
logger.info("Ładowanie modelu YOLO...")`
model, device = load_yolov8_fast()

# Inicjalizacja trackera
tracker = FastObjectTracker() if use_custom_tracker else None

# Inicjalizacja źródła wideo
if args.video_path:
    logger.info(f"Użycie lokalnego wideo: {args.video_path}")
    video_source = args.video_path
else:
    rtsp_url = "rtsp://admin:12345@150.254.16.85/live.sdp"
    logger.info(f"Użycie strumienia RTSP: {rtsp_url}")
    video_source = rtsp_url

video_stream = VideoStream(video_source)
video_stream.start()
```

#### Główna pętla przetwarzania:
Pobieranie klatek i zarządzanie historią klatek do analizy ruchu.

```python
    while True:
        ret, frame = video_stream.read()
        
        if not ret or frame is None:
            logger.warning("Brak klatki, ponawiam próbę...")
            time.sleep(0.1)
            continue

        frames_history[frame_count] = frame.copy()
        if len(frames_history) > 50:
            frames_history.pop(frame_count - 50, None)
```

#### Przetwarzanie z oboma trackerami
Równoległe przetwarzanie klatki przez oba trackery i zapisywanie wyników.

```python
if use_custom_tracker and use_yolo_tracker:
    frame1 = frame.copy()
    frame2 = frame.copy()
    
    custom_results = process_frame_fast(model, frame1, tracker, frame_count, device)
    for track_id, track in tracker.tracks.items():
        if track.is_moving(frames_history):
            if track.first_seen == frame_count and track.disappeared == 0 and not track.logged:
                save_active_track_to_file(track, "custom")
                track.logged = True
    
    yolo_results = process_frame_yolo_builtin(model, frame2)
```
#### Wyświetlanie wyników i pomiar FPS

Obliczanie i wyświetlanie FPS oraz prezentacja wyników w oknie.
```python
current_time_sec = time.time()
fps = 1.0 / (current_time_sec - last_time) if frame_count > 0 else 0
last_time = current_time_sec
fps_list.append(fps)

cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow("Tracker Comparison", display_frame)
```

[Powrót do strony głównej](#wprowadzenie)




</div>

<div style="max-width: 700px; margin: 0 auto;"> 

[Powrót do strony głównej](#wprowadzenie)

### procesowanie\.py
Ten plik zawiera funkcje do przetwarzania klatek wideo i wykrywania obiektów przy użyciu modelu YOLO.
___

#### Funkcja zapisu do pliku:
Zapisuje informacje o wykrytych obiektach do pliku tekstowego w formacie: 
`timestamp - typ_obiektu - ID: track_id.`

```python
def save_to_file(obj_type, track_id, timestamp):
    """Zapisywanie zdarzenia do pliku txt"""
    filename = "wyniki_detekcji.txt"
    try:
        with open(filename, "a", encoding='utf-8') as f:
            f.write(f"{timestamp} - {obj_type} - ID: {track_id}\n")
    except Exception as e:
        logger.error(f"Błąd zapisu do pliku: {e}")
```

#### Przetwarzanie wstępne klatki:
Pobranie oryginalnych wymiarów klatki, przeskalowanie do 640x640, konwersja BGR→RGB, transpozycja kanałów i zapewnienie ciągłości pamięci.

```python
try:
    h0, w0 = frame.shape[:2]
    input_frame = cv2.resize(frame, (640, 640))
    input_frame = input_frame[:, :, ::-1].transpose(2, 0, 1)
    input_frame = np.ascontiguousarray(input_frame)
```

#### Przygotowanie tensora wejściowego:
Konwersja do tensora PyTorch, przeniesienie na odpowiednie urządzenie (CPU/GPU), normalizacja wartości pikseli do zakresu [0,1] i dodanie wymiaru batch.

```python
try:
    input_tensor = torch.from_numpy(input_frame).to(device)
    input_tensor = input_tensor.half() if device.type == 'cuda' else input_tensor.float()
    input_tensor /= 255.0
    if input_tensor.ndimension() == 3:
        input_tensor = input_tensor.unsqueeze(0)
```

#### Wykonywanie predykcji modelu:
Wykonanie inferencji modelu YOLO bez obliczania gradientów (tryb ewaluacji) i pominięcie verbose output.

```python
with torch.no_grad():
    pred = model(input_tensor, verbose=False)[0]
```

#### Przetwarzanie wyników detekcji:
Inicjalizacja list dla bboxów i klas, sprawdzenie czy są detekcje, przeniesienie wyników na CPU i obliczenie współczynników skalowania.

```python
bboxes, class_ids = [], []

if pred is not None and pred.boxes is not None:
    boxes = pred.boxes.cpu().numpy()
    scale_x, scale_y = w0 / 640.0, h0 / 640.0
```

#### Iteracja przez wykryte obiekty:
Ekstrakcja współrzędnych bboxa, confidence i klasy, pominięcie detekcji poniżej progu ufności.

```python
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    if conf < conf_thresh:
        continue
```

#### Skalowanie bboxów do oryginalnego rozmiaru:
Przeskalowanie współrzędnych do oryginalnych wymiarów klatki, obliczenie środka bboxa i przygotowanie danych detekcji.

```python
x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
detection_data = (cx, cy, cls, conf, (x1, y1, x2, y2))
```

#### Kategoryzacja detekcji:
Dodanie detekcji osób (klasa 0) i pojazdów (klasy 2,3,5,7) do odpowiednich list oraz przygotowanie danych dla trackera.

```python
if cls == 0:
    results['people_detections'].append(detection_data)
elif cls in [2, 3, 5, 7]:
    results['vehicle_detections'].append(detection_data)

bboxes.append([x1, y1, x2, y2])
class_ids.append(cls)
```

#### Aktualizacja trackera:
Aktualizacja trackera z wykrytymi bboxami (jako float32), klasami i numerem klatki.

```python
tracker.update(
    detections=np.array(bboxes, dtype=np.float32) if bboxes else np.empty((0,4), dtype=np.float32),
    embeddings=None,
    class_ids=class_ids,
    frame_number=frame_count
)
```

#### Aktualizacja trackera:
Aktualizacja trackera z wykrytymi bboxami (jako float32), klasami i numerem klatki.

```python
    results['people_count'] = len(results['people_detections'])
    results['vehicle_count'] = len(results['vehicle_detections'])
    results['tracks'] = tracker.tracks

except Exception as e:
    logger.error(f"Błąd w process_frame_fast: {e}", exc_info=True)

return results
```

#### Finalizacja wyników:
Uzupełnienie liczników, dodanie śledzeń do wyników i obsługa błędów z logowaniem.

```python
    results['people_count'] = len(results['people_detections'])
    results['vehicle_count'] = len(results['vehicle_detections'])
    results['tracks'] = tracker.tracks

except Exception as e:
    logger.error(f"Błąd w process_frame_fast: {e}", exc_info=True)

return results
```

[Powrót do strony głównej](#wprowadzenie)


</div>

<div style="max-width: 700px; margin: 0 auto;"> 

[Powrót do strony głównej](#wprowadzenie)

### tracker\.py
Ten plik implementuje zaawansowany tracker obiektów wykorzystujący algorytm węgierski do dopasowywania detekcji do istniejących śledzeń.
___

#### Inicjalizacja i importy:
Import niezbędnych bibliotek do obliczeń numerycznych, optymalizacji przypisań i logowania.

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
from filtr_Kalmana_i_ReID import Track
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```


#### Klasa główna tracker'a:
Inicjalizacja tracker'a z parametrami: maksymalna liczba klatek zniknięcia, progi IoU dla różnych rozmiarów obiektów i waga re-identyfikacji.

```python
class FastObjectTracker:
    def __init__(self, max_disappeared=30, iou_threshold_small=0.5,
                 iou_threshold_medium=0.6, iou_threshold_large=0.7,
                 reid_weight=0.5):
        self.tracks = {}  # track_id -> Track
        self.all_time_track_count = 0
        self.max_disappeared = max_disappeared
        self.iou_threshold_small = iou_threshold_small
        self.iou_threshold_medium = iou_threshold_medium
        self.iou_threshold_large = iou_threshold_large
        self.reid_weight = reid_weight
```

#### Funkcja resetująca:
Czyści wszystkie śledzenia zachowując konfigurację parametrów.

```python
def reset(self):
    """Resetuje tracker, zachowując parametry ale usuwając wszystkie tory"""
    self.tracks = {}
    self.all_time_track_count = 0
```

#### Funkcja obliczania IoU:
Oblicza Intersection over Union (IoU) między dwoma bounding boxami z zabezpieczeniem przed dzieleniem przez zero.

```python
@staticmethod
def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
```

#### Funkcja odległości kosinusowej:
Oblicza odległość kosinusową między embeddingami do re-identyfikacji z obsługą przypadków brzegowych.

```python
@staticmethod
def _cosine_distance(a, b):
    if a is None or b is None:
        return 1.0
    a = np.asarray(a)
    b = np.asarray(b)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return 1.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return 1.0 - np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
```

#### Adaptacyjny próg IoU:
Dostosowuje próg IoU w zależności od rozmiaru obiektu - mniejsze obiekty mają niższe wymagania.

```python
def _adaptive_iou_threshold(self, bbox):
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    size = (w+h)/2
    if size < 30: return self.iou_threshold_small
    elif size < 80: return self.iou_threshold_medium
    else: return self.iou_threshold_large
```

#### Główna funkcja aktualizacji:
Inicjalizuje domyślne wartości dla embeddingów i klas jeśli nie są podane.

```python
def update(self, detections, embeddings=None, class_ids=None, frame_number=0):
    if embeddings is None:
        embeddings = [None] * len(detections)
    if class_ids is None:
        class_ids = [None] * len(detections)
```

#### Predykcja istniejących śledzeń:
Dla każdego istniejącego track'a wykonuje predykcję pozycji na podstawie filtra Kalmana.

```python
predicted_boxes = []
track_ids = []
track_classes = []
for tid, track in self.tracks.items():
    pred_box = track.predict()  # [x1,y1,x2,y2] (już poprawione w Track)
    predicted_boxes.append(pred_box)
    track_ids.append(tid)
    track_classes.append(track.class_id)
```

#### Algorytm węgierski (Hungarian):
Tworzy macierz kosztów łącząc odległość IoU i odległość re-identyfikacji, następnie rozwiązuje problem przypisania.

```python
if len(predicted_boxes) > 0 and len(detections) > 0:
    cost_matrix = np.zeros((len(predicted_boxes), len(detections)), dtype=np.float32)

    for i, pred in enumerate(predicted_boxes):
        for j, det in enumerate(detections):
            # różne klasy = bardzo duży koszt
            if track_classes[i] is not None and class_ids[j] is not None and track_classes[i] != class_ids[j]:
                cost_matrix[i, j] = 1e6
                continue

            iou = self._iou(pred, det)
            iou_dist = 1 - iou
            reid_dist = self._cosine_distance(self.tracks[track_ids[i]].embedding, embeddings[j])
            total_cost = self.reid_weight * reid_dist + (1 - self.reid_weight) * iou_dist

            if np.isnan(total_cost) or np.isinf(total_cost):
                total_cost = 1.0

            cost_matrix[i, j] = total_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
```

#### Dopasowywanie detekcji do śledzeń:
Aktualizuje dopasowane track'i i zaznacza je jako dopasowane.

```python
for r, c in zip(row_ind, col_ind):
    iou = self._iou(predicted_boxes[r], detections[c])
    adaptive_iou = self._adaptive_iou_threshold(detections[c])
    if iou >= adaptive_iou:
        tid = track_ids[r]
        self.tracks[tid].update(
            detections[c],
            embedding=embeddings[c],
            class_id=class_ids[c],
            bbox=detections[c]
        )
        matched_tracks.add(tid)
        matched_detections.add(c)
```

#### Obsługa niedopasowanych śledzeń:
Zwiększa licznik zniknięć dla niedopasowanych track'ów i usuwa te które zniknęły na zbyt długo.

```python
for tid, track in list(self.tracks.items()):
    if tid not in matched_tracks:
        track.disappeared += 1
        if track.disappeared > self.max_disappeared:
            del self.tracks[tid]
```

#### Tworzenie nowych śledzeń:
Tworzy nowe track'i dla niedopasowanych detekcji.

```python
for i, det in enumerate(detections):
    if i not in matched_detections:
        new_track = Track(det, frame_number, self.all_time_track_count,
                        embedding=embeddings[i], class_id=class_ids[i], bbox=det)
        self.tracks[self.all_time_track_count] = new_track
        self.all_time_track_count += 1
```

#### Usuwanie duplikatów:
Usuwa zduplikowane track'i które śledzą ten sam obiekt (IoU > 0.7) i zwraca zaktualizowane śledzenia.

```python
seen_boxes = []
for tid, track in list(self.tracks.items()):
    if track.position is None:
        continue
    duplicate = False
    for sb in seen_boxes:
        if self._iou(track.position, sb) > 0.7:
            duplicate = True
            break
    if duplicate:
        del self.tracks[tid]
    else:
        seen_boxes.append(track.position)

return self.tracks
```


[Powrót do strony głównej](#wprowadzenie)



</div>

<div style="max-width: 700px; margin: 0 auto;"> 

[Powrót do strony głównej](#wprowadzenie)

### Wstępna konfiguracja

Aby korzystać z dobrodziejstw technologii CUDA potrzebne jest ręczne skompilowanie biblioteki opencv wraz z tą technologią, ponieważ wersje prekompilowane nie chcą poprawnie działać.
___

#### Wymagania wstępne:
- system Windows 10/11
- Python 3.10.0+
- NVIDIA GPU z obsługą CUDA - opcjonalnie
- NVIDIA Drivers (najnowsze)
- cmake
- Visual Studio 2022

#### Instalacja sterowników NVIDIA:

![sterowniki.png](img/sterowniki.png "Formularz sterowników graficznych")
 [pobierz sterowniki](https://nvidia.com/pl-pl/drivers/)

Należy pobrać i zainstalować sterowniki w najnowszej wersji z oficjalnej strony producenta układu graficznego.

#### Sprawdzenie wersji CUDA dla GPU:

[wsparcie cuda](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)

Wchodząc w powyższy link należy sprawdzić jaką wersję SDK wspiera architektura karty. Ponadto należy sprawdzić wersję Compute capability.

Dla naszej karty graficznej - Nvidia GeForce RTX 5060 - abędzie to odpowiednio:
- CUDA SDK: 13.0
- Compute capability : 12.0

#### Instalacja CUDA Toolkit:
![cuda-tooltit.png](img/cuda-toolkit.png "CUDA Toolkit")
[pobierz CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

Wchodząc w powyższy link należy pobrać i zainstalować odpowiednią wersję CUDA Toolkit.

#### Instalacja Pythona:
![python.png](img/python.png "Python setup screen")

Zalecana wersja to 3.10.0. Można ją znaleźć w sekcji "Looking for a specific release?"
Wybierz wersję Windows installer (64-bit).

[pobierz Python](https://www.python.org/downloads/)

Należy pamiętać, by zaznaczyć opcję "Add Python to PATH".

#### Pobieranie opencv do kompilacji:
[pobierz opencv](https://github.com/opencv/opencv)
[pobierz opencv contrib](https://github.com/opencv/opencv_contrib)

Pobrane pliki należy wypakować i umieścić w następującej lokalizacji:
> /opencv_tutorial/build/ 
/opencv_tutorial/opencv/
/opencv_tutorial/opencv_contrib/

Folder build ma być pusty - to w nim nasza kompilacja będzie się "budować".

#### Konfiguracja ustawień kompilacji w cmake:

Uruchom program cmake i podaj potrzebne ścieżki:
**Where is the source code** : /opencv_tutorial/opencv
**Where is to build the binaries** : /opencv_tutorial/build

Naciśnij "configure" i ustaw w następujący sposób:

![cmake_configure1.png](img/cmake_configure1.png "Cmake configure")

Następnie ustaw następujące parametry:
| Nazwa    | Wartość |
| -------- | ------- |
| OPENCV_EXTRA_MODULES_PATH  |  ścieżka katalogu modułów w wyodrębnionym pliku opencv_contrib   |
| OPENCV_PYTHON3_VERSION    | TAK    |
| WITH_CUDA    | TAK    |
| ENABLE_FAST_MATH | TAK     |


Sprawdź parametry PYTHON3_..., aby ścieżki odpowiadały twojemu Pythonowi!

Naciśnij Configure.

Następnie wprowadź dwa parametry:

| Nazwa    | Wartość |
| -------- | ------- |
| CUDA_FAST_MATH  |  TAK   |
| CUDA_ARCH_BIN    | x.x    |

gdzie x.x odpowiada wersji architektury karty graficznej, którą sprawdziliśmy wcześniej. Dla nas będzie to 12.0.

Naciśnij Configure. Gdy program zakończy proces, naciśnij Generate.

Następnie wydaj komendy w cmd:
>python
cmake --build "twoja ścieżka do folderu build" --target INSTALL --config Release

Rozpocznie się proces kompilacji i instalacji opencv z dodanym wsparciem CUDA.

Po zakończeniu powinieneś uzyskać taki widok:

![koniec_instalacji.png](img/koniec_instalacji.png "Koniec instalacji openv")

Aby sprawdzić działanie programu można uruchomić następujący kod:

```python
import cv2

print(cv2.getBuildInformation())
print(f"OpenCV version: {cv2.__version__}")
print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
```

W przypadku braku wykrycia lub innych błędów, konieczne może być dodanie dodatkowych ścieżek w __init\__:

>os.add_dll_directory(r"C:\Python38\opencv\build\install\x64\vc16\bin")
os.add_dll_directory(r"C:\Python38\opencv\build\bin\Release")
os.add_dll_directory(r"C:\Python38\opencv\build\install\x64\vc16\bin")
os.add_dll_directory(r"C:\Python38\opencv\build\bin\Release")

<br>

Poradnik oparty na poniższym źródle:
[OpenCV-CUDA-installation](https://github.com/chrismeunier/OpenCV-CUDA-installation)


[Powrót do strony głównej](#wprowadzenie)



</div>
