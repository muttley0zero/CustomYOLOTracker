# CustomYOLOTracker

<div style="max-width: 700px; margin: 0 auto;"> 

# Dokumentacja techniczna programu do śledzenia obiektów

![main_program.png](img/main_program.png "Główne okno programu")

Autorzy: Bartosz Pawlaczyk, Paweł Roszak
___

## Wprowadzenie
Nasz program to zaawansowany system do śledzenia obiektów w czasie rzeczywistym, wykorzystujący modele głębokiego uczenia (YOLOv8) oraz różne algorytmy śledzenia (BoT-SORT, ByteTrack, filtr Kalmana). System obsługuje zarówno strumienie wideo na żywo (RTSP), jak i pliki wideo, oferując funkcje liczenia obiektów, analizy ruchu i eksportu wyników w formatach benchmarkowych (KITTI, MOT16).

Program został napisany w ramach odbywania praktyk na kierunku teleinformatyka na terenie uczelni Politechniki Poznańskiej w terminie 08.2025-09.2025.

### Spis treści:
1. Wykorzystane biblioteki
2. Struktura projektu
3. Benchmarki i metryki
4. Podsumowanie
___

### Wykorzystane biblioteki
Główne zależności projektu:
- **OpenCV (cv2)** - przetwarzanie obrazu i wideo
- **PyTorch** - framework deep learning
- **Ultralytics YOLO** - detekcja obiektów
- **FilterPy** - implementacja filtru Kalmana
- **Supervision** - narzędzia do annotacji wideo
- **SciPy** - optymalizacja i obliczenia naukowe

```python
import cv2
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
```

**[Wstępna konfiguracja środowiska](wstepna_konfiguracja.md)**

___

### Struktura projektu
- **[licznik_pojazdow_i_osob_v3.py](licznik_pojazdow_i_osob_v3.py.md)** - główny skrypt uruchomieniowy 
- **[procesowanie.py](procesowanie.py.md)** - przetwarzanie klatek i detekcja
- **[tracker.py](tracker.py.md)** - customowy tracker obiektów
- **[filtr_Kalmana_i_ReID.py](filtr_Kalmana_i_ReID.py.md)** - implementacja filtru Kalmana i śledzenia
- **[yolo_tracker.py](yolo_tracker.py.md)** - integracja z trackerami YOLO
- **[KittiResultsWriterYOLO.py](KittiResultsWriterYOLO.py.md)** - eksport wyników do formatu KITTI
- **[MOT16ResultsWriter.py](MOT16ResultsWriter.py.md)** - eksport wyników do formatu MOT16

#### Benchmarki i metryki





#### Podsumowanie



</div>
