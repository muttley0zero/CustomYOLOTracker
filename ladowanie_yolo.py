# ladowanie_yolo.py
import torch
import logging
import os
import argparse
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def load_yolov8_fast(model_name="yolov8n.pt", conf_threshold=0.4):
    """Ładuje model YOLOv8 i przygotowuje do pracy z wbudowanym trackerem."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)

    # Sprawdź czy plik modelu istnieje
    if not os.path.exists(model_path):
        logger.warning(f"Plik modelu {model_name} nie istnieje, używam domyślnego yolov8n.pt")
        model_path = os.path.join(script_dir, "yolov8n.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Nie znaleziono pliku modelu: {model_name} ani domyślnego yolov8n.pt")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"CUDA dostępne - używam GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        logger.info("CUDA niedostępne - używam CPU")

    try:
        model = YOLO(model_path)

        model.to(device)
        model.overrides['conf'] = conf_threshold
        model.overrides['iou'] = 0.5
        model.overrides['agnostic_nms'] = True
        model.overrides['max_det'] = 50

        if device.type == 'cuda':
            #model.model.half()
            dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float16).to(device)
            for _ in range(3):
                with torch.no_grad():
                    _ = model(dummy_input)

        logger.info(f"Model {model_name} załadowany pomyślnie na urządzenie: {device}")
        logger.info(f"Próg ufności (conf): {conf_threshold}")
        return model, device

    except Exception as e:
        logger.error(f"Błąd ładowania modelu: {e}")
        model = YOLO(model_path)
        device = torch.device('cpu')
        return model, device

# Funkcja do parsowania argumentów (opcjonalnie, jeśli chcesz używać tego pliku niezależnie)
def parse_args():
    parser = argparse.ArgumentParser(description='Ładowanie modelu YOLO z parametrami')
    parser.add_argument('--model_name', type=str, default="yolov8n.pt",
                        help='Nazwa pliku modelu YOLO (domyślnie: yolov8n.pt)')
    parser.add_argument('--conf_threshold', type=float, default=0.4,
                        help='Próg ufności dla detekcji (domyślnie: 0.4)')
    return parser.parse_args()

if __name__ == "__main__":
    # Przykład użycia jako niezależny skrypt
    args = parse_args()
    model, device = load_yolov8_fast(args.model_name, args.conf_threshold)