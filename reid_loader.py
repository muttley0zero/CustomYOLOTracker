import torch
from torchreid import models  # import z deep-person-reid
import logging
import torchvision.transforms as T
import numpy as np
import os

# Transformacja dla ReID
reid_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

logger = logging.getLogger(__name__)

def load_reid_model(model_path=None, device='cuda'):
    """Ładuje model ReID (OSNet)."""
    device_str = device if isinstance(device, str) else ('cuda' if device.type=='cuda' else 'cpu')
    
    model = models.build_model(
        name='osnet_ain_x0_75',
        num_classes=751,
        loss='softmax',
        pretrained=True
    )

    if model_path and os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device_str)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Wczytano checkpoint ReID z: {model_path}")
    else:
        logger.warning(f"Nie znaleziono checkpointu '{model_path}', używam pretrained=True")

    model.to(device_str)
    model.eval()
    logger.info(f"ReID model gotowy na urządzeniu: {device_str}")
    return model

def get_embedding(reid_model, device, frame, bbox):
    """Generuje embedding dla wykrytego obiektu z klatki."""
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    tensor = reid_transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        features = reid_model(tensor)
    return features.squeeze().cpu().numpy()