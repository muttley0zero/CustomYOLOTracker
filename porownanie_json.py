import json
import motmetrics as mm

# Wczytanie ground truth i predykcji
with open("C://Users//Bartek//Downloads//licznik_osob_i_pojazdow- 03.09.2025//challenge//Hospital_000//ground_truth.json") as f:
    gt_data = json.load(f)

with open("C://Users//Bartek//Downloads//licznik_osob_i_pojazdow- 03.09.2025//predictions.json") as f:
    pred_data = json.load(f)

acc = mm.MOTAccumulator(auto_id=True)

# Iteracja po wszystkich wideo
for video_file in gt_data.keys():
    gt_video = gt_data[video_file]
    pred_video = pred_data.get(video_file, [])

    # Tworzymy słownik frame -> lista bbox + id
    frames_gt = {}
    frames_pred = {}

    for obj in gt_video:
        frame = obj['frame']
        frames_gt.setdefault(frame, []).append((obj['track_id'], obj['bbox']))

    for obj in pred_video:
        frame = obj['frame']
        frames_pred.setdefault(frame, []).append((obj['track_id'], obj['bbox']))

    # Iteracja po wszystkich klatkach wideo
    all_frames = sorted(set(list(frames_gt.keys()) + list(frames_pred.keys())))
    for frame_id in all_frames:
        gt_objs = frames_gt.get(frame_id, [])
        pred_objs = frames_pred.get(frame_id, [])

        gt_ids = [o[0] for o in gt_objs]
        gt_boxes = [o[1] for o in gt_objs]

        pred_ids = [o[0] for o in pred_objs]
        pred_boxes = [o[1] for o in pred_objs]

        if gt_boxes and pred_boxes:
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            # jeśli nie ma gt lub pred, podajemy pustą macierz
            distances = []

        acc.update(gt_ids, pred_ids, distances)

# Obliczenie metryk
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1'])
print(summary)
