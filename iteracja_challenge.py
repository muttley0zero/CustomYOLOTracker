import os
import json
import cv2
import numpy as np
import torch
from datetime import datetime

from filtr_Kalmana_i_ReID import Track

# ================== KONFIG ==================
video_folder = r"C:\Users\Bartek\Downloads\licznik_osob_i_pojazdow- 05.09.2025\challenge\Hospital_000\videos"
calibration_file = r"C:\Users\Bartek\Downloads\licznik_osob_i_pojazdow- 05.09.2025\challenge\Hospital_000\calibration.json"
output_file = r"C:\Users\Bartek\Downloads\licznik_osob_i_pojazdow- 05.09.2025\track1.txt"

scene_id = 0               # ID sceny (dla Hospital_000 zostaw 0)
default_dims = (0.6, 0.4, 1.7)  # width, length, height (m)
default_yaw = 0.0
class_id_default = 0       # 0 = Person

# ================== LOGGER ==================
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ================== KAMERA ==================
def list_video_files(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
    files.sort()
    return files

def load_calibration(path):
    if not os.path.exists(path):
        print(f"[WARN] Brak pliku kalibracji: {path}")
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except:
            return {}

def get_camera_params(calib, camera_name):
    if camera_name not in calib:
        return None
    try:
        K = np.array(calib[camera_name]["intrinsic"], dtype=np.float32)
        dist = np.array(calib[camera_name]["distortion"], dtype=np.float32)
        extr = np.array(calib[camera_name]["extrinsic"], dtype=np.float32)
        return K, dist, extr
    except:
        return None

def camera_name_from_file(video_filename):
    return os.path.splitext(os.path.basename(video_filename))[0]

def bbox_to_xyxy(b):
    if b is None or len(b) < 4:
        return None
    b = np.array(b, dtype=float).flatten()
    if b.size != 4:
        return None
    if b[2] > 0 and b[3] > 0 and b[2] < 1000 and b[3] < 1000:
        x1, y1, w, h = b
        x2, y2 = x1 + w, y1 + h
    else:
        x1, y1, x2, y2 = b
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def backproject_to_world(u, v, K, dist, extrinsic):
    try:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
        pts_undist = cv2.undistortPoints(pts, K, dist, P=K)
        x, y = pts_undist[0, 0]
        cam_dir = np.array([x, y, 1.0, 0.0], dtype=np.float32)
        E = extrinsic.astype(np.float32)
        R = E[:3, :3]
        t = E[:3, 3]
        cam_pos_world = -R.T @ t
        ray_dir_world = R.T @ cam_dir[:3]
        if abs(ray_dir_world[2]) < 1e-6:
            return 0.0, 0.0, 0.0
        s = -cam_pos_world[2] / ray_dir_world[2]
        world = cam_pos_world + s * ray_dir_world
        return float(world[0]), float(world[1]), float(world[2])
    except:
        return 0.0, 0.0, 0.0

# ================== TRACKER 3D ==================
class Track3D(Track):
    def __init__(self, detection, frame_number, track_id, embedding=None, class_id=None, bbox=None,
                 K=None, dist=None, extrinsic=None, default_dims=(0.6,0.4,1.7)):
        super().__init__(detection, frame_number, track_id, embedding, class_id, bbox)
        self.K = K
        self.dist = dist
        self.extrinsic = extrinsic
        self.default_dims = default_dims
        self.world_positions = []
        self.update_world_position()

    def update_world_position(self):
        if self.K is None or self.extrinsic is None:
            wx, wy, wz = 0.0, 0.0, 0.0
        else:
            x1, y1, x2, y2 = self.bboxes[-1] if self.bboxes else self._position
            cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
            wx, wy, wz = backproject_to_world(cx, cy, self.K, self.dist, self.extrinsic)
        self.world_positions.append((wx, wy, wz))

class FastObjectTracker3D:
    def __init__(self, max_disappeared=50):
        self.tracks = {}
        self.all_time_track_count = 0
        self.max_disappeared = max_disappeared

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections, embeddings=None, class_ids=None, bboxes=None, frame_number=0,
               K=None, dist=None, extrinsic=None):
        if embeddings is None: embeddings = [None]*len(detections)
        if class_ids is None: class_ids = [None]*len(detections)
        if bboxes is None: bboxes = [None]*len(detections)
        flattened_embeddings = [emb.flatten() if emb is not None else None for emb in embeddings]
        matched_tracks = set()
        updated_tracks = {}

        for i, (det, emb, cls_id, bbox) in enumerate(zip(detections, flattened_embeddings, class_ids, bboxes)):
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                if track.disappeared > self.max_disappeared: continue
                if track.class_id != cls_id: continue
                iou = self._iou(det[:4], track.position)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id is not None:
                track = self.tracks[best_track_id]
                track.update(det, embedding=emb, class_id=cls_id, bbox=bbox)
                track.update_world_position()
                updated_tracks[best_track_id] = track
                matched_tracks.add(best_track_id)

        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.disappeared += 1
                if track.disappeared <= self.max_disappeared:
                    updated_tracks[track_id] = track

        for i, (det, emb, cls_id, bbox) in enumerate(zip(detections, flattened_embeddings, class_ids, bboxes)):
            already_matched = False
            for track_id in matched_tracks:
                if np.allclose(det[:2], updated_tracks[track_id].positions[-1], atol=10.0):
                    already_matched = True
                    break
            if already_matched: continue

            new_track = Track3D(det, frame_number, self.all_time_track_count, embedding=emb,
                                class_id=cls_id, bbox=bbox, K=K, dist=dist, extrinsic=extrinsic,
                                default_dims=default_dims)
            updated_tracks[self.all_time_track_count] = new_track
            self.all_time_track_count += 1

        to_remove = [tid for tid, t in updated_tracks.items() if t.disappeared > self.max_disappeared]
        for tid in to_remove: del updated_tracks[tid]

        self.tracks = updated_tracks
        return self.tracks

# ================== PROCESS FRAME ==================
def process_frame_fast(model, frame, tracker, frame_count, device, conf_thresh=0.4, K=None, dist=None, extrinsic=None):
    results = {'people_count':0,'vehicle_count':0,'people_detections':[],'vehicle_detections':[],'tracks':{}}
    try:
        h0, w0 = frame.shape[:2]
        inp = cv2.resize(frame,(640,640))
        inp = inp[:,:,::-1].transpose(2,0,1)
        inp = np.ascontiguousarray(inp)
        inp_tensor = torch.from_numpy(inp).to(device)
        inp_tensor = inp_tensor.half() if device.type=='cuda' else inp_tensor.float()
        inp_tensor /= 255.0
        if inp_tensor.ndimension()==3: inp_tensor = inp_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(inp_tensor, verbose=False)[0]

        bboxes, class_ids = [], []
        if pred is not None and pred.boxes is not None:
            boxes = pred.boxes.cpu().numpy()
            scale_x, scale_y = w0/640.0, h0/640.0
            for box in boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf<conf_thresh: continue
                x1,x2 = int(x1*scale_x), int(x2*scale_x)
                y1,y2 = int(y1*scale_y), int(y2*scale_y)
                cx,cy = (x1+x2)//2,(y1+y2)//2
                det_data = (cx,cy,cls,conf,(x1,y1,x2,y2))
                if cls==0: results['people_detections'].append(det_data)
                elif cls in [2,3,5,7]: results['vehicle_detections'].append(det_data)
                bboxes.append([x1,y1,x2,y2])
                class_ids.append(cls)

        tracker.update(np.array(bboxes,dtype=np.float32) if bboxes else np.empty((0,4),dtype=np.float32),
                       embeddings=None,class_ids=class_ids,bboxes=bboxes,frame_number=frame_count,
                       K=K,dist=dist,extrinsic=extrinsic)

        results['people_count'] = len(results['people_detections'])
        results['vehicle_count'] = len(results['vehicle_detections'])
        results['tracks'] = tracker.tracks

    except Exception as e:
        logger.error(f"process_frame_fast error: {e}", exc_info=True)

    return results

# ================== MAIN ==================
def main():
    video_files = list_video_files(video_folder)
    if not video_files: raise FileNotFoundError(f"Brak wideo w {video_folder}")
    calib = load_calibration(calibration_file)

    # Model YOLOv8
    from ladowanie_yolo import load_yolov8_fast
    model, device = load_yolov8_fast()
    print(f"[INFO] YOLOv8 loaded on {device}")

    cv2.namedWindow("Challenge Tracker", cv2.WINDOW_NORMAL)
    total_written = 0
    object_id_map = {}
    scene_id_local = scene_id

    with open(output_file,"w",encoding="utf-8") as fout:
        global_object_id = 0
        for cam_idx, video_file in enumerate(video_files):
            video_path = os.path.join(video_folder, video_file)
            cam_key = camera_name_from_file(video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): continue
            params = get_camera_params(calib, cam_key)
            if params is not None:
                K, dist, extr = params
            else:
                K = dist = extr = None

            tracker = FastObjectTracker3D()
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                results = process_frame_fast(model, frame, tracker, frame_id, device, K=K, dist=dist, extrinsic=extr)

                for track_id, tr in tracker.tracks.items():
                    cls = tr.class_id if tr.class_id is not None else class_id_default
                    width,length,height = default_dims
                    yaw = default_yaw
                    wx, wy, wz = tr.world_positions[-1] if tr.world_positions else (0.0,0.0,0.0)
                    id_key = (cam_idx, track_id)
                    if id_key not in object_id_map:
                        object_id_map[id_key] = global_object_id
                        global_object_id += 1
                    obj_id = object_id_map[id_key]
                    line = f"{scene_id_local} {int(cls)} {obj_id} {frame_id} {wx:.3f} {wy:.3f} {wz:.3f} {width:.3f} {length:.3f} {height:.3f} {yaw:.3f}\n"
                    fout.write(line)
                    fout.flush()
                    total_written += 1

                    # Rysowanie
                    x1,y1,x2,y2 = bbox_to_xyxy(tr.bboxes[-1])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"ID {obj_id}",(x1,max(15,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                cv2.imshow("Challenge Tracker", frame)
                frame_id += 1
                if cv2.waitKey(1) & 0xFF == 27: break

            cap.release()
            print(f"[INFO] Zakończono {video_file}, zapisano linii: {total_written}")

    cv2.destroyAllWindows()
    print(f"\n✅ Zakończono. Zapisano łącznie linii: {total_written}")
    print(f"📄 Plik wynikowy: {output_file}")

if __name__=="__main__":
    main()
