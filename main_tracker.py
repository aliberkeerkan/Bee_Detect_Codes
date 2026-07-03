import cv2
import numpy as np
import supervision as sv
import torch  # CUDA hızlandırması için PyTorch eklendi
from rfdetr import RFDETRSmall

# ==========================================
# YARDIMCI SINIF: Kalıcı Hafıza (Persistent Tracking)
# ==========================================
class TrackedBee:
    """Her bir arının bilgilerini tutan ve ArUco ID'sini kalıcı kılan sınıf."""
    def __init__(self, byte_track_id):
        self.byte_track_id = byte_track_id
        self.aruco_id = None  # Başlangıçta ArUco etiketi bilinmiyor
        self.bbox = None
        self.class_id = None
        self.confidence = 0.0

    def update_info(self, bbox, class_id, confidence, detected_aruco_id=None):
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        
        # Eğer bu frame'de ArUco okunduysa, hafızaya kazı (ID Correction)
        if detected_aruco_id is not None:
            self.aruco_id = detected_aruco_id

    def get_display_id(self):
        # Ekrana yazdırırken öncelik her zaman ArUco ID'nindir (Çünkü kesin bilgidir)
        if self.aruco_id is not None:
            return f"Aruco:{self.aruco_id}"
        return f"BT:{self.byte_track_id}"


# ==========================================
# 1. MODÜL: ArUco Algılama (Laboratuvarın Özel Algoritması)
# ==========================================
class ArucoDetector:
    """Laboratuvarın kendi yazdığı Red-Channel Otsu ve 0-Hamming algoritmasını kullanan okuyucu."""
    def __init__(self):
        self.custom_markers = {
            0: np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.uint8),
            1: np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=np.uint8),
            2: np.array([[0,0,0],[0,1,0],[1,1,1]], dtype=np.uint8),
            3: np.array([[0,0,0],[1,1,1],[0,1,1]], dtype=np.uint8),
            4: np.array([[0,0,1],[1,0,0],[1,0,1]], dtype=np.uint8),
            5: np.array([[0,0,1],[1,0,1],[1,1,0]], dtype=np.uint8),
            6: np.array([[0,1,0],[0,1,1],[1,0,1]], dtype=np.uint8),
            7: np.array([[0,1,0],[1,0,1],[1,1,1]], dtype=np.uint8),
            8: np.array([[0,1,1],[0,0,0],[1,1,1]], dtype=np.uint8),
            9: np.array([[1,0,1],[1,1,1],[1,1,1]], dtype=np.uint8),
        }

        self.MARKER_SIZE = 3
        self.BORDER_BITS = 1
        self.TOTAL_CELLS = self.MARKER_SIZE + 2 * self.BORDER_BITS
        self.CELL_PIXELS = 24
        self.MAX_HAMMING = 0
        self.MAX_BORDER_ERRORS = 10
        self.MIN_AREA = 80
        self.MIN_SOLIDITY = 0.6
        self.POLY_EPS_RATIO = 0.05
        self.MAX_CONTOUR_AREA = 500
        self.QUAD_EXPAND_SCALE = 1.18
        self.SAMPLE_INSET_RATIO = 0.20

    def order_quad(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        return np.array([
            pts[np.argmin(s)], pts[np.argmin(d)], 
            pts[np.argmax(s)], pts[np.argmax(d)]
        ], dtype=np.float32)

    def detect(self, frame_bgr):
        red = frame_bgr[:, :, 2]
        _, bw_inv = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(bw_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quads = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA or area > self.MAX_CONTOUR_AREA: continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0: continue

            approx = cv2.approxPolyDP(cnt, self.POLY_EPS_RATIO * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx): continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0 or (area / hull_area) < self.MIN_SOLIDITY: continue

            quad = approx.reshape(4, 2).astype(np.float32)
            ordered = self.order_quad(quad)

            side_lengths = [
                np.linalg.norm(ordered[0] - ordered[1]), np.linalg.norm(ordered[1] - ordered[2]),
                np.linalg.norm(ordered[2] - ordered[3]), np.linalg.norm(ordered[3] - ordered[0]),
            ]
            min_side = min(side_lengths)
            if min_side < 6 or max(side_lengths) / max(min_side, 1e-6) > 1.8: continue

            quads.append(ordered)

        kept_quads = []
        if quads:
            items = [(q, q.mean(axis=0), abs(cv2.contourArea(q))) for q in quads]
            items.sort(key=lambda x: -x[2])
            for q, center, area in items:
                if not any(np.linalg.norm(center - kc) < 12.0 for _, kc, _ in kept_quads):
                    kept_quads.append((q, center, area))
        quads = [x[0] for x in kept_quads]

        detected_corners = []
        detected_ids = []

        for quad in quads:
            center = quad.mean(axis=0)
            expanded = (quad - center) * self.QUAD_EXPAND_SCALE + center
            src = self.order_quad(expanded)
            
            side = self.TOTAL_CELLS * self.CELL_PIXELS
            dst = np.array([[0,0], [side-1,0], [side-1,side-1], [0,side-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(red, M, (side, side), flags=cv2.INTER_LINEAR)

            _, bw = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h, w = bw.shape
            cell_w, cell_h = w / self.TOTAL_CELLS, h / self.TOTAL_CELLS
            bits = np.zeros((self.TOTAL_CELLS, self.TOTAL_CELLS), dtype=np.uint8)

            for r in range(self.TOTAL_CELLS):
                for c in range(self.TOTAL_CELLS):
                    x0 = int((c + self.SAMPLE_INSET_RATIO) * cell_w)
                    x1 = int((c + 1 - self.SAMPLE_INSET_RATIO) * cell_w)
                    y0 = int((r + self.SAMPLE_INSET_RATIO) * cell_h)
                    y1 = int((r + 1 - self.SAMPLE_INSET_RATIO) * cell_h)
                    patch = bw[y0:y1, x0:x1]
                    if patch.size > 0 and (patch.mean() / 255.0) >= 0.5:
                        bits[r, c] = 1

            border_errors = np.count_nonzero(np.concatenate([
                bits[0, :], bits[-1, :], bits[1:-1, 0], bits[1:-1, -1]
            ]) == 1)

            inner = bits[1:-1, 1:-1]
            best_id = None

            for marker_id, pattern in self.custom_markers.items():
                for rot_cw in (0, 90, 180, 270):
                    k = (-rot_cw // 90) % 4
                    expected = np.rot90(pattern, k=k)
                    hamming = np.count_nonzero(inner != expected)

                    if hamming <= self.MAX_HAMMING and border_errors <= self.MAX_BORDER_ERRORS:
                        best_id = marker_id
                        break
                if best_id is not None: break

            if best_id is not None:
                detected_corners.append(np.array([quad]))
                detected_ids.append([best_id])

        if len(detected_ids) > 0:
            return tuple(detected_corners), np.array(detected_ids, dtype=np.int32)
        
        return None, None

# ==========================================
# 2. MODÜL: Arı Tespiti ve Takibi (DETR + ByteTrack GPU Hızlandırmalı)
# ==========================================
class BeeDetector:
    """Yapay Zeka (RF-DETR) ve ByteTrack algoritmasını yöneten GPU destekli modül."""
    def __init__(self, model_path):
        print("[SİSTEM] Yapay Zeka Modeli Yükleniyor...")
        
        # --- CUDA (GPU) KONTROLÜ VE AKTİVASYONU ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DONANIM] İşlem Birimi: {self.device.upper()} aktif edildi.")
        
        # Modeli başlat (Eğer kütüphane device argümanını destekliyorsa geçiriyoruz, desteklemiyorsa PyTorch üzerinden to() ile taşıyoruz)
        try:
            self.model = RFDETRSmall(pretrain_weights=model_path, num_classes=3, device=self.device)
        except TypeError:
            self.model = RFDETRSmall(pretrain_weights=model_path, num_classes=3)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                self.model.model.to(self.device)
                
        self.model.optimize_for_inference()
        
        print("[SİSTEM] ByteTrack Başlatılıyor...")
        self.tracker = sv.ByteTrack()

    def detect_and_track(self, frame_rgb):
        # Çıkarım işlemini GPU üzerinde zorlamak için (kütüphane destekliyorsa device parametresi gönderilir)
        try:
            rf_out = self.model.predict(frame_rgb, threshold=0.50, shape=(512, 512), device=self.device)
        except TypeError:
            rf_out = self.model.predict(frame_rgb, threshold=0.50, shape=(512, 512))
        
        if len(rf_out.xyxy) == 0:
            return None
            
        detections = sv.Detections(
            xyxy=rf_out.xyxy,
            confidence=rf_out.confidence,
            class_id=rf_out.class_id
        )
        
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections


# ==========================================
# 3. MODÜL: Video Feed ve Ana Kontrolcü (Controller)
# ==========================================
class VideoFeed:
    """Görüntü akışını, modülleri ve çizim işlemlerini senkronize eden ana modül."""
    def __init__(self, video_source, model_path):
        self.video_source = video_source
        self.bee_detector = BeeDetector(model_path)
        self.aruco_detector = ArucoDetector()
        
        self.active_bees = {}
        self.class_names = {1: "Polenli", 2: "Polensiz"}

    def check_aruco_in_bbox(self, aruco_corners, aruco_ids, bbox):
        if aruco_corners is None or aruco_ids is None or len(aruco_ids) == 0:
            return None
            
        x1, y1, x2, y2 = bbox
        flat_ids = aruco_ids.flatten()
        
        for i in range(len(flat_ids)):
            corner = aruco_corners[i][0]
            center_x = int(np.mean(corner[:, 0]))
            center_y = int(np.mean(corner[:, 1]))
            
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                return int(flat_ids[i])
                
        return None

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("[HATA] Video kaynağı açılamadı!")
            return

        print("[SİSTEM] Analiz Başladı. Çıkmak için 'q' tuşuna basın.")
        
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[SİSTEM] Video akışı tamamlandı.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            aruco_corners, aruco_ids = self.aruco_detector.detect(frame_bgr)
            tracked_detections = self.bee_detector.detect_and_track(frame_rgb)

            if tracked_detections is not None:
                current_frame_bt_ids = []
                
                for bbox, bt_id, class_id, conf in zip(
                    tracked_detections.xyxy, 
                    tracked_detections.tracker_id, 
                    tracked_detections.class_id, 
                    tracked_detections.confidence
                ):
                    current_frame_bt_ids.append(bt_id)
                    
                    if bt_id not in self.active_bees:
                        self.active_bees[bt_id] = TrackedBee(bt_id)
                        
                    bee = self.active_bees[bt_id]
                    detected_aruco_id = self.check_aruco_in_bbox(aruco_corners, aruco_ids, bbox)
                    bee.update_info(bbox, class_id, conf, detected_aruco_id)

                    x1, y1, x2, y2 = map(int, bee.bbox)
                    renk = (0, 255, 255) if bee.class_id == 1 else (0, 0, 255)
                    
                    display_id = bee.get_display_id()
                    cls_name = self.class_names.get(bee.class_id, "Ari")
                    etiket = f"{display_id} | {cls_name}"
                    
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), renk, 2)
                    (w, h), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame_bgr, (x1, y1 - 20), (x1 + w, y1), renk, -1)
                    cv2.putText(frame_bgr, etiket, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if aruco_ids is not None:
                cv2.aruco.drawDetectedMarkers(frame_bgr, aruco_corners, aruco_ids)

            cv2.imshow("BeeTracker PRO - Fusion System GPU", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_DOSYASI = "test_cropped.mp4"
    MODEL_DOSYASI = r"C:\Users\BeeLab\Desktop\yolo_test\checkpoint_best_total.pth"
    
    sistem = VideoFeed(video_source=VIDEO_DOSYASI, model_path=MODEL_DOSYASI)
    sistem.run()
