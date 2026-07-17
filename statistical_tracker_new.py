import cv2
import numpy as np
import supervision as sv
import torch
import csv
import math
import multiprocessing as mp
from rfdetr import RFDETRSmall

# ==========================================
# YARDIMCI SINIF: Kalıcı Hafıza ve İstatistik
# ==========================================
class TrackedBee:
    def __init__(self, byte_track_id):
        self.byte_track_id = byte_track_id
        self.aruco_id = None
        self.bbox = None
        self.class_id = None
        self.confidence = 0.0

        # --- Açı ve Hareket Takibi İçin Değişkenler ---
        self.angle_deg = 0.0
        self.last_center = None

        self.total_frames_seen = 0
        self.aruco_detected_frames = 0

    def update_info(self, bbox, class_id, confidence, detected_aruco_id=None, aruco_corners=None):
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence

        # 1. Mevcut Bounding Box Merkezini Hesapla
        x1, y1, x2, y2 = bbox
        current_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        # 2. Açıyı (Yönü) Hesapla
        if aruco_corners is not None:
            # order_quad: 0:Sol-Üst, 1:Sağ-Üst, 2:Sağ-Alt, 3:Sol-Alt
            top_center_x = (aruco_corners[0][0] + aruco_corners[1][0]) / 2.0
            top_center_y = (aruco_corners[0][1] + aruco_corners[1][1]) / 2.0

            bottom_center_x = (aruco_corners[2][0] + aruco_corners[3][0]) / 2.0
            bottom_center_y = (aruco_corners[2][1] + aruco_corners[3][1]) / 2.0

            delta_x = top_center_x - bottom_center_x
            delta_y = bottom_center_y - top_center_y

            angle = math.degrees(math.atan2(delta_y, delta_x))
            if angle < 0: angle += 360
            self.angle_deg = angle

        elif self.last_center is not None:
            delta_x = current_center[0] - self.last_center[0]
            delta_y = self.last_center[1] - current_center[1]

            if math.hypot(delta_x, delta_y) > 2.0:
                angle = math.degrees(math.atan2(delta_y, delta_x))
                if angle < 0: angle += 360
                self.angle_deg = angle

        self.last_center = current_center
        self.total_frames_seen += 1

        if detected_aruco_id is not None:
            self.aruco_id = detected_aruco_id
            self.aruco_detected_frames += 1

# ==========================================
# 1. MODÜL: ArUco Algılama
# ==========================================
class ArucoDetector:
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
        # NOT: Eskiden 10'du -> 16 kenar (border) hücresinden 10'unun (%62.5) yanlış
        # olmasına izin veriyordu; bu, gerçek bir marker olmayan rastgele karanlık
        # bloklarin da border testini gecip iç 3x3 desene (MAX_HAMMING=0 sayesinde
        # ~%2 ihtimalle 10 kodtan birine denk gelerek) yanlış bir Aruco ID olarak
        # okunmasına izin veriyordu. 16 hücrenin çoğu tutarlı siyah olmalı; 2-3 hata
        # payı görüntü gürültüsü için yeterli, ama sahte adayları eler.
        self.MAX_BORDER_ERRORS = 2
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
# 2. SÜREÇ (PROCESS) İŞÇİSİ: Video Okuma + ArUco
# ==========================================
# ÖNEMLİ: Eski halinde okuma + ArUco + GPU inference + CSV yazma HEPSİ tek bir
# döngüde sırayla çalışıyordu -> aralarında hiç örtüşme yoktu -> GPU sürekli CPU'yu
# bekliyordu (bu %30 kullanımın ana sebebiydi). Şimdi okuma+ArUco ayrı bir process'te,
# GPU ana process'te -> biri bir sonraki frame'i hazırlarken diğeri öncekini işleyebiliyor.
def reader_and_aruco_worker(video_source, frame_queue: mp.Queue):
    aruco_detector = ArucoDetector()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[HATA] {video_source} açılamadı!")
        frame_queue.put(None)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_queue.put(("META", total_frames))
    print(f"[SİSTEM] Multiprocess Veri Okuyucu Başladı. Video toplam {total_frames} kare.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        aruco_corners, aruco_ids = aruco_detector.detect(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_queue.put((frame_rgb, aruco_corners, aruco_ids))

    cap.release()
    frame_queue.put(None)  # sentinel: video bitti

# ==========================================
# 3. MODÜL: GPU Destekli Deteksiyon ve Takip
# ==========================================
class BeeDetector:
    def __init__(self, model_path, fps=30):
        # Öncelik: NVIDIA (cuda) > Intel Arc (xpu) > CPU.
        # torch.xpu, PyTorch 2.5+ ile native olarak geliyor (ayrı bir Intel eklentisi
        # gerekmiyor - eski "intel_extension_for_pytorch" artık EOL). XPU'nun
        # gerçekten çalışması için: (1) torch'un xpu-destekli sürümü kurulu olmalı
        # (pip install torch --index-url https://download.pytorch.org/whl/xpu),
        # (2) güncel Intel Arc/Graphics sürücüsü kurulu olmalı.
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
        else:
            self.device = "cpu"
        print(f"[SİSTEM] Kullanılan cihaz: {self.device}")

        # Sabit girdi boyutu (512x512) için cuDNN'in en hızlı convolution
        # algoritmalarını seçip cache'lemesini sağlar; doğruluğu etkilemez.
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        try:
            self.model = RFDETRSmall(pretrain_weights=model_path, num_classes=3, device=self.device)
        except TypeError:
            self.model = RFDETRSmall(pretrain_weights=model_path, num_classes=3)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                self.model.model.to(self.device)

        self.model.optimize_for_inference()

        # ESKI HALI: sv.ByteTrack() - tamamen varsayılan ayarlar. Bu, 8 saatlik
        # gerçek videoda 5 arı için 11.496 farklı ham ID üretilmesinin
        # sebeplerinden biriydi (kısa kesintilerde/örtüşmede ID hemen değişiyordu).
        # Şimdi 3 parametreyi ayarladık:
        #   lost_track_buffer: 30 -> 90 kare. Bir arı kısaca kaybolduğunda (başka
        #     arının altına girme, ArUco okunamaması vb.) ID'sinin ne kadar süre
        #     "canlı" tutulup tekrar eşleştirileceği. 90 kare = ~3 saniye @30fps.
        #   minimum_consecutive_frames: 1 -> 3. Bir izin "gerçek" sayılması için
        #     art arda kaç karede eşleşmesi gerektiği. Tek karelik yanlış
        #     tespitlerin sahte/kısa ömürlü ID açmasını azaltır.
        #   frame_rate: videonun gerçek fps'i - lost_track_buffer'ın kaç kareye
        #     denk geldiğini doğru hesaplaması için (kütüphane içeride
        #     frame_rate/30 oranıyla ölçekliyor).
        # NOT: Bunlar deneysel başlangıç değerleri, "kesin doğru" değil - küçük
        # bir testte (20sn/1sa) eski/yeni BT_ID sayısını karşılaştırıp
        # gerekirse ayarlamak gerekebilir.
        self.tracker = sv.ByteTrack(
            lost_track_buffer=90,
            minimum_consecutive_frames=3,
            frame_rate=fps,
        )

    def detect_and_track(self, frame_rgb):
        # inference_mode: autograd defteri tutulmaz (eğitim yapmıyoruz), bellek ve
        # hafif hız kazandırır. FP16 DEĞİL, sayısal hassasiyeti etkilemez.
        with torch.inference_mode():
            try:
                rf_out = self.model.predict(frame_rgb, threshold=0.50, shape=(512, 512), device=self.device)
            except TypeError:
                rf_out = self.model.predict(frame_rgb, threshold=0.50, shape=(512, 512))

        if len(rf_out.xyxy) == 0: return None

        detections = sv.Detections(xyxy=rf_out.xyxy, confidence=rf_out.confidence, class_id=rf_out.class_id)
        return self.tracker.update_with_detections(detections)

# ==========================================
# 4. MODÜL: Headless (Ekransız) Veri Toplayıcı
# ==========================================
class VideoFeed:
    def __init__(self, video_source, model_path, fps=30):
        self.video_source = video_source
        self.bee_detector = BeeDetector(model_path, fps=fps)
        self.active_bees = {}
        self.frame_queue = mp.Queue(maxsize=64)
        self.total_frames = 0

    def check_aruco_in_bbox(self, aruco_corners, aruco_ids, bbox):
        # ESKI HALI: bbox icine düsen ilk aruco'yu döndürüyordu. Bu, cv2.findContours'un
        # kontur bulma sırası (görüntüdeki rastgele detaylara bağlı, mesafeyle ilgisi yok)
        # yüzünden, birbirine yakın/çakışan iki arı olduğunda YANLIŞ arının etiketini
        # diğerine atayabiliyordu - aynı fiziksel sahne, sırf kontur keşif sırası
        # değiştiği için farklı (ve yanlış) sonuç verebiliyordu. Şimdi bbox içine düşen
        # TÜM adaylar arasından bbox merkezine EN YAKIN olanı seçiyoruz.
        if aruco_corners is None or aruco_ids is None or len(aruco_ids) == 0: return None, None
        x1, y1, x2, y2 = bbox
        bbox_center_x, bbox_center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        flat_ids = aruco_ids.flatten()
        best_id, best_corner, best_dist = None, None, None
        for i in range(len(flat_ids)):
            corner = aruco_corners[i][0]
            center_x, center_y = np.mean(corner[:, 0]), np.mean(corner[:, 1])
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                dist = math.hypot(center_x - bbox_center_x, center_y - bbox_center_y)
                if best_dist is None or dist < best_dist:
                    best_id, best_corner, best_dist = int(flat_ids[i]), corner, dist
        return best_id, best_corner

    def run(self):
        reader_process = mp.Process(
            target=reader_and_aruco_worker,
            args=(self.video_source, self.frame_queue),
            daemon=True
        )
        reader_process.start()

        print("[SİSTEM] Hızlı Analiz (Headless Mode) Başladı.")
        print("[SİSTEM] Konum (X,Y) ve Yön (Açı) verileri 'bee_trajectories.csv' dosyasına kaydediliyor...")

        frame_count = 0

        with open('bee_trajectories.csv', mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Frame", "BT_ID", "Aruco_ID", "Center_X", "Center_Y", "Angle_Deg"])

            while True:
                # BLOKLAYAN get() -> spin-loop yok, veri gelene kadar bekler
                item = self.frame_queue.get()

                if item is None:
                    break

                if isinstance(item, tuple) and len(item) == 2 and item[0] == "META":
                    self.total_frames = item[1]
                    print(f"[SİSTEM] Video toplam {self.total_frames} kare. Lütfen bekleyin...")
                    continue

                frame_rgb, aruco_corners, aruco_ids = item
                frame_count += 1

                if frame_count % 500 == 0:
                    if self.total_frames > 0:
                        print(f"> İşlenen Kare: {frame_count} / {self.total_frames} "
                              f"(%{(frame_count / self.total_frames) * 100:.1f})")
                    else:
                        print(f"> İşlenen Kare: {frame_count}")
                    # 8 saatlik uzun koşuda kesinti olursa veri kaybını azaltır
                    csv_file.flush()

                tracked_detections = self.bee_detector.detect_and_track(frame_rgb)

                if tracked_detections is not None:
                    for bbox, bt_id, class_id, conf in zip(
                        tracked_detections.xyxy,
                        tracked_detections.tracker_id,
                        tracked_detections.class_id,
                        tracked_detections.confidence
                    ):
                        if bt_id not in self.active_bees:
                            self.active_bees[bt_id] = TrackedBee(bt_id)

                        bee = self.active_bees[bt_id]

                        detected_aruco_id, matched_corner = self.check_aruco_in_bbox(aruco_corners, aruco_ids, bbox)

                        bee.update_info(bbox, class_id, conf, detected_aruco_id, matched_corner)

                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0

                        final_aruco_id = detected_aruco_id if detected_aruco_id is not None else -1

                        csv_writer.writerow([frame_count, bt_id, final_aruco_id, f"{center_x:.1f}", f"{center_y:.1f}", f"{bee.angle_deg:.1f}"])

        reader_process.join()
        self.print_final_statistics()

    def print_final_statistics(self):
        print("\n" + "="*80)
        print("🐝 VİDEO ANALİZİ TAMAMLANDI - BİREYSEL ARI TAKİP (TRACKING) İSTATİSTİKLERİ 🐝")
        print("="*80)
        print(f"{'Arı (BT ID)':<12} | {'Aruco ID':<10} | {'Toplam Kare':<12} | {'ArUco Ağırlığı':<18} | {'Sadece BT Ağırlığı':<18}")
        print("-" * 80)

        for bt_id, bee in self.active_bees.items():
            if bee.total_frames_seen > 5:
                aruco_pct = (bee.aruco_detected_frames / bee.total_frames_seen) * 100
                bt_pct = 100 - aruco_pct

                a_id = bee.aruco_id if bee.aruco_id is not None else "Bulunamadı"
                print(f"ID: {bt_id:<8} | {str(a_id):<10} | {bee.total_frames_seen:<12} | % {aruco_pct:.1f}{'':<14} | % {bt_pct:.1f}")

        print("="*80 + "\n")

if __name__ == "__main__":
    # Windows'ta multiprocessing (spawn) için gerekli
    mp.freeze_support()

    VIDEO_DOSYASI = "test_one_minute.mp4"  # <-- KONTROL ET: gerçek dosya adın buysa dokunma, farklıysa (örn. tire/alt çizgi) burayı düzelt
    MODEL_DOSYASI = r"C:\Users\erkan\Desktop\Bee_Detect_Codes\checkpoint_best_total.pth"
    VIDEO_FPS = 30  # videonun gerçek fps'i - ByteTrack'in lost_track_buffer hesabı için önemli

    sistem = VideoFeed(video_source=VIDEO_DOSYASI, model_path=MODEL_DOSYASI, fps=VIDEO_FPS)
    sistem.run()