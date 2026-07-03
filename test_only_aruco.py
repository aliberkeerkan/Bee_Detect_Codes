import cv2
import numpy as np

# ==========================================
# 1. MODÜL: ArUco Algılama (Laboratuvarın Özel Algoritması)
# ==========================================
class ArucoDetector:
    """Laboratuvarın kendi yazdığı Red-Channel Otsu ve 0-Hamming algoritmasını kullanan okuyucu."""
    def __init__(self):
        # Sözlüğü doğrudan hafızaya alıyoruz (TXT dosyasına bağlı kalmamak için)
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

        # Laboratuvar Kodundan Alınan Optimum Ayarlar
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
        # 1. Sadece Kırmızı Kanalı Al ve Otsu Threshold Uygula
        red = frame_bgr[:, :, 2]
        _, bw_inv = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Kontürleri Bul
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

        # 3. Aynı kutuları (Deduplicate) temizle
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

        # 4. Perspective Warp ve Bit Çözümleme
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

            # Formatı OpenCV çizim fonksiyonunun anlayacağı tipe çeviriyoruz
            if best_id is not None:
                detected_corners.append(np.array([quad]))
                detected_ids.append([best_id])

        if len(detected_ids) > 0:
            return tuple(detected_corners), np.array(detected_ids, dtype=np.int32)
        
        return None, None

# ==========================================
# ANA TEST FONKSİYONU
# ==========================================
def run_aruco_test():
    # Hızlı test için kırpılmış videoyu kullanabiliriz
    VIDEO_DOSYASI = "test_kirpilmiş.mp4" 
    print("[SİSTEM] Sadece Özel ArUco Modülü Başlatılıyor...")
    
    # Modülü örnekle
    aruco_dedektor = ArucoDetector()
    
    cap = cv2.VideoCapture(VIDEO_DOSYASI)
    if not cap.isOpened():
        print("[HATA] Video açılamadı. Dosya adını kontrol edin.")
        return

    print("[SİSTEM] Test başladı. Kapatmak için 'q' tuşuna basın.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[SİSTEM] Video sonlandı.")
            break

        # Sadece ArUco taraması yap
        corners, ids = aruco_dedektor.detect(frame_bgr)

        # Eğer etiket bulduysa OpenCV'nin kendi yeşil çerçevesiyle çizdir
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

        # Videoyu ekranda oynat
        cv2.imshow("Saf Özel ArUco Testi", frame_bgr)

        # Çıkış kontrolü
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_aruco_test()