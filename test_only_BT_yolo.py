import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

def run_bytetrack_test():
    # Test için kırptığımız 20 saniyelik video
    VIDEO_DOSYASI = "test_kirpilmiş.mp4"

    # Hocanın attığı yeni YOLO modelinin yolu
    MODEL_DOSYASI = r"C:\Users\BeeLab\Desktop\yolo_test\best.pt"

    print("[SİSTEM] Yeni YOLO11 Modeli yükleniyor, ArUco kapalı, sadece ByteTrack aktif...")

    # 1. Modeli Başlat ve Doğrudan Ekran Kartına (CUDA) İt
    model = YOLO(MODEL_DOSYASI)
    model.to('cuda')

    # --- KRİTİK DÜZELTME: Modelin eğitildiği imgsz'i otomatik bul ---
    # imgsz uyuşmazlığı, küçük nesnelerde (arılar gibi) confidence skorlarının
    # frame'den frame'e dalgalanmasına ve tespitlerin kaybolmasına sebep olur.
    IMGSZ = 640  # bulunamazsa kullanılacak varsayılan
    try:
        train_args = model.ckpt.get('train_args', {}) if hasattr(model, 'ckpt') and model.ckpt else {}
        found = train_args.get('imgsz')
        if found:
            IMGSZ = found
            print(f"[SİSTEM] Modelin eğitim imgsz değeri bulundu: {IMGSZ}")
        else:
            print(f"[UYARI] Eğitim imgsz'i bulunamadı, varsayılan {IMGSZ} kullanılıyor. "
                  f"Gerekirse IMGSZ değişkenini elle ayarlayın (640/960/1280 vb.).")
    except Exception as e:
        print(f"[UYARI] imgsz otomatik tespit edilemedi ({e}). Varsayılan {IMGSZ} kullanılıyor.")

    # 2. ByteTrack'i Başlat
    # lost_track_buffer: birkaç frame tespit gelmese bile ID'yi koru (flicker azaltır)
    # track_activation_threshold: confidence eşiği ByteTrack tarafında da tutarlı olsun
    print("[SİSTEM] ByteTrack Başlatılıyor...")
    tracker = sv.ByteTrack(
        lost_track_buffer=30,
        track_activation_threshold=0.25,
        minimum_matching_threshold=0.8
    )

    # Sınıf isimleri (Hocanın modelindeki sınıflar: 0=bee_with_pollen, 1=bee_without_pollen)
    class_names = {0: "Polenli", 1: "Polensiz"}

    print("[SİSTEM] CUDA warmup yapılıyor...")
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    for _ in range(3):
        model(dummy, device="cuda", verbose=False)

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

        # 3. YOLO ile Tespit
        # imgsz artık modelin eğitildiği boyutla eşleşiyor -> stabil confidence skorları
        # conf eşiğini biraz düşürdük (0.35), ByteTrack zaten track_activation_threshold
        # ile kendi içinde filtreleme yapıyor; bu sayede sınırda kalan tespitler kaybolmuyor
        results = model(frame_bgr, conf=0.5, imgsz=IMGSZ, iou=0.3, half=False, verbose=False, device="cuda")[0]

        # 4. YOLO çıktılarını ByteTrack'in anlayacağı Supervision formatına çevir
        detections = sv.Detections.from_ultralytics(results)

        # 5. ByteTrack ile Takip Algoritmasını Güncelle
        tracked_detections = tracker.update_with_detections(detections)

        # 6. Çizim İşlemleri (Sadece tespit edilen ve takip edilen arılar için)
        if tracked_detections is not None and len(tracked_detections) > 0:
            for bbox, bt_id, class_id, conf in zip(
                tracked_detections.xyxy,
                tracked_detections.tracker_id,
                tracked_detections.class_id,
                tracked_detections.confidence
            ):
                x1, y1, x2, y2 = map(int, bbox)

                # Polen durumuna göre renk seçimi (Sarı: Polenli, Kırmızı: Polensiz)
                renk = (0, 255, 255) if class_id == 0 else (0, 0, 255)
                cls_name = class_names.get(class_id, "Ari")

                # Sadece ByteTrack ID'sini yazdırıyoruz
                etiket = f"BT:{bt_id} | {cls_name}"

                # Sınır Kutusunu (BBox) Çiz
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), renk, 2)

                # Yazı Arka Planını ve Yazıyı Çiz
                (w, h), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - 20), (x1 + w, y1), renk, -1)
                cv2.putText(frame_bgr, etiket, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Görüntüyü ekrana ver
        cv2.imshow("Saf ByteTrack Stres Testi - YOLO11 (CUDA)", frame_bgr)

        # 'q' ile çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_bytetrack_test()