import cv2
import torch
import supervision as sv
from rfdetr import RFDETRSmall

def run_bytetrack_test():
    VIDEO_DOSYASI = "test_cropped.mp4"
    MODEL_DOSYASI = r"C:\Users\BeeLab\Desktop\yolo_test\checkpoint_best_total.pth"

    # CUDA kontrolü
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[SİSTEM] Modeller yükleniyor. Kullanılan cihaz: {device.upper()}")
    
    # 1. Modeli Başlat
    # Model cihaz yönetimini kendi içinde yapar, .to() komutuna ihtiyaç duymaz
    model = RFDETRSmall(pretrain_weights=MODEL_DOSYASI, num_classes=3)
    model.optimize_for_inference()
    
    # 2. ByteTrack'i Başlat
    tracker = sv.ByteTrack()
    
    # Sınıf isimleri
    class_names = {1: "Polenli", 2: "Polensiz"}

    cap = cv2.VideoCapture(VIDEO_DOSYASI)
    if not cap.isOpened():
        print("[HATA] Video açılamadı.")
        return

    print("[SİSTEM] Test başladı. Kapatmak için 'q' tuşuna basın.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 3. RF-DETR ile Tespit
        # Eğer CUDA aktifse, model kütüphanesi otomatik olarak GPU üzerinden çalışacaktır
        rf_out = model.predict(frame_rgb, threshold=0.50, shape=(512, 512))

        # Ekranda arı varsa takibe al
        if len(rf_out.xyxy) > 0:
            detections = sv.Detections(
                xyxy=rf_out.xyxy,
                confidence=rf_out.confidence,
                class_id=rf_out.class_id
            )
            
            # 4. ByteTrack ile Güncelle
            tracked_detections = tracker.update_with_detections(detections)
            
            # 5. Çizim İşlemleri
            for bbox, bt_id, class_id, conf in zip(
                tracked_detections.xyxy, 
                tracked_detections.tracker_id, 
                tracked_detections.class_id, 
                tracked_detections.confidence
            ):
                x1, y1, x2, y2 = map(int, bbox)
                
                renk = (0, 255, 255) if class_id == 1 else (0, 0, 255)
                cls_name = class_names.get(class_id, "Ari")
                etiket = f"BT:{bt_id} | {cls_name}"
                
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), renk, 2)
                (w, h), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - 20), (x1 + w, y1), renk, -1)
                cv2.putText(frame_bgr, etiket, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("DETR + ByteTrack (CUDA Destekli)", frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_bytetrack_test()
