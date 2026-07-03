import cv2
import supervision as sv
from rfdetr import RFDETRSmall

# 1. Modeli Yükle
print("Model yükleniyor, lütfen bekleyin...")
checkpoint_path = r"C:\Users\BeeLab\Desktop\yolo_test\checkpoint_best_total.pth"
bee_detector = RFDETRSmall(pretrain_weights=checkpoint_path, num_classes=3)
bee_detector.optimize_for_inference()

# 2. ByteTrack Algoritmasını Başlat (Hafıza)
tracker = sv.ByteTrack()

# Sınıf ID'lerini isme çevirmek için sözlük (RF-DETR çıktılarına göre)
# 1: bee_with_pollen, 2: bee_without_pollen
CLASS_NAMES_DICT = {
    1: "Polenli",
    2: "Polensiz"
}

# 3. Videoyu Aç
kaynak_dosya = "test_cropped.mp4" 
cap = cv2.VideoCapture(kaynak_dosya)

if not cap.isOpened():
    print("HATA: Video bulunamadı veya açılamadı!")
    exit()

print("Video analiz ediliyor ve arılar takip ediliyor... (Çıkmak için 'q')")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("Video tamamlandı.")
        break
        
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 4. Modeli Çalıştır (Eşik %50)
    rf_out = bee_detector.predict(frame_rgb, threshold=0.50, shape=(512, 512))

    # Ekranda arı varsa takip işlemini başlat
    if len(rf_out.xyxy) > 0:
        # RF-DETR çıktılarını ByteTrack'in anlayacağı formata çevir
        detections = sv.Detections(
            xyxy=rf_out.xyxy,
            confidence=rf_out.confidence,
            class_id=rf_out.class_id
        )
        
        # 5. Algoritmayı Güncelle (ID ataması ve eşleştirme burada yapılıyor)
        tracked_detections = tracker.update_with_detections(detections)
        
        # 6. Ekrana Çizim Yap
        for bbox, tracker_id, class_id, conf in zip(
            tracked_detections.xyxy, 
            tracked_detections.tracker_id, 
            tracked_detections.class_id, 
            tracked_detections.confidence
        ):
            # Koordinatları tam sayı yap
            x1, y1, x2, y2 = map(int, bbox)
            
            # Sınıf adını ve rengini belirle
            cls_name = CLASS_NAMES_DICT.get(class_id, "Ari")
            if class_id == 1:
                renk = (0, 255, 255) # Sarı (Polenli)
            else:
                renk = (0, 0, 255) # Kırmızı (Polensiz)
                
            # ID Numarasını ve bilgileri etikete yaz
            etiket = f"ID:{tracker_id} {cls_name} %{conf*100:.0f}"
            
            # Kutu ve yazıyı çiz
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), renk, 2)
            (w, h), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - 20), (x1 + w, y1), renk, -1)
            cv2.putText(frame_bgr, etiket, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 7. Videoyu Ekranda Oynat
    cv2.imshow("ByteTrack + RF-DETR Ari Takibi", frame_bgr)

    # Kapatmak için klavyeden 'q' tuşunu dinle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
