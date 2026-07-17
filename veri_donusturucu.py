import pandas as pd
import numpy as np
import os
import cv2

# --- AYARLAR ---
CSV_FILE = "bee_trajectories.csv"
VIDEO_FILE = "test_one_minute.mp4"  # <-- statistical_tracker_new.py'deki VIDEO_DOSYASI ile BİREBİR AYNI olmalı
EXP_NAME = "Test20sn_Exp"        # 8 saatlik gerçek koşunun (AI_Tracker_Exp) üzerine yazmaması için farklı isim
OUTPUT_DIR = "Test20sn_Exp_output"

# Videonun gerçek süresi (saat). fps dosyadan/video metadata'sından okunamazsa
# ya da şüpheliyse, bu süre ile CSV'deki son frame numarasından fps tahmini
# çapraz kontrol için hesaplanır ve konsola yazdırılır.
KNOWN_VIDEO_DURATION_HOURS = 20 / 3600  # 20 saniyelik test videosu

# --- BİLİNEN ARI SAYISI / ID FİLTRESİ ---
# Videoda gerçekte kaç arı olduğunu buraya yaz. CSV'deki Aruco_ID sütununda
# bundan fazla farklı değer görülürse (hamming-tolerant decode yanlış okumaları
# yüzünden), en az sık görülenler otomatik olarak gürültü sayılıp atılır.
# EN GÜVENİLİR YÖNTEM: Videoya taktığın gerçek 5 aruco ID numarasını burada
# elle belirt (örn. [2, 3, 5, 6, 8]). None bırakırsan otomatik seçim yapılır
# ama bu tahmine dayalıdır - kontrol etmen önerilir.
KNOWN_BEE_ARUCO_IDS = None  # örn: [2, 3, 5, 6, 8]
EXPECTED_BEE_COUNT = 5

# Aruco etiketi olmayan (Isimsiz_Arı) takip parçaları varsayılan olarak tamamen
# atılır çünkü bu çalışmada sadece etiketli 5 arı önemli; kovandaki diğer
# arılardan gelen kısa/gürültülü ByteTrack ID'leri onlarca-yüzlerce sahte
# "arı" kolonuna dönüşüp analiz süresini/hafızasını katlayarak büyütüyordu.
KEEP_UNLABELED_TRACKS = False
# Eğer KEEP_UNLABELED_TRACKS=True yaparsan, bu süre eşiğinin altındaki
# (frame cinsinden) etiketsiz parçalar yine de gürültü sayılıp atılır.
MIN_UNLABELED_TRACK_FRAMES = 300  # ~10 saniye @30fps (eski değer: 30 = 1 saniye, çok gevşekti)

def main():
    print("[SİSTEM] Dönüştürme işlemi başlıyor. 8 saatlik veri için bu işlem 1-2 dakika sürebilir...")

    # 1. Video boyutlarını ve FPS'i dinamik olarak al
    cap = cv2.VideoCapture(VIDEO_FILE)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    else:
        print(f"[UYARI] {VIDEO_FILE} dosyası okunamadı, varsayılan (1920x1080, 30fps) değerler kullanılıyor.")
        width, height, fps = 1920, 1080, 30.0
        print(
            "[BİLGİ] Gerçek fps'i öğrenmenin en güvenilir yolu: video dosyasını burada okuyabilmek "
            "(VIDEO_FILE yolunu doğru ayarla), ya da terminalden: "
            "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nk=1:nw=1 <video_dosyasi>"
        )

    # 2. Yapay zekanın ürettiği CSV'yi oku
    print("[SİSTEM] CSV dosyası belleğe alınıyor...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"[HATA] {CSV_FILE} dosyası bulunamadı!")
        return

    print("[SİSTEM] RAM optimizasyonu ve geçici kimlikleri (BT_ID) birleştirme algoritması çalışıyor...")

    # --- HAFIZA KORUMA VE ID BİRLEŞTİRME MANTIĞI ---
    # Her BT_ID için en çok okunan geçerli ( != -1) Aruco_ID'yi bul
    valid_aruco = df[df['Aruco_ID'] != -1]
    if not valid_aruco.empty:
        bt_to_aruco = valid_aruco.groupby('BT_ID')['Aruco_ID'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else -1
        ).to_dict()
    else:
        bt_to_aruco = {}

    # Geçici BT_ID'leri kalıcı isimlere dönüştür
    final_ids = []
    for bt_id in df['BT_ID']:
        a_id = bt_to_aruco.get(bt_id, -1)
        if a_id != -1:
            final_ids.append(f"Aruco_{int(a_id)}")
        else:
            final_ids.append(f"Isimsiz_Arı_{bt_id}")

    df['Final_ID'] = final_ids

    # --- Hangi Aruco ID'ler gerçek arı, hangileri muhtemel yanlış okuma? ---
    aruco_counts = df[df['Final_ID'].str.startswith('Aruco_')].groupby('Final_ID').size().sort_values(ascending=False)
    print("[BİLGİ] CSV'de görülen Aruco ID'leri ve kayıt sayıları:")
    for final_id, count in aruco_counts.items():
        print(f"         {final_id}: {count} kayıt")

    if KNOWN_BEE_ARUCO_IDS is not None:
        allowed_aruco_final_ids = {f"Aruco_{int(a)}" for a in KNOWN_BEE_ARUCO_IDS}
        dropped = [fid for fid in aruco_counts.index if fid not in allowed_aruco_final_ids]
        if dropped:
            print(f"[BİLGİ] KNOWN_BEE_ARUCO_IDS listesinde olmadığı için atılan Aruco ID'ler: {dropped}")
    else:
        if len(aruco_counts) > EXPECTED_BEE_COUNT:
            print(
                f"[UYARI] {len(aruco_counts)} farklı Aruco ID bulundu ama EXPECTED_BEE_COUNT={EXPECTED_BEE_COUNT}. "
                f"En sık görülen {EXPECTED_BEE_COUNT} tanesi gerçek arı, geri kalanı muhtemel yanlış okuma sayılacak. "
                f"Emin olmak için KNOWN_BEE_ARUCO_IDS'i elle ayarla."
            )
            allowed_aruco_final_ids = set(aruco_counts.index[:EXPECTED_BEE_COUNT])
            print(f"[BİLGİ] Otomatik seçilen Aruco ID'ler: {sorted(allowed_aruco_final_ids)}")
            print(f"[BİLGİ] Otomatik olarak atılan Aruco ID'ler: {sorted(set(aruco_counts.index) - allowed_aruco_final_ids)}")
        else:
            allowed_aruco_final_ids = set(aruco_counts.index)

    if KEEP_UNLABELED_TRACKS:
        track_lengths = df.groupby('Final_ID').size()
        allowed_unlabeled = set(track_lengths[
            (track_lengths >= MIN_UNLABELED_TRACK_FRAMES)
            & (~track_lengths.index.isin(aruco_counts.index))
        ].index)
        valid_final_ids = allowed_aruco_final_ids | allowed_unlabeled
    else:
        valid_final_ids = allowed_aruco_final_ids

    df = df[df['Final_ID'].isin(valid_final_ids)]

    # Eğer aynı karede aynı ID'den birden fazla varsa (çakışma), ortalamalarını alarak pürüzsüzleştir
    df = df.groupby(['Frame', 'Final_ID']).mean(numeric_only=True).reset_index()
    # -----------------------------------------------

    unique_ids = sorted(df['Final_ID'].unique())
    max_frame = int(df['Frame'].max())

    if KNOWN_VIDEO_DURATION_HOURS:
        estimated_fps = max_frame / (KNOWN_VIDEO_DURATION_HOURS * 3600.0)
        print(f"[BİLGİ] Süreye dayalı fps tahmini: son frame {max_frame} / {KNOWN_VIDEO_DURATION_HOURS} saat -> ~{estimated_fps:.2f} fps")
        if abs(estimated_fps - fps) > 0.5:
            print(f"[UYARI] Kullanılan fps ({fps}) ile süreye dayalı tahmin ({estimated_fps:.2f}) arasında fark var, kontrol et.")

    print(f"[SİSTEM] Başarılı! Toplam {len(unique_ids)} adet kalıcı arı kimliğine sıkıştırıldı.")
    print("[SİSTEM] Veriler matrise yazılıyor...")

    # 3. Analiz programı için matris oluştur (Her yer -1 ile dolu)
    data = np.full((max_frame, len(unique_ids) * 3), -1.0)

    # 4. Verileri matristeki doğru hücrelere yerleştir (vektörize - eski iterrows() döngüsü yerine)
    id_to_idx = {b_id: i for i, b_id in enumerate(unique_ids)}

    f_idx = df['Frame'].to_numpy(dtype=int) - 1
    b_idx = df['Final_ID'].map(id_to_idx).to_numpy(dtype=int)

    data[f_idx, b_idx * 3] = df['Center_X'].to_numpy(dtype=float)
    data[f_idx, b_idx * 3 + 1] = df['Center_Y'].to_numpy(dtype=float)
    data[f_idx, b_idx * 3 + 2] = df['Angle_Deg'].to_numpy(dtype=float)

    # 5. Sütun isimlerini belirle (Analiz programının beklediği format: ID_Aruco_1_X, vb.)
    cols = []
    for f_id in unique_ids:
        cols.extend([f"ID_{f_id}_X", f"ID_{f_id}_Y", f"ID_{f_id}_Ang"])

    out_df = pd.DataFrame(data, columns=cols)

    # 6. Klasör ve TSV dosyalarını oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    raw_path = os.path.join(OUTPUT_DIR, f"{EXP_NAME}_coordinates_raw.txt")
    out_df.to_csv(raw_path, sep="\t", index=False)

    filtered_path = os.path.join(OUTPUT_DIR, f"{EXP_NAME}_coordinates_filtered.txt")
    out_df.to_csv(filtered_path, sep="\t", index=False)

    # 7. INFO (.txt) Dosyasını Oluştur
    info_path = os.path.join(OUTPUT_DIR, f"{EXP_NAME}_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("# Video Info\n")
        f.write(f"video_path\t{VIDEO_FILE}\n")
        f.write(f"fps\t{fps}\n")
        f.write(f"frame_width\t{width}\n")
        f.write(f"frame_height\t{height}\n")
        f.write("start_time_str\t00:00:00\n")
        
        duration_s = max_frame / fps
        m, s = divmod(int(duration_s), 60)
        h, m = divmod(m, 60)
        f.write(f"end_time_str\t{h:02d}:{m:02d}:{s:02d}\n\n")

        f.write("# Inputs\n")
        f.write("tracking_method\tAI Tracker (RF-DETR + ByteTrack)\n\n")

        f.write("# Marker IDs\n")
        f.write("marker_ids\t" + ",".join(str(x) for x in unique_ids) + "\n")
        f.write("raw_missing_rule\t-1,-1,-1 means no real detection on that frame\n")
        f.write("filtered_prediction_rule\tIf raw is missing but filtered exists, that point was generated by Kalman prediction\n\n")

        f.write("# Scale\n")
        f.write("pixel_to_mm\t1.0\n\n")

    print(f"\n[BAŞARILI] Dönüşüm Tamamlandı!")
    print(f"Klasör Yolu: {OUTPUT_DIR}")
    print("Artık arayüzü ('analysis_gui.py') başlatıp bu klasörü yükleyebilirsiniz.")

if __name__ == "__main__":
    main()