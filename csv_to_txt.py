import pandas as pd

# 1. Kaydettiğin güncel CSV dosyasını oku
df = pd.read_csv("bee_trajectories.csv")

# 2. TXT dosyasına düzgün hizalanmış bir tablo formatında yaz
with open("bee_trajectories_duzenli.txt", "w", encoding="utf-8") as f:
    # index=False: Baştaki gereksiz satır numaralarını kaldırır
    # justify='center': Sütun başlıklarını verilerin tam ortasına hizalar
    f.write(df.to_string(index=False, justify='center'))

print("[BAŞARILI] 'bee_trajectories_duzenli.txt' dosyası oluşturuldu!")