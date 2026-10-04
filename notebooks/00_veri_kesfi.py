# =====================================================================
#  Veri Seti Keşfi (EDA): PlantVillage
#  Google Colab üzerinde çalıştırılır. Sınıf dağılımı, görsel boyutları ve
#  tekrar eden (duplicate) görseller incelenir; çıktılar eda_ciktilari/ altına yazılır.
# =====================================================================

# %% 0) Kurulum + veri yolu (01_train_model_colab.py'deki indirme hücresi çalıştırılmış olmalı)
import os, json, random
os.system("pip -q install imagehash")

CANDIDATES = [
    os.path.join(DATASET_PATH, "color"),
    os.path.join(DATASET_PATH, "PlantVillage", "color"),
    os.path.join(DATASET_PATH, "plantvillage dataset", "color"),
]
RAW_DIR = next((p for p in CANDIDATES if os.path.isdir(p)), None)
if RAW_DIR is None:
    import glob as _glob
    bulunanlar = _glob.glob(os.path.join(DATASET_PATH, "**", "color"), recursive=True)
    RAW_DIR = bulunanlar[0] if bulunanlar else None
assert RAW_DIR, (
    "Veri seti bulunamadi — once 01_train_model_colab.py'deki indirme hucresini "
    "calistir (DATASET_PATH degiskeni tanimli olmali)."
)

OUT = "/content/eda_ciktilari"
os.makedirs(OUT, exist_ok=True)

SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]

# %% 1) Sınıf dağılımı — TÜM sınıflar (ham veri, henüz train/valid'e bölünmemiş)
import matplotlib.pyplot as plt

def sinif_sayilari(klasor):
    return {c: len(os.listdir(os.path.join(klasor, c)))
            for c in sorted(os.listdir(klasor)) if os.path.isdir(os.path.join(klasor, c))}

tum_sayilar = sinif_sayilari(RAW_DIR)
toplam_gorsel = sum(tum_sayilar.values())
print(f"TÜM VERİ SETİ (ham) — {len(tum_sayilar)} sınıf, {toplam_gorsel} görsel")

# Domates alt kümesi
tomato_sayilar = {k: v for k, v in tum_sayilar.items() if k in SELECTED_CLASSES}
print(f"\nDOMATES ALT KÜMESİ (MVP) — {len(tomato_sayilar)} sınıf")
for c in SELECTED_CLASSES:
    print(f"  {c:35s} {tomato_sayilar.get(c, 0):5d} görsel")

# Dengesizlik oranı (en büyük / en küçük sınıf)
if tomato_sayilar:
    max_c, min_c = max(tomato_sayilar.values()), min(tomato_sayilar.values())
    print(f"\nDengesizlik oranı (domates alt kümesi): {max_c/min_c:.2f}x "
          f"(en çok: {max_c}, en az: {min_c})")
    print(f"%80 train / %20 valid ile bölününce yaklaşık: "
          f"train={int(min_c*0.8)}-{int(max_c*0.8)}, valid={int(min_c*0.2)}-{int(max_c*0.2)} "
          f"görsel/sınıf aralığında olacak.")

# Grafik: tüm sınıflar (domates olanlar vurgulu)
fig, ax = plt.subplots(figsize=(14, 10))
siniflar = list(tum_sayilar.keys())
degerler = [tum_sayilar[c] for c in siniflar]
renkler = ["#e74c3c" if c in SELECTED_CLASSES else "#3498db" for c in siniflar]
ax.barh(siniflar, degerler, color=renkler)
ax.set_xlabel("Görsel sayısı (ham veri)")
ax.set_title("Sınıf dağılımı — tüm sınıflar (KIRMIZI = MVP'de kullanılan domates alt kümesi)")
plt.tight_layout(); plt.savefig(f"{OUT}/sinif_dagilimi.png", dpi=120); plt.show()

# %% 2) Örnek görseller — domates 5 sınıftan birer izgara
from PIL import Image

fig, axes = plt.subplots(1, len(SELECTED_CLASSES), figsize=(4 * len(SELECTED_CLASSES), 4))
for ax, cls in zip(axes, SELECTED_CLASSES):
    klasor = os.path.join(RAW_DIR, cls)
    dosya = random.choice(os.listdir(klasor))
    img = Image.open(os.path.join(klasor, dosya))
    ax.imshow(img)
    ax.set_title(f"{cls}\n{img.size[0]}x{img.size[1]}px", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/ornek_gorseller_izgara.png", dpi=120); plt.show()

# %% 3) Görsel boyutu dağılımı (örnekleme — 300 görsel, hepsini açmak yavaş)
genislikler, yukseklikler = [], []
ornek_havuzu = []
for cls in SELECTED_CLASSES:
    klasor = os.path.join(RAW_DIR, cls)
    dosyalar = os.listdir(klasor)
    ornek_havuzu += [os.path.join(klasor, f) for f in random.sample(dosyalar, min(60, len(dosyalar)))]

for yol in ornek_havuzu:
    with Image.open(yol) as im:
        genislikler.append(im.size[0]); yukseklikler.append(im.size[1])

print(f"\nÖrneklenen {len(ornek_havuzu)} görsel — boyut istatistiği:")
print(f"  genişlik: min={min(genislikler)} max={max(genislikler)} ortalama={sum(genislikler)/len(genislikler):.0f}")
print(f"  yükseklik: min={min(yukseklikler)} max={max(yukseklikler)} ortalama={sum(yukseklikler)/len(yukseklikler):.0f}")
tekboyut = len(set(zip(genislikler, yukseklikler))) == 1
print(f"  Tüm görseller aynı boyutta mı: {'EVET — ' + str(genislikler[0]) + 'x' + str(yukseklikler[0]) if tekboyut else 'HAYIR, karışık'}")

plt.figure(figsize=(7, 4))
plt.hist(genislikler, bins=20, alpha=0.6, label="genişlik")
plt.hist(yukseklikler, bins=20, alpha=0.6, label="yükseklik")
plt.legend(); plt.title("Görsel boyutu dağılımı (piksel, örneklem)")
plt.tight_layout(); plt.savefig(f"{OUT}/boyut_dagilimi.png", dpi=120); plt.show()

# %% 3.5) BOZUK / AÇILAMAYAN GÖRSEL kontrolü ("boş veri" kontrolü)
# Görsel veri setlerinde "eksik deger" tablo verisindeki gibi olmaz — bunun
# karsiligi: 0 byte'lik, bozuk veya PIL'in acamadigi dosyalardir. Her sinifin
# TAMAMINI (orneklem degil) kontrol ediyoruz - sadece ac/dogrula oldugu icin hizli.
bozuk_rapor = {}
for cls in SELECTED_CLASSES:
    klasor = os.path.join(RAW_DIR, cls)
    dosyalar = os.listdir(klasor)
    bozuk = []
    for f in dosyalar:
        yol = os.path.join(klasor, f)
        try:
            if os.path.getsize(yol) == 0:
                bozuk.append(f)
                continue
            with Image.open(yol) as im:
                im.verify()
        except Exception:
            bozuk.append(f)
    bozuk_rapor[cls] = {"toplam": len(dosyalar), "bozuk": len(bozuk), "bozuk_dosyalar": bozuk[:20]}
    print(f"{cls:35s} toplam={len(dosyalar):5d}  bozuk/acilamayan={len(bozuk)}")

toplam_bozuk = sum(v["bozuk"] for v in bozuk_rapor.values())
if toplam_bozuk:
    print(f"\nUYARI: {toplam_bozuk} bozuk/acilamayan gorsel bulundu — bunlar egitimde "
          f"hataya sebep olabilir, split'ten once temizlenmesi onerilir.")
else:
    print("\nOK — bozuk/acilamayan gorsel bulunamadi.")

# %% 4) TEKRAR EDEN (duplicate) GÖRSEL kontrolü — perceptual hash
# Split'ten ÖNCE, ham veri havuzunda ayni/cok benzer goruntu var mi kontrol
# ediyoruz. Varsa, 01_train_model_colab.py'nin rastgele train/valid bolmesi
# bu ikiliyi ayirip yapay bir sizinti yaratabilir.
import imagehash

def hashle(klasor, sinif, adet=150):
    dosyalar = os.listdir(os.path.join(klasor, sinif))
    secilen = random.sample(dosyalar, min(adet, len(dosyalar)))
    out = {}
    for f in secilen:
        try:
            with Image.open(os.path.join(klasor, sinif, f)) as im:
                out[f] = imagehash.phash(im)
        except Exception:
            pass
    return out

tekrar_rapor = []
for cls in SELECTED_CLASSES:
    hashler = hashle(RAW_DIR, cls, adet=200)
    isimler = list(hashler.keys())
    degerler_h = list(hashler.values())
    tekrar_sayisi = 0
    for i in range(len(degerler_h)):
        for j in range(i + 1, len(degerler_h)):
            if (degerler_h[i] - degerler_h[j]) <= 4:  # Hamming mesafesi esigi
                tekrar_sayisi += 1
    oran = tekrar_sayisi / len(isimler) if isimler else 0
    tekrar_rapor.append((cls, len(isimler), tekrar_sayisi, oran))
    print(f"{cls:35s} örneklenen={len(isimler):3d} -> tekrar eden CIFT sayisi: "
          f"{tekrar_sayisi} (%{oran*100:.1f})")

with open(f"{OUT}/tekrar_raporu.txt", "w", encoding="utf-8") as f:
    f.write("TEKRAR EDEN (duplicate) GORSEL kontrolu — orneklem bazli perceptual hash\n")
    f.write("Not: Bu KESIN degil, orneklem uzerinden bir tahmindir. Oran yuksekse\n")
    f.write("(ör. >%5) split oncesi tekilleştirme (deduplication) dusunulmeli —\n")
    f.write("aksi halde ayni fotografin iki kopyasi train/valid'e ayri ayri dusebilir.\n\n")
    for cls, n, tk, oran in tekrar_rapor:
        f.write(f"{cls}: orneklenen={n} tekrar_cift={tk} oran=%{oran*100:.1f}\n")

# %% 5) Özet dosyası + Streamlit için JSON
ozet = f"""VERİ SETİ ÖZETİ — PlantVillage Dataset (abdallahalidev, Kaggle, HAM veri)

TÜM VERİ SETİ: {len(tum_sayilar)} sınıf, {toplam_gorsel} görsel (henüz bölünmemiş)
DOMATES ALT KÜMESİ (MVP, {len(SELECTED_CLASSES)} sınıf):
""" + "\n".join(f"  - {c}: {tomato_sayilar.get(c, 0)} görsel"
                 for c in SELECTED_CLASSES) + f"""

Dengesizlik oranı (en çok/en az): {max(tomato_sayilar.values())/min(tomato_sayilar.values()):.2f}x
Görsel boyutu: {'sabit ' + str(genislikler[0]) + 'x' + str(yukseklikler[0]) + 'px' if tekboyut else 'değişken'}
Bozuk/açılamayan görsel: {toplam_bozuk} adet (bkz. yukarıdaki sınıf bazlı döküm)
Train/valid bölmesi: 01_train_model_colab.py tarafından %80/%20, TEK SEFERDE ve
  sabit seed (42) ile rastgele yapılıyor — bu yüzden vipoooool veri setindeki gibi
  bir "önceden çoğaltılmış" kaynaklı sızıntı riski YOK.

SINIRLILIKLAR (rapora eklenecek):
1. Görseller laboratuvar/kontrollü arka planda çekilmiş (PlantVillage kökenli),
   gerçek tarla fotoğrafları DEĞİL — gerçek koşullarda başarım düşebilir.
2. Ham veri havuzunda tespit edilen tekrar eden (duplicate) görsel oranı
   tekrar_raporu.txt'de listelendi — yüksekse (>%5) split öncesi tekilleştirme
   düşünülmeli, aksi halde train/valid arasında dolaylı bir sızıntı oluşabilir.
3. Sınıf dengesizliği {max(tomato_sayilar.values())/min(tomato_sayilar.values()):.1f}x — eğitimde
   class_weight veya daha fazla augmentation ile dengelenmesi düşünülebilir.
4. {"Bozuk/açılamayan " + str(toplam_bozuk) + " görsel bulundu — split öncesi temizlenmeli." if toplam_bozuk else "Bozuk/açılamayan görsel bulunamadı."}
"""
with open(f"{OUT}/veri_ozeti.txt", "w", encoding="utf-8") as f:
    f.write(ozet)
print("\n" + ozet)

# Streamlit'in "Veri Analizi" sekmesinin okuyacağı yapılandırılmış özet.
json_ozet = {
    "toplam_sinif": len(tum_sayilar),
    "toplam_gorsel": toplam_gorsel,
    "domates_sayilar": tomato_sayilar,
    "dengesizlik_orani": round(max(tomato_sayilar.values()) / min(tomato_sayilar.values()), 2),
    "gorsel_boyutu": {"sabit": tekboyut, "genislik": genislikler[0], "yukseklik": yukseklikler[0]}
                      if tekboyut else {"sabit": False},
    "bozuk_gorsel": {"toplam": toplam_bozuk, "sinif_bazli": {c: v["bozuk"] for c, v in bozuk_rapor.items()}},
    "tekrar_eden": {c: {"orneklenen": n, "tekrar_cift": tk, "oran_yuzde": round(oran * 100, 1)}
                    for c, n, tk, oran in tekrar_rapor},
}
with open(f"{OUT}/veri_ozeti.json", "w", encoding="utf-8") as f:
    json.dump(json_ozet, f, ensure_ascii=False, indent=2)

# %% 6) İndir
import shutil
from google.colab import files
shutil.make_archive("/content/eda_ciktilari", "zip", OUT)
files.download("/content/eda_ciktilari.zip")
print("BİTTİ — eda_ciktilari.zip indi. report/ klasörüne koy, rapora görselleri ve veri_ozeti.txt'yi işle.")
