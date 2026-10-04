# =====================================================================
#  Üç Modelin Karşılaştırılması: MobileNetV2, MobileNetV3Small, EfficientNetB0
#  Google Colab (GPU) üzerinde çalıştırılır. Veri sınıf bazında %70/%15/%15
#  bölünür (split_manifest.json); üç model aynı koşullarla eğitilip test setinde
#  doğruluk, macro F1, model boyutu ve işlem süresiyle karşılaştırılır.
# =====================================================================

# %% 0) Kaggle veri setini indir — kagglehub ile (hesap SADECE veri için)
from google.colab import files
import os

if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("kaggle.json yükle (Kaggle > Settings > API > Create Legacy Token):")
    uploaded = files.upload()  # kaggle.json seç
    os.makedirs("/root/.kaggle", exist_ok=True)
    for fname in uploaded:
        os.rename(fname, "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    import json as _json
    with open("/root/.kaggle/kaggle.json") as _f:
        _cred = _json.load(_f)
    os.environ["KAGGLE_USERNAME"] = _cred["username"]
    os.environ["KAGGLE_KEY"] = _cred["key"]

os.system("pip -q install kagglehub")
import kagglehub

DATASET_PATH = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print("Veri indirildi:", DATASET_PATH)
print("İçerik:", os.listdir(DATASET_PATH))

# %% 1) Importlar + ortak ayarlar
import csv
import glob
import json
import random
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet, mobilenet_v2, mobilenet_v3
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
assert tf.config.list_physical_devices("GPU"), (
    "GPU görünmüyor! Runtime > Change runtime type > GPU (T4) seçip tekrar dene."
)

IMG = 224
BATCH = 32
SEED = 42
MAX_PER_CLASS = None
EPOCHS_HEAD = 8
EPOCHS_FINE = 5
UNFREEZE_ORANI = 0.25  # base modelin son %25'i fine-tune'a açılır — 3 mimaride de AYNI oran
OUT_DIR = "/content/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ORANLAR = {"train": 0.70, "valid": 0.15, "test": 0.15}

# LeadLeaf AI — 10 günlük ZORUNLU kapsam: 38 sınıf değil, sadece domates + 4 yaygın hastalık.
SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]  # None = veri setindeki tüm 38 sınıf

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Karşılaştırılacak 3 model — her biri kendi doğru preprocess_input'unu taşıyor.
# (mobilenet_v3.preprocess_input ve efficientnet.preprocess_input BİLEREK pass-through:
#  bkz. yukarıdaki "ÇİFT NORMALİZASYON NOTU".)
MODEL_TANIMLARI = [
    {
        "ad": "MobileNetV2",
        "fabrika": keras.applications.MobileNetV2,
        "on_isle": mobilenet_v2.preprocess_input,
    },
    {
        "ad": "MobileNetV3Small",
        "fabrika": keras.applications.MobileNetV3Small,
        "on_isle": mobilenet_v3.preprocess_input,
    },
    {
        "ad": "EfficientNetB0",
        "fabrika": keras.applications.EfficientNetB0,
        "on_isle": efficientnet.preprocess_input,
    },
]

# %% 2) Veri yolu — HAM PlantVillage (renkli görseller). Mirror'a göre iç klasör
# adı değişebilir ("color", "PlantVillage/color" vb.) — bu yüzden dinamik arıyoruz.
CANDIDATES = [
    os.path.join(DATASET_PATH, "color"),
    os.path.join(DATASET_PATH, "PlantVillage", "color"),
    os.path.join(DATASET_PATH, "plantvillage dataset", "color"),
]
RAW_DIR = next((p for p in CANDIDATES if os.path.isdir(p)), None)
if RAW_DIR is None:
    bulunanlar = glob.glob(os.path.join(DATASET_PATH, "**", "color"), recursive=True)
    RAW_DIR = bulunanlar[0] if bulunanlar else None
assert RAW_DIR, (
    f"'color' klasoru otomatik bulunamadi. DATASET_PATH icerigi: {os.listdir(DATASET_PATH)}\n"
    f"Yukaridaki listeye bakip CANDIDATES listesine dogru yolu elle ekle."
)
print("Ham veri (renkli):", RAW_DIR)
mevcut_siniflar = sorted(os.listdir(RAW_DIR))
print(f"Veri setindeki toplam sinif sayisi: {len(mevcut_siniflar)}")

eksikler = [c for c in (SELECTED_CLASSES or []) if c not in mevcut_siniflar]
if eksikler:
    print(f"\nUYARI: su siniflar bulunamadi: {eksikler}")
    print("Domates ile ilgili gercek klasor adlari:")
    for c in mevcut_siniflar:
        if "tomato" in c.lower():
            print(" -", c)
    raise AssertionError("SELECTED_CLASSES'i yukaridaki gercek klasor adlarina gore duzelt.")

# Veri analizi — split'ten ÖNCE, ham hâliyle sınıf başına görüntü sayısı (dengesizlik kontrolü)
print("\nSeçilen 5 sınıfın görüntü sayıları (split öncesi, ham veri):")
toplam_goruntu = 0
for cls in (SELECTED_CLASSES or mevcut_siniflar):
    n = len(os.listdir(os.path.join(RAW_DIR, cls)))
    toplam_goruntu += n
    print(f"  {cls:32s} {n:5d}")
print(f"  {'TOPLAM':32s} {toplam_goruntu:5d}")

# %% 3) TEK SEFERLİK stratified train/valid/test bölmesi (%70/%15/%15) + manifest
SPLIT_DIR = "/content/_split"


def _stratified_uc_yonlu_split(kaynak_dir, hedef_kok, siniflar, oranlar, max_per_class, seed):
    """Her sınıfın kendi dosya listesini TEK SEFERDE karıştırıp train/valid/test'e
    böler (stratified: her sınıf kendi oranında bölünür, böylece küçük sınıflar
    testte kaybolmaz). Aynı görsel asla iki kümede birden olamaz — sızıntı yapısal
    olarak imkansız. Sonuç, split_manifest.json'da hangi dosyanın nereye düştüğü
    okunabilir şekilde saklanır (tekrarlanabilirlik + denetim için)."""
    hedefler = {k: os.path.join(hedef_kok, k) for k in ("train", "valid", "test")}
    if all(os.path.isdir(d) for d in hedefler.values()):
        print("Split klasörleri zaten var, yeniden oluşturulmuyor:", hedef_kok)
        return hedefler, None

    rng = random.Random(seed)
    kaynak_siniflar = siniflar if siniflar else os.listdir(kaynak_dir)
    manifest_dosyalar = {"train": {}, "valid": {}, "test": {}}

    for cls in kaynak_siniflar:
        src = os.path.join(kaynak_dir, cls)
        if not os.path.isdir(src):
            print(f"UYARI: sinif klasoru yok, atlaniyor: {cls}")
            continue
        dosyalar = os.listdir(src)
        rng.shuffle(dosyalar)
        if max_per_class:
            dosyalar = dosyalar[:max_per_class]

        n = len(dosyalar)
        n_test = max(1, int(round(n * oranlar["test"])))
        n_valid = max(1, int(round(n * oranlar["valid"])))
        n_train = n - n_test - n_valid
        assert n_train > 0, f"{cls}: train için yeterli görsel yok (n={n})"

        bolumler = {
            "test": dosyalar[:n_test],
            "valid": dosyalar[n_test:n_test + n_valid],
            "train": dosyalar[n_test + n_valid:],
        }
        for split_adi, split_dosyalari in bolumler.items():
            manifest_dosyalar[split_adi][cls] = split_dosyalari
            dst = os.path.join(hedefler[split_adi], cls)
            os.makedirs(dst, exist_ok=True)
            for f in split_dosyalari:
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        print(f"  {cls}: {len(bolumler['train'])} train, {len(bolumler['valid'])} valid, "
              f"{len(bolumler['test'])} test")

    manifest = {
        "veri_kaynagi": "abdallahalidev/plantvillage-dataset (kagglehub)",
        "seed": seed,
        "oranlar": oranlar,
        "siniflar": kaynak_siniflar,
        "sayilar": {
            split_adi: {cls: len(dosyalar) for cls, dosyalar in split_dict.items()}
            for split_adi, split_dict in manifest_dosyalar.items()
        },
        "dosyalar": manifest_dosyalar,
    }
    return hedefler, manifest


SPLIT_DIRS, MANIFEST = _stratified_uc_yonlu_split(
    RAW_DIR, SPLIT_DIR, SELECTED_CLASSES, ORANLAR, MAX_PER_CLASS, SEED,
)
if MANIFEST is not None:
    with open(f"{OUT_DIR}/split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, ensure_ascii=False, indent=2)
    print("split_manifest.json kaydedildi.")
else:
    print("UYARI: split klasörleri önceden vardı, split_manifest.json bu çalıştırmada YENİDEN üretilmedi.")

TRAIN_DIR, VALID_DIR, TEST_DIR = SPLIT_DIRS["train"], SPLIT_DIRS["valid"], SPLIT_DIRS["test"]

# %% 4) tf.data setleri — RAW piksel (0-255, float32); normalizasyon MODELİN İÇİNDE
# yapılıyor (her model kendi preprocess_input'unu uyguluyor), böylece dataset
# katmanında YANLIŞLIKLA bir normalizasyon daha eklenip "çift normalizasyon"
# hatasına düşme riski yok.
def veri_seti_yukle(dizin, batch_size, shuffle):
    return keras.utils.image_dataset_from_directory(
        dizin, image_size=(IMG, IMG), batch_size=batch_size, seed=SEED, shuffle=shuffle,
    )


train_ds_ham = veri_seti_yukle(TRAIN_DIR, BATCH, shuffle=True)
val_ds_ham = veri_seti_yukle(VALID_DIR, BATCH, shuffle=False)
# Test seti batch_size=1: hem sklearn metrikleri için tek tek tahmin, hem de
# "görüntü başına ortalama inference süresi" ölçümü aynı döngüde yapılabiliyor.
test_ds_ham = veri_seti_yukle(TEST_DIR, 1, shuffle=False)

class_names = train_ds_ham.class_names
print(len(class_names), "sinif:", class_names)
with open(f"{OUT_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds_ham = train_ds_ham.prefetch(AUTOTUNE)
val_ds_ham = val_ds_ham.prefetch(AUTOTUNE)
# test_ds_ham prefetch YOK — inference süresi ölçümünü prefetch'in gizli
# paralelliğiyle bulandırmamak için bilerek.

y_true_test = np.concatenate([y.numpy() for _, y in test_ds_ham])

# %% 5) Ortak veri artırma katmanı — 3 modelde de AYNI parametrelerle kullanılıyor
def yeni_augment_katmani():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augment")


def yeni_early_stopping():
    return keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True,
    )


def model_kur(fabrika, on_isle_fn, num_classes):
    inputs = keras.Input((IMG, IMG, 3))
    x = yeni_augment_katmani()(inputs)
    x = on_isle_fn(x)  # her model kendi doğru preprocess'ini burada uyguluyor
    base = fabrika(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)  # az tutuluyor — küçük veri setinde 0.3+ eksik öğrenmeye (underfit) yol açabiliyor
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


# %% 6) Değerlendirme — SADECE bağımsız test setinde, tüm modeller için AYNI protokol
def modeli_degerlendir(model, test_ds, y_true):
    # sklearn metrikleri için toplu tahmin
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    dogruluk = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    # macro AUC (one-vs-rest) — accuracy tek başına dengesiz sınıflarda yanıltıcı olabildiği
    # için modelin sınıfları ne kadar iyi AYIRT ettiğini de ölçüyoruz
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")  # test setinde bir sınıf hiç örnek içermiyorsa
    cm = confusion_matrix(y_true, y_pred)

    # Görüntü başına ortalama inference süresi — tek tek, gerçek kullanım senaryosuna
    # yakın (FastAPI da tek görsel alıyor). İlk çağrı (warmup / graph tracing) dahil
    # edilmiyor, ölçümü çarpıtmasın diye.
    sureler_ms = []
    for i, (x, _) in enumerate(test_ds):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        t1 = time.perf_counter()
        if i > 0:  # ilk çağrı warmup, ölçüme katılmıyor
            sureler_ms.append((t1 - t0) * 1000)
    ort_ms = float(np.mean(sureler_ms)) if sureler_ms else float("nan")

    return {
        "dogruluk": float(dogruluk),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_auc": macro_auc,
        "confusion_matrix": cm,
        "ort_inference_ms": ort_ms,
    }


def grafik_kaydet_confusion(cm, ad, class_names):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {ad} (bağımsız test seti)")
    plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(class_names)), class_names, fontsize=7)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix_{ad}.png", dpi=120)
    plt.close()


def parametre_ozeti(model, etiket):
    """Açıklanabilirlik için: bu anda kaç parametre eğitilebilir, kaçı donuk.
    Referans ders defterindeki (cnn2_ders.ipynb, 11.3) aynı isimli fonksiyonun
    karşılığı — hangi aşamada ne kadarının açık olduğunu somut sayıyla gösterir."""
    egitilebilir = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    donuk = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    print(f"  {etiket:38s} eğitilebilir: {egitilebilir:>10,}   donuk: {donuk:>10,}")
    return egitilebilir, donuk


def grafik_kaydet_ogrenme_egrisi(h1, h2, ad):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    kayip = h1.history["loss"] + h2.history["loss"]
    vkayip = h1.history["val_loss"] + h2.history["val_loss"]
    sinir = len(h1.history["accuracy"]) - 0.5

    sekil, (sol, sag) = plt.subplots(1, 2, figsize=(13, 4.4))
    sol.plot(acc, label="eğitim"); sol.plot(vacc, label="doğrulama")
    sol.axvline(sinir, color="gray", linestyle="--", label="fine-tuning başlangıcı")
    sol.set_xlabel("epoch"); sol.set_ylabel("doğruluk"); sol.legend()
    sol.set_title(f"{ad} — Doğruluk")

    sag.plot(kayip, label="eğitim"); sag.plot(vkayip, label="doğrulama")
    sag.axvline(sinir, color="gray", linestyle="--", label="fine-tuning başlangıcı")
    sag.set_xlabel("epoch"); sag.set_ylabel("kayıp (loss)"); sag.legend()
    sag.set_title(f"{ad} — Kayıp")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ogrenme_egrisi_{ad}.png", dpi=120)
    plt.close()


# %% 7) Üç modeli AYNI koşullarda sırayla eğit + değerlendir
karsilastirma_satirlari = []

for tanim in MODEL_TANIMLARI:
    ad, fabrika, on_isle_fn = tanim["ad"], tanim["fabrika"], tanim["on_isle"]
    print(f"\n{'=' * 60}\n{ad} eğitiliyor\n{'=' * 60}")

    model, base = model_kur(fabrika, on_isle_fn, len(class_names))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print(f"  Gövde toplam katman: {len(base.layers)}   toplam parametre: {base.count_params():,}")
    egit_1, donuk_1 = parametre_ozeti(model, "1. aşama (base tamamen donuk)")

    # 1. aşama — üst katmanı eğit (base dondurulmuş)
    h1 = model.fit(train_ds_ham, validation_data=val_ds_ham, epochs=EPOCHS_HEAD,
                    callbacks=[yeni_early_stopping()])

    # 2. aşama — fine-tuning: base'in son %UNFREEZE_ORANI'ı açılıyor (3 mimaride de AYNI oran)
    base.trainable = True
    n_freeze = int(len(base.layers) * (1 - UNFREEZE_ORANI))
    for layer in base.layers[:n_freeze]:
        layer.trainable = False
    print(f"{ad}: {len(base.layers)} katmandan son {len(base.layers) - n_freeze} tanesi fine-tune'a açıldı.")

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    egit_2, donuk_2 = parametre_ozeti(model, f"2. aşama (son %{int(UNFREEZE_ORANI * 100)} açık)")

    h2 = model.fit(train_ds_ham, validation_data=val_ds_ham, epochs=EPOCHS_FINE,
                    callbacks=[yeni_early_stopping()])

    sonuc = modeli_degerlendir(model, test_ds_ham, y_true_test)
    print(f"{ad} — test doğruluğu: {sonuc['dogruluk']:.4f}, macro F1: {sonuc['macro_f1']:.4f}, "
          f"macro AUC: {sonuc['macro_auc']:.4f}, ort. inference: {sonuc['ort_inference_ms']:.1f} ms/görüntü")

    model_dosya = f"{OUT_DIR}/model_{ad}.keras"
    model.save(model_dosya)
    model_boyutu_mb = round(os.path.getsize(model_dosya) / (1024 * 1024), 2)

    grafik_kaydet_confusion(sonuc["confusion_matrix"], ad, class_names)
    grafik_kaydet_ogrenme_egrisi(h1, h2, ad)

    karsilastirma_satirlari.append({
        "model": ad,
        "dogruluk": round(sonuc["dogruluk"], 4),
        "macro_precision": round(sonuc["macro_precision"], 4),
        "macro_recall": round(sonuc["macro_recall"], 4),
        "macro_f1": round(sonuc["macro_f1"], 4),
        "macro_auc": round(sonuc["macro_auc"], 4),
        "model_boyutu_mb": model_boyutu_mb,
        "ort_inference_ms": round(sonuc["ort_inference_ms"], 2),
        # açıklanabilirlik: her aşamada kaç parametre gerçekten güncellendi
        "egitilebilir_parametre_1_asama": egit_1,
        "donuk_parametre_1_asama": donuk_1,
        "egitilebilir_parametre_2_asama": egit_2,
        "donuk_parametre_2_asama": donuk_2,
    })

    # bellek temizliği — 3 model art arda eğitilirken Colab GPU RAM'i şişmesin
    del model, base
    keras.backend.clear_session()

# %% 8) model_comparison.csv
with open(f"{OUT_DIR}/model_comparison.csv", "w", newline="", encoding="utf-8") as f:
    yazici = csv.DictWriter(f, fieldnames=list(karsilastirma_satirlari[0].keys()))
    yazici.writeheader()
    yazici.writerows(karsilastirma_satirlari)

print("\nmodel_comparison.csv:")
for satir in karsilastirma_satirlari:
    print(satir)

# %% 8a) Doğruluk karşılaştırma grafiği (3 model yan yana) — Streamlit ve rapor için
isimler = [s["model"] for s in karsilastirma_satirlari]
dogruluklar = [s["dogruluk"] for s in karsilastirma_satirlari]
f1ler = [s["macro_f1"] for s in karsilastirma_satirlari]

x = np.arange(len(isimler))
genislik = 0.35
plt.figure(figsize=(7, 5))
plt.bar(x - genislik / 2, dogruluklar, genislik, label="doğruluk", color="#346cb0")
plt.bar(x + genislik / 2, f1ler, genislik, label="macro F1", color="#4e8a6b")
for i, (d, f1) in enumerate(zip(dogruluklar, f1ler)):
    plt.text(i - genislik / 2, d + 0.01, f"{d:.3f}", ha="center", fontsize=9)
    plt.text(i + genislik / 2, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=9)
plt.xticks(x, isimler)
plt.ylabel("skor (bağımsız test seti)")
plt.ylim(0, 1.08)
plt.title("Model karşılaştırması — doğruluk ve macro F1")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_karsilastirma_dogruluk.png", dpi=120)
plt.close()
print("model_karsilastirma_dogruluk.png kaydedildi.")

# %% 8b) Üretim (tek-model) akışı için en iyi modeli seç ve model.keras olarak kopyala
# inference/app.py'nin n8n/Telegram'a bakan /predict'i TEK model bekliyor. Hangi mimari
# kazanırsa kazansın (MobileNetV2 olması ŞART değil) — bu yüzden hangi mimari olduğunu
# model_meta.json'a da yazıyoruz; inference/app.py bunu okuyup DOĞRU preprocess_input'u
# seçiyor. Bunu atlarsak üretimde yanlış modele yanlış preprocessing uygulanır
# (tam da kaçınmaya çalıştığımız "çift/yanlış normalizasyon" hatası, ama üretimde).
en_iyi = max(karsilastirma_satirlari, key=lambda s: s["dogruluk"])
print(f"\nÜretim (tek-model) akışı için seçilen model: {en_iyi['model']} "
      f"(doğruluk={en_iyi['dogruluk']}, macro_f1={en_iyi['macro_f1']})")
shutil.copy(f"{OUT_DIR}/model_{en_iyi['model']}.keras", f"{OUT_DIR}/model.keras")
with open(f"{OUT_DIR}/model_meta.json", "w", encoding="utf-8") as f:
    json.dump({"mimari": en_iyi["model"]}, f, ensure_ascii=False, indent=2)

# %% 9) Demo görselleri (Streamlit karşılaştırma sekmesi için) — test setinden örnek
os.makedirs(f"{OUT_DIR}/demo_images", exist_ok=True)
for cls in random.sample(class_names, min(8, len(class_names))):
    dosyalar = glob.glob(os.path.join(TEST_DIR, cls, "*"))
    if dosyalar:
        src = random.choice(dosyalar)
        shutil.copy(src, f"{OUT_DIR}/demo_images/{cls}.jpg")
print("demo_images/ hazir.")

# %% 10) Hepsini ziple ve indir (Colab oturumu kapanınca /content silinir!)
shutil.make_archive("/content/leadleaf_tubitak_models", "zip", OUT_DIR)
files.download("/content/leadleaf_tubitak_models.zip")
print(f"""
BİTTİ. leadleaf_tubitak_models.zip indi -> aç, içindekileri İKİ AYRI yere dağıt:

  1) model/  (ÜRETİM — n8n/Telegram akışı ve Streamlit "Analiz" sekmesi bunu kullanır)
       - model.keras        (otomatik seçilen en iyi model: {en_iyi['model']})
       - model_meta.json    (inference/app.py bunu okuyup doğru preprocess'i seçer)
       - class_names.json
       - demo_images/

  2) model/tubitak/  (Streamlit "Model Karşılaştırma" sekmesi)
       - model_MobileNetV2.keras
       - model_MobileNetV3Small.keras
       - model_EfficientNetB0.keras
       - class_names.json
       - split_manifest.json
       - model_comparison.csv
       - model_karsilastirma_dogruluk.png   (Streamlit'te otomatik gösterilir)

class_names.json ikisine de aynı şekilde kopyalanabilir (3 model de aynı sınıfları kullanıyor).
""")
