# =====================================================================
#  EfficientNetB0: Kademeli Fine-Tuning
#  Google Colab (GPU) üzerinde çalıştırılır. Aynı veri bölmesiyle gövde üç
#  aşamada açılır (%15, %30, %40) ve ReduceLROnPlateau ile azalan öğrenme oranı
#  kullanılır; sonuç ilk eğitim tarifiyle karşılaştırılır.
# =====================================================================

# %% 0) Kaggle veri setini indir
from google.colab import files
import os

if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("kaggle.json yükle (Kaggle > Settings > API > Create Legacy Token):")
    uploaded = files.upload()
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

# %% 1) Importlar + ortak ayarlar (01_train_model_colab.py ile AYNI split/seed)
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
from tensorflow.keras.applications import efficientnet
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

print("TensorFlow:", tf.__version__)
assert tf.config.list_physical_devices("GPU"), (
    "GPU görünmüyor! Runtime > Change runtime type > GPU (T4) seçip tekrar dene."
)

IMG = 224
BATCH = 32
SEED = 42  # 01_train_model_colab.py ile AYNI seed -> AYNI split_manifest üretir
EPOCHS_HEAD = 8

# --- Değişiklik 1: daha uzun eğitim + daha sabırlı early stopping ---
FINE_TUNE_PATIENCE = 5          # eskisi: 3
REDUCE_LR_PATIENCE = 2          # ReduceLROnPlateau icin

# --- Değişiklik 2 + 3: kademeli açma, nihayetinde %40'a kadar (eskisi: tek adımda %25) ---
UNFREEZE_ASAMALARI = [0.15, 0.30, 0.40]
ASAMA_EPOCH_SAYILARI = [6, 6, 8]   # her aşamaya ayrılan ÜST SINIR (early stopping erken kesebilir)

OUT_DIR = "/content/outputs_gelismis"
os.makedirs(OUT_DIR, exist_ok=True)

# 01_train_model_colab.py'nin bu çalıştırmasındaki EfficientNetB0 sonucu (model_comparison.csv'den) —
# eski-yeni karşılaştırma grafiği için burada sabit değer olarak taşınıyor (Colab'da o CSV yok).
ESKI_SONUC = {
    "dogruluk": 0.9468,
    "macro_precision": 0.9454,
    "macro_recall": 0.937,
    "macro_f1": 0.940,
    "macro_auc": 0.9958,
    "model_boyutu_mb": 36.76,
    "ort_inference_ms": 91.28,
}

ORANLAR = {"train": 0.70, "valid": 0.15, "test": 0.15}

SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# %% 2) Veri yolu + sınıf kontrolü (01_train_model_colab.py ile aynı mantık)
CANDIDATES = [
    os.path.join(DATASET_PATH, "color"),
    os.path.join(DATASET_PATH, "PlantVillage", "color"),
    os.path.join(DATASET_PATH, "plantvillage dataset", "color"),
]
RAW_DIR = next((p for p in CANDIDATES if os.path.isdir(p)), None)
if RAW_DIR is None:
    bulunanlar = glob.glob(os.path.join(DATASET_PATH, "**", "color"), recursive=True)
    RAW_DIR = bulunanlar[0] if bulunanlar else None
assert RAW_DIR, f"'color' klasoru bulunamadi. Icerik: {os.listdir(DATASET_PATH)}"
print("Ham veri:", RAW_DIR)

# %% 3) AYNI stratified split (SEED=42 sayesinde 01_train_model_colab.py'deki ile BİREBİR AYNI)
SPLIT_DIR = "/content/_split"


def _stratified_uc_yonlu_split(kaynak_dir, hedef_kok, siniflar, oranlar, seed):
    hedefler = {k: os.path.join(hedef_kok, k) for k in ("train", "valid", "test")}
    if all(os.path.isdir(d) for d in hedefler.values()):
        return hedefler
    rng = random.Random(seed)
    for cls in siniflar:
        src = os.path.join(kaynak_dir, cls)
        dosyalar = os.listdir(src)
        rng.shuffle(dosyalar)
        n = len(dosyalar)
        n_test = max(1, int(round(n * oranlar["test"])))
        n_valid = max(1, int(round(n * oranlar["valid"])))
        bolumler = {
            "test": dosyalar[:n_test],
            "valid": dosyalar[n_test:n_test + n_valid],
            "train": dosyalar[n_test + n_valid:],
        }
        for split_adi, split_dosyalari in bolumler.items():
            dst = os.path.join(hedefler[split_adi], cls)
            os.makedirs(dst, exist_ok=True)
            for f in split_dosyalari:
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    return hedefler


SPLIT_DIRS = _stratified_uc_yonlu_split(RAW_DIR, SPLIT_DIR, SELECTED_CLASSES, ORANLAR, SEED)
TRAIN_DIR, VALID_DIR, TEST_DIR = SPLIT_DIRS["train"], SPLIT_DIRS["valid"], SPLIT_DIRS["test"]

# %% 4) tf.data setleri
def veri_seti_yukle(dizin, batch_size, shuffle):
    return keras.utils.image_dataset_from_directory(
        dizin, image_size=(IMG, IMG), batch_size=batch_size, seed=SEED, shuffle=shuffle,
    )


train_ds_ham = veri_seti_yukle(TRAIN_DIR, BATCH, shuffle=True)
val_ds_ham = veri_seti_yukle(VALID_DIR, BATCH, shuffle=False)
test_ds_ham = veri_seti_yukle(TEST_DIR, 1, shuffle=False)

# class_names, .prefetch() UYGULANMADAN ÖNCE okunmalı — prefetch() döndürdüğü
# _PrefetchDataset nesnesi bu özniteliği taşımıyor (AttributeError verir).
class_names = train_ds_ham.class_names
print(len(class_names), "sinif:", class_names)

train_ds_ham = train_ds_ham.prefetch(tf.data.AUTOTUNE)
val_ds_ham = val_ds_ham.prefetch(tf.data.AUTOTUNE)
y_true_test = np.concatenate([y.numpy() for _, y in test_ds_ham])

# %% 5) Model kurma (01_train_model_colab.py'deki model_kur ile AYNI iskelet)
def yeni_augment_katmani():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augment")


inputs = keras.Input((IMG, IMG, 3))
x = yeni_augment_katmani()(inputs)
x = efficientnet.preprocess_input(x)
base = keras.applications.EfficientNetB0(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False
x = base(x, training=False)  # BatchNorm katmanları her zaman inference modunda kalır (tuzak 4'e karşı)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)


def parametre_ozeti(model, etiket):
    egitilebilir = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    donuk = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    print(f"  {etiket:45s} eğitilebilir: {egitilebilir:>10,}   donuk: {donuk:>10,}")


# %% 6) AŞAMA 0 — kafa eğitimi (öncekiyle AYNI: base tamamen donuk)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
parametre_ozeti(model, "Aşama 0 (kafa eğitimi, base donuk)")

h_kafa = model.fit(
    train_ds_ham, validation_data=val_ds_ham, epochs=EPOCHS_HEAD,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                              patience=3, restore_best_weights=True)],
)

# %% 7) AŞAMA 1-2-3 — KADEMELİ açma + AZALAN öğrenme oranı + DAHA SABIRLI early stopping
gecmisler = [h_kafa]
sonraki_baslangic_epoch = h_kafa.epoch[-1] + 1

for i, (oran, ust_sinir_epoch) in enumerate(zip(UNFREEZE_ASAMALARI, ASAMA_EPOCH_SAYILARI), start=1):
    base.trainable = True
    n_freeze = int(len(base.layers) * (1 - oran))
    for layer in base.layers[:n_freeze]:
        layer.trainable = False
    print(f"\n--- Aşama {i}: son %{int(oran * 100)} açık "
          f"({len(base.layers) - n_freeze}/{len(base.layers)} katman) ---")

    # Her aşamada YENİDEN derleniyor (Adam optimizer durumu sıfırlanır) ; yeniden
    # derlenmezse trainable değişiklikleri geçerli olmaz.
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    parametre_ozeti(model, f"Aşama {i} (son %{int(oran * 100)} açık)")

    azalan_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1,
    )
    erken_durdurma = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=FINE_TUNE_PATIENCE, restore_best_weights=True,
    )

    hedef_epoch = sonraki_baslangic_epoch + ust_sinir_epoch
    h = model.fit(
        train_ds_ham, validation_data=val_ds_ham,
        initial_epoch=sonraki_baslangic_epoch, epochs=hedef_epoch,
        callbacks=[azalan_lr, erken_durdurma],
    )
    gecmisler.append(h)
    sonraki_baslangic_epoch = h.epoch[-1] + 1

# %% 8) Değerlendirme — SADECE bağımsız test setinde (öncekiyle BİREBİR AYNI protokol)
y_prob = model.predict(test_ds_ham, verbose=0)
y_pred = y_prob.argmax(axis=1)

dogruluk = accuracy_score(y_true_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_test, y_pred, average="macro", zero_division=0,
)
try:
    macro_auc = float(roc_auc_score(y_true_test, y_prob, multi_class="ovr", average="macro"))
except ValueError:
    macro_auc = float("nan")
cm = confusion_matrix(y_true_test, y_pred)

sureler_ms = []
for i, (x_batch, _) in enumerate(test_ds_ham):
    t0 = time.perf_counter()
    model.predict(x_batch, verbose=0)
    t1 = time.perf_counter()
    if i > 0:
        sureler_ms.append((t1 - t0) * 1000)
ort_ms = float(np.mean(sureler_ms))

print(f"\nEfficientNetB0 (gelişmiş) — test doğruluğu: {dogruluk:.4f}, macro F1: {f1:.4f}, "
      f"macro AUC: {macro_auc:.4f}, ort. inference: {ort_ms:.1f} ms/görüntü")
print("Önceki (temel tarif) sonucu hatırlatma: doğruluk=0.9468, macro_f1=0.940, macro_auc=0.9958")

model_dosya = f"{OUT_DIR}/model_EfficientNetB0_gelismis.keras"
model.save(model_dosya)
model_boyutu_mb = round(os.path.getsize(model_dosya) / (1024 * 1024), 2)

with open(f"{OUT_DIR}/karsilastirma_gelismis.csv", "w", newline="", encoding="utf-8") as f:
    yazici = csv.DictWriter(f, fieldnames=[
        "model", "dogruluk", "macro_precision", "macro_recall", "macro_f1",
        "macro_auc", "model_boyutu_mb", "ort_inference_ms",
    ])
    yazici.writeheader()
    yazici.writerow({
        "model": "EfficientNetB0_gelismis",
        "dogruluk": round(dogruluk, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "macro_auc": round(macro_auc, 4),
        "model_boyutu_mb": model_boyutu_mb,
        "ort_inference_ms": round(ort_ms, 2),
    })

# %% 9) Grafikler — confusion matrix + 4 aşamayı birleştiren tek öğrenme eğrisi
plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — EfficientNetB0 (gelişmiş, bağımsız test seti)")
plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=7)
plt.yticks(range(len(class_names)), class_names, fontsize=7)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_EfficientNetB0_gelismis.png", dpi=120)
plt.close()

acc = sum((h.history["accuracy"] for h in gecmisler), [])
vacc = sum((h.history["val_accuracy"] for h in gecmisler), [])
kayip = sum((h.history["loss"] for h in gecmisler), [])
vkayip = sum((h.history["val_loss"] for h in gecmisler), [])
sinirlar = []
toplam = 0
for h in gecmisler[:-1]:
    toplam += len(h.history["accuracy"])
    sinirlar.append(toplam - 0.5)

sekil, (sol, sag) = plt.subplots(1, 2, figsize=(13, 4.4))
sol.plot(acc, label="eğitim"); sol.plot(vacc, label="doğrulama")
for s in sinirlar:
    sol.axvline(s, color="gray", linestyle="--")
sol.set_xlabel("epoch"); sol.set_ylabel("doğruluk"); sol.legend()
sol.set_title("EfficientNetB0 (gelişmiş) — Doğruluk (kesikli çizgiler: yeni aşama)")

sag.plot(kayip, label="eğitim"); sag.plot(vkayip, label="doğrulama")
for s in sinirlar:
    sag.axvline(s, color="gray", linestyle="--")
sag.set_xlabel("epoch"); sag.set_ylabel("kayıp (loss)"); sag.legend()
sag.set_title("EfficientNetB0 (gelişmiş) — Kayıp")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ogrenme_egrisi_EfficientNetB0_gelismis.png", dpi=120)
plt.close()

# %% 9b) ÖNCEKİ tarif vs GELİŞMİŞ tarif — karşılaştırma bar chart (asıl istenen görsel bu)
yeni_sonuc = {
    "dogruluk": dogruluk, "macro_precision": precision, "macro_recall": recall,
    "macro_f1": f1, "macro_auc": macro_auc,
}
metrikler = ["dogruluk", "macro_precision", "macro_recall", "macro_f1", "macro_auc"]
etiketler = ["Doğruluk", "Precision", "Recall", "Macro F1", "Macro AUC"]
eski_degerler = [ESKI_SONUC[m] for m in metrikler]
yeni_degerler = [yeni_sonuc[m] for m in metrikler]

x = np.arange(len(metrikler))
genislik = 0.35
plt.figure(figsize=(9, 5))
b1 = plt.bar(x - genislik / 2, eski_degerler, genislik, label="Önceki tarif (son %25, sabit LR)", color="#9e9e9e")
b2 = plt.bar(x + genislik / 2, yeni_degerler, genislik, label="Gelişmiş tarif (kademeli %15→%40, azalan LR)", color="#346cb0")
for bars in (b1, b2):
    for bar in bars:
        yukseklik = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yukseklik + 0.005, f"{yukseklik:.3f}",
                  ha="center", fontsize=8)
plt.xticks(x, etiketler)
plt.ylabel("skor (bağımsız test seti)")
plt.ylim(0, 1.08)
plt.title("EfficientNetB0 — Önceki Fine-Tuning Tarifi vs Gelişmiş Tarif")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/efficientnetb0_onceki_vs_gelismis.png", dpi=120)
plt.close()
print(f"\nefficientnetb0_onceki_vs_gelismis.png kaydedildi. "
      f"Doğruluk farkı: {yeni_sonuc['dogruluk'] - ESKI_SONUC['dogruluk']:+.4f}")

# %% 10) Ziple ve indir
shutil.make_archive("/content/leadleaf_efficientnetb0_gelismis", "zip", OUT_DIR)
files.download("/content/leadleaf_efficientnetb0_gelismis.zip")
print("\nBİTTİ. leadleaf_efficientnetb0_gelismis.zip indi. İçinde:")
print("  - efficientnetb0_onceki_vs_gelismis.png  -> ÖNCEKİ/GELİŞMİŞ karşılaştırma bar chart")
print("  - ogrenme_egrisi_EfficientNetB0_gelismis.png, confusion_matrix_EfficientNetB0_gelismis.png")
print("  - model_EfficientNetB0_gelismis.keras, karsilastirma_gelismis.csv")
print("İçindeki karsilastirma_gelismis.csv'yi model/tubitak/model_comparison.csv'deki "
      "EfficientNetB0 satırıyla karşılaştır — hangisi daha iyiyse ONU model/model.keras yap.")
