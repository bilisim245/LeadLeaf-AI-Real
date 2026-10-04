# =====================================================================
#  EfficientNetB0: 38 Sınıf (Üretim Modeli)
#  Google Colab (GPU) üzerinde çalıştırılır. Kademeli fine-tuning tarifi
#  PlantVillage'ın 14 bitki ve 38 sınıfının tamamına uygulanır; üretimde
#  kullanılan model.keras bu betikle eğitilmiştir.
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
SEED = 42
EPOCHS_HEAD = 8

FINE_TUNE_PATIENCE = 5
REDUCE_LR_PATIENCE = 2
UNFREEZE_ASAMALARI = [0.15, 0.30, 0.40]
ASAMA_EPOCH_SAYILARI = [6, 6, 8]

OUT_DIR = "/content/outputs_38sinif"
os.makedirs(OUT_DIR, exist_ok=True)

ORANLAR = {"train": 0.70, "valid": 0.15, "test": 0.15}

# None = veri setindeki TÜM 38 sınıf (bu script'in amacı bu — bkz. üstteki not)
SELECTED_CLASSES = None

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# %% 2) Veri yolu + sınıf listesi
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

mevcut_siniflar = sorted(os.listdir(RAW_DIR))
print(f"Veri setindeki toplam sinif sayisi: {len(mevcut_siniflar)}")
for c in mevcut_siniflar:
    print(" -", c)

# %% 3) TEK SEFERLİK stratified train/valid/test bölmesi (%70/%15/%15) + manifest
SPLIT_DIR = "/content/_split_38"


def _stratified_uc_yonlu_split(kaynak_dir, hedef_kok, siniflar, oranlar, seed):
    hedefler = {k: os.path.join(hedef_kok, k) for k in ("train", "valid", "test")}
    if all(os.path.isdir(d) for d in hedefler.values()):
        print("Split klasörleri zaten var, yeniden oluşturulmuyor:", hedef_kok)
        return hedefler, None

    rng = random.Random(seed)
    kaynak_siniflar = siniflar if siniflar else sorted(os.listdir(kaynak_dir))
    manifest_dosyalar = {"train": {}, "valid": {}, "test": {}}

    for cls in kaynak_siniflar:
        src = os.path.join(kaynak_dir, cls)
        if not os.path.isdir(src):
            print(f"UYARI: sinif klasoru yok, atlaniyor: {cls}")
            continue
        dosyalar = os.listdir(src)
        rng.shuffle(dosyalar)

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


SPLIT_DIRS, MANIFEST = _stratified_uc_yonlu_split(RAW_DIR, SPLIT_DIR, SELECTED_CLASSES, ORANLAR, SEED)
if MANIFEST is not None:
    with open(f"{OUT_DIR}/split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, ensure_ascii=False, indent=2)
    print("split_manifest.json kaydedildi.")

TRAIN_DIR, VALID_DIR, TEST_DIR = SPLIT_DIRS["train"], SPLIT_DIRS["valid"], SPLIT_DIRS["test"]

# %% 4) tf.data setleri
def veri_seti_yukle(dizin, batch_size, shuffle):
    return keras.utils.image_dataset_from_directory(
        dizin, image_size=(IMG, IMG), batch_size=batch_size, seed=SEED, shuffle=shuffle,
    )


train_ds_ham = veri_seti_yukle(TRAIN_DIR, BATCH, shuffle=True)
val_ds_ham = veri_seti_yukle(VALID_DIR, BATCH, shuffle=False)
test_ds_ham = veri_seti_yukle(TEST_DIR, 1, shuffle=False)

# class_names, .prefetch() UYGULANMADAN ÖNCE okunmalı (02'deki bilinen hata burada da geçerli)
class_names = train_ds_ham.class_names
print(len(class_names), "sinif:", class_names)
with open(f"{OUT_DIR}/class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

train_ds_ham = train_ds_ham.prefetch(tf.data.AUTOTUNE)
val_ds_ham = val_ds_ham.prefetch(tf.data.AUTOTUNE)
y_true_test = np.concatenate([y.numpy() for _, y in test_ds_ham])

# %% 5) Model kurma (01/02 ile AYNI iskelet)
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
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)


def parametre_ozeti(model, etiket):
    egitilebilir = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    donuk = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    print(f"  {etiket:45s} eğitilebilir: {egitilebilir:>10,}   donuk: {donuk:>10,}")


# %% 6) AŞAMA 0 — kafa eğitimi (base tamamen donuk)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
parametre_ozeti(model, "Aşama 0 (kafa eğitimi, base donuk)")

h_kafa = model.fit(
    train_ds_ham, validation_data=val_ds_ham, epochs=EPOCHS_HEAD,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                              patience=3, restore_best_weights=True)],
)

# %% 7) AŞAMA 1-2-3 — kademeli açma + azalan öğrenme oranı + sabırlı early stopping
gecmisler = [h_kafa]
sonraki_baslangic_epoch = h_kafa.epoch[-1] + 1

for i, (oran, ust_sinir_epoch) in enumerate(zip(UNFREEZE_ASAMALARI, ASAMA_EPOCH_SAYILARI), start=1):
    base.trainable = True
    n_freeze = int(len(base.layers) * (1 - oran))
    for layer in base.layers[:n_freeze]:
        layer.trainable = False
    print(f"\n--- Aşama {i}: son %{int(oran * 100)} açık "
          f"({len(base.layers) - n_freeze}/{len(base.layers)} katman) ---")

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

# %% 8) Değerlendirme — SADECE bağımsız test setinde
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

print(f"\nEfficientNetB0 (38 sınıf) — test doğruluğu: {dogruluk:.4f}, macro F1: {f1:.4f}, "
      f"macro AUC: {macro_auc:.4f}, ort. inference: {ort_ms:.1f} ms/görüntü")

model_dosya = f"{OUT_DIR}/model.keras"
model.save(model_dosya)
model_boyutu_mb = round(os.path.getsize(model_dosya) / (1024 * 1024), 2)

with open(f"{OUT_DIR}/model_meta.json", "w", encoding="utf-8") as f:
    json.dump({"mimari": "EfficientNetB0"}, f, ensure_ascii=False, indent=2)

with open(f"{OUT_DIR}/model_comparison.csv", "w", newline="", encoding="utf-8") as f:
    yazici = csv.DictWriter(f, fieldnames=[
        "model", "dogruluk", "macro_precision", "macro_recall", "macro_f1",
        "macro_auc", "model_boyutu_mb", "ort_inference_ms",
    ])
    yazici.writeheader()
    yazici.writerow({
        "model": "EfficientNetB0_38sinif",
        "dogruluk": round(dogruluk, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4),
        "macro_auc": round(macro_auc, 4),
        "model_boyutu_mb": model_boyutu_mb,
        "ort_inference_ms": round(ort_ms, 2),
    })

# %% 9) Grafikler
plt.figure(figsize=(14, 13))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — EfficientNetB0 (38 sınıf, bağımsız test seti)")
plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=5)
plt.yticks(range(len(class_names)), class_names, fontsize=5)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_EfficientNetB0_38sinif.png", dpi=120)
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
sol.set_title("EfficientNetB0 (38 sınıf) — Doğruluk (kesikli çizgiler: yeni aşama)")

sag.plot(kayip, label="eğitim"); sag.plot(vkayip, label="doğrulama")
for s in sinirlar:
    sag.axvline(s, color="gray", linestyle="--")
sag.set_xlabel("epoch"); sag.set_ylabel("kayıp (loss)"); sag.legend()
sag.set_title("EfficientNetB0 (38 sınıf) — Kayıp")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ogrenme_egrisi_EfficientNetB0_38sinif.png", dpi=120)
plt.close()

# %% 10) Demo görselleri (Streamlit için) — test setinden örnek
os.makedirs(f"{OUT_DIR}/demo_images", exist_ok=True)
for cls in random.sample(class_names, min(8, len(class_names))):
    dosyalar = glob.glob(os.path.join(TEST_DIR, cls, "*"))
    if dosyalar:
        src = random.choice(dosyalar)
        shutil.copy(src, f"{OUT_DIR}/demo_images/{cls}.jpg")
print("demo_images/ hazir.")

# %% 11) Ziple ve indir
shutil.make_archive("/content/leadleaf_38sinif", "zip", OUT_DIR)
files.download("/content/leadleaf_38sinif.zip")
print(f"""
BİTTİ. leadleaf_38sinif.zip indi -> aç, içindekileri model/ klasörüne koy
(ÜRETİM — n8n/Telegram akışı ve Streamlit bunu kullanır):

  - model.keras         (38 sınıflı EfficientNetB0)
  - model_meta.json      -> {{"mimari": "EfficientNetB0"}}
  - class_names.json      (artık 38 sınıf var)
  - demo_images/
  - split_manifest.json, model_comparison.csv, confusion_matrix, öğrenme eğrisi
    -> bunları model/tubitak/ klasörüne de kopyalayabilirsin (referans için)

SONRAKI ADIM (Ben tarafı): inference/app.py'deki TR_ADLAR sözlüğü ve
agent/knowledge/ klasöründeki RAG dosyaları yeni sınıflara göre
GENİŞLETİLMELİ — model 38 sınıfı tanısa da, karşılığı olmayan sınıflar
için sadece Türkçe ad/RAG bağlamı eksik kalır (sistem çökmez, ama o
sınıflar için "genel bilgi" moduna düşer).
""")
