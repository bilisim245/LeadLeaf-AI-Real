"""
38 sınıflık üretim modelini (model/model.keras) BAĞIMSIZ TEST KÜMESİNDE yerelde yeniden
değerlendirir.

Neden: Colab'da ölçülen %99,02'yi bağımsız olarak doğrulamak ve sunum panosu (ui/sunum.py)
için görsel bazında sonuçları (her test görselinin gerçek sınıfı + 38 sınıf olasılığı)
kaydetmek — karışıklık matrisi, sınıf bazında metrikler, yanlış bilinen görseller buradan çizilir.

Test kümesi = model/model_38sinif/split_manifest.json'daki "test" listesi (eğitimde hiç
kullanılmamış 8.146 görsel). Görseller data/plantvillage/raw/color altında olmalı
(GitHub spMohanty/PlantVillage-Dataset, raw/color — Colab'daki Kaggle kopyasıyla aynı dosyalar).

Çalıştırma:  .venv\\Scripts\\python notebooks\\04_yerel_test_degerlendirme.py
Çıktı:       model/model_38sinif/test_sonuclari.npz
"""
import json
import os
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
from tensorflow.keras.applications import efficientnet  # noqa: E402
from tensorflow.keras.utils import img_to_array, load_img  # noqa: E402

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERI = os.path.join(KOK, "data", "plantvillage", "raw", "color")
MANIFEST = os.path.join(KOK, "model", "model_38sinif", "split_manifest.json")
MODEL = os.path.join(KOK, "model", "model.keras")
CIKTI = os.path.join(KOK, "model", "model_38sinif", "test_sonuclari.npz")
IMG, BATCH = 224, 64

with open(os.path.join(KOK, "model", "class_names.json"), encoding="utf-8") as f:
    siniflar = json.load(f)
with open(MANIFEST, encoding="utf-8") as f:
    test = json.load(f)["dosyalar"]["test"]

ornekler = [(sinif, dosya) for sinif in siniflar for dosya in test.get(sinif, [])]
print(f"Test görseli: {len(ornekler)} · sınıf: {len(siniflar)}")

model = keras.models.load_model(MODEL, compile=False)
y_true = np.array([siniflar.index(s) for s, _ in ornekler], dtype=np.int16)
y_prob = np.zeros((len(ornekler), len(siniflar)), dtype=np.float32)

t0 = time.time()
for i in range(0, len(ornekler), BATCH):
    grup = ornekler[i:i + BATCH]
    # image_dataset_from_directory ile AYNI: 224x224, bilinear; preprocess EfficientNet'te geçişli
    x = np.stack([
        img_to_array(load_img(os.path.join(VERI, s, d), target_size=(IMG, IMG), interpolation="bilinear"))
        for s, d in grup
    ])
    y_prob[i:i + len(grup)] = model.predict(efficientnet.preprocess_input(x), verbose=0)
    if (i // BATCH) % 10 == 0:
        dogru = (y_prob[:i + len(grup)].argmax(1) == y_true[:i + len(grup)]).mean()
        print(f"  {i + len(grup):>5}/{len(ornekler)}  ara doğruluk %{dogru * 100:.2f}  "
              f"({time.time() - t0:.0f} sn)", flush=True)

dogruluk = (y_prob.argmax(1) == y_true).mean()
print(f"\nTEST DOĞRULUĞU (yerel): %{dogruluk * 100:.2f} — süre {time.time() - t0:.0f} sn")
np.savez_compressed(
    CIKTI, y_true=y_true, y_prob=y_prob.astype(np.float16),
    siniflar=np.array(siniflar), dosyalar=np.array([f"{s}/{d}" for s, d in ornekler]),
)
print("Kaydedildi:", CIKTI)
