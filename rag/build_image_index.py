"""
Görsel RAG indeksi: referans yaprak görsellerinin öznitelik vektörlerini çıkarır.

Ayrı bir görsel embedding modeli kullanılmaz. Eğitilmiş sınıflandırıcının son katmandan bir
önceki katmanı (GlobalAveragePooling2D, 1280 boyutlu vektör) embedding olarak kullanılır.
Böylece benzerlik, modelin hastalıkları ayırt ederken öğrendiği özelliklere göre hesaplanır.

Çalıştırma:
    python rag/build_image_index.py

Çıktı: rag/image_embeddings.npz (embeddings, labels, dosya_yollari). Dosya varsa model servisi
/predict yanıtına "benzer_gorseller" alanını (en yakın referans görseller ve benzerlik oranı) ekler.
Betik, model/model.keras ve örnek görseller olmadan çalışmaz.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "./model/model.keras")
DEMO_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "demo_images")
OUT_PATH = os.path.join(os.path.dirname(__file__), "image_embeddings.npz")
IMG_SIZE = 224


def build():
    assert os.path.exists(MODEL_PATH), (
        f"Model bulunamadı: {MODEL_PATH}. Önce Colab'da eğitip model/ klasörüne koy "
        "(bkz. notebooks/01_train_model_colab.py)."
    )
    referans_gorseller = glob.glob(os.path.join(DEMO_IMAGES_DIR, "*"))
    assert referans_gorseller, (
        f"Referans görsel bulunamadı: {DEMO_IMAGES_DIR}. Colab çıktısındaki "
        "demo_images/ klasörünü model/demo_images/ altına kopyala."
    )

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from PIL import Image

    print(f"Model yükleniyor: {MODEL_PATH}")
    full_model = keras.models.load_model(MODEL_PATH)

    # Son Dense (softmax) katmanından ÖNCEKİ katmanı (GlobalAveragePooling2D
    # çıktısı) embedding olarak kullanan yeni bir model kur.
    embedding_layer = full_model.layers[-2].output  # Dropout'tan önceki GAP çıktısı
    embedding_model = keras.Model(inputs=full_model.input, outputs=embedding_layer)

    embeddings, labels, dosyalar = [], [], []
    for yol in referans_gorseller:
        sinif = os.path.splitext(os.path.basename(yol))[0]  # demo_images/{sinif}.jpg
        img = Image.open(yol).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = preprocess_input(np.array(img).astype("float32"))
        arr = np.expand_dims(arr, axis=0)
        emb = embedding_model.predict(arr, verbose=0)[0]
        embeddings.append(emb)
        labels.append(sinif)
        dosyalar.append(os.path.basename(yol))
        print(f"  + {sinif}: {yol}")

    np.savez(OUT_PATH, embeddings=np.array(embeddings), labels=np.array(labels),
             dosyalar=np.array(dosyalar))
    print(f"\nOK — {len(embeddings)} referans görsel embed edildi: {OUT_PATH}")


if __name__ == "__main__":
    build()
