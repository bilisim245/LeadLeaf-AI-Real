"""
RAG indeksi oluşturma: hastalık bilgi tabanı (agent/knowledge/*.md) Chroma vektör veritabanına
aktarılır. Rapor, modelin genel bilgisi yerine projeye özel, doğrulanmış yazılı kaynağa
dayanılarak üretilir ve hangi kaynağın kullanıldığı gösterilebilir.

Çalıştırma:
    python rag/build_index.py

Çıktı: rag/chroma_db/ (her ortamda yeniden oluşturulur, depoya eklenmez).
"""
from __future__ import annotations

import os

# transformers, TensorFlow de kuruluysa (inference/app.py icin var) otomatik TF
# entegrasyonunu da yuklemeye calisip Keras 3 ile catisiyor. Sadece PyTorch
# kullanmasini zorluyoruz (sentence-transformers zaten PyTorch tabanli).
os.environ.setdefault("USE_TF", "0")

import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "agent", "knowledge")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION = "hastalik_bilgi_tabani"

# Türkçe metin için çok dilli model — İngilizce-odaklı modellere göre çok daha iyi embedding kalitesi
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def build():
    dosyalar = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".md")]
    assert dosyalar, f"Bilgi dosyası bulunamadı: {KNOWLEDGE_DIR}"

    print(f"[1/3] {len(dosyalar)} bilgi dosyası bulundu: {dosyalar}")
    print(f"[2/3] Embedding modeli yükleniyor: {EMBED_MODEL} (ilk seferde indirilir, ~470MB)")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(
        path=DB_DIR, settings=chromadb.Settings(anonymized_telemetry=False)
    )
    # Her build'de temiz başla (küçük bilgi tabanı — yeniden oluşturmak ucuz)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    ids, docs, metas = [], [], []
    for fname in dosyalar:
        sinif = fname.replace(".md", "")
        with open(os.path.join(KNOWLEDGE_DIR, fname), "r", encoding="utf-8") as f:
            metin = f.read()
        # Basit bölümleme: "## " başlıklarına göre chunk'lara ayır (her chunk kendi
        # bağlamıyla anlamlı kalsın diye başlığı chunk'ın başına da ekliyoruz)
        parcalar = metin.split("\n## ")
        for i, parca in enumerate(parcalar):
            if i > 0:
                parca = "## " + parca
            parca = parca.strip()
            if len(parca) < 20:
                continue
            ids.append(f"{sinif}__{i}")
            docs.append(parca)
            metas.append({"sinif": sinif, "dosya": fname})

    print(f"[3/3] {len(docs)} parça embed edilip indexleniyor...")
    embeddings = model.encode(docs, show_progress_bar=False).tolist()
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    print(f"OK — index oluşturuldu: {DB_DIR} ({len(docs)} parça, {len(dosyalar)} sınıf)")


if __name__ == "__main__":
    build()
