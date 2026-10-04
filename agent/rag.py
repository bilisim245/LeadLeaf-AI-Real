"""
RAG sorgulama: tahmin edilen hastalık için bilgi tabanından doğrulanmış kaynak metni getirir.

İndeks yoksa (rag/build_index.py çalıştırılmamışsa) hata vermez; boş bağlam döner ve rapor
kaynak metin olmadan üretilir. İlk kullanımda indeksi otomatik oluşturmayı dener.
"""
from __future__ import annotations

import os

# transformers, TensorFlow de kuruluysa (inference/app.py icin var) otomatik TF
# entegrasyonunu da yuklemeye calisip Keras 3 ile catisiyor. Sadece PyTorch
# kullanmasini zorluyoruz (sentence-transformers zaten PyTorch tabanli).
os.environ.setdefault("USE_TF", "0")
# Model zaten rag/build_index.py ile indirilmis oluyor (bir kere). Her rapor
# uretiminde HuggingFace'e "guncel surum mu" diye internet kontrolu yapmaya
# calismasi bazi aglarda cok yavas/askida kalabiliyor - offline modu zorlayarak
# dogrudan yerel onbellekten yukletiyoruz.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "rag", "chroma_db")
COLLECTION = "hastalik_bilgi_tabani"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None
_collection = None
_hazir = False


def _yukle() -> bool:
    """Chroma koleksiyonunu + embedding modelini bir kere yükler. Başarısızsa False döner."""
    global _model, _collection, _hazir
    if _hazir:
        return True
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        if not os.path.isdir(DB_DIR):
            print(f"[RAG] Index bulunamadı ({DB_DIR}). Önce çalıştır: python rag/build_index.py")
            return False

        # anonymized_telemetry=False: Chroma'nın varsayılan telemetri isteği bazı ağlarda
        # (internete kısıtlı erişim) uzun süre asılı kalabiliyor — kapatıyoruz.
        client = chromadb.PersistentClient(
            path=DB_DIR, settings=chromadb.Settings(anonymized_telemetry=False)
        )
        _collection = client.get_collection(COLLECTION)
        _model = SentenceTransformer(EMBED_MODEL)
        _hazir = True
        return True
    except Exception as e:
        print(f"[RAG] Yüklenemedi ({e}); RAG'siz devam ediliyor.")
        return False


def _temel_parca_mi(parca: str) -> bool:
    """Parça, her raporda bulunması gereken bölümlerden biri mi (belirtiler, önlemler, bakım)?"""
    baslik = parca.split("\n", 1)[0].lower()
    return baslik.startswith("## belirtiler") or "önlem" in baslik or baslik.startswith(("## yönetim", "## ne yapılmalı"))


def retrieve_context(hastalik: str, ek_sorgu: str = "", k: int = 2) -> str:
    """Verilen hastalık sınıfı için doğrulanmış kaynak metin parçalarını getirir.

    Öncelik: doğrudan sınıf filtresiyle o hastalığın kendi dosyasından en alakalı
    parçaları getirmek (küçük, iyi organize bilgi tabanında en güvenilir yöntem).
    `ek_sorgu` verilirse (ör. çiftçinin serbest metin açıklaması) semantik aramaya dahil edilir.
    """
    # "Tanımsız belirti" / "desteklenmeyen bitki" bilgi tabanında bir hastalık değil: arama yapılırsa
    # alakasız parçalar (ör. "sağlıklı soya") gelir ve LLM "yaprak sağlıklı olabilir" diye yanıltır.
    if hastalik.endswith("___Tanimsiz") or hastalik == "Desteklenmeyen_bitki":
        return ""
    if not _yukle():
        return ""

    # Arama zaten hastalığa göre filtrelendiği için sorguda sınıf adı kullanılmaz (İngilizce sınıf
    # adı sorguya baskın çıkıp çiftçinin açıklamasını etkisiz bırakıyordu).
    sorgu = ek_sorgu.strip() or "hastalığın nedeni ve yayılmasını kolaylaştıran koşullar"
    embedding = _model.encode([sorgu]).tolist()

    try:
        hepsi = _collection.get(where={"sinif": hastalik}, include=["documents"])["documents"]
        # Belirtiler ve önlemler her raporda gerekli: anlamsal sıralamaya bırakılmaz, her zaman eklenir.
        temel = [d for d in hepsi if _temel_parca_mi(d)]
        sonuc = _collection.query(query_embeddings=embedding, n_results=min(len(hepsi), len(temel) + 1),
                                  where={"sinif": hastalik}) if hepsi else None
        # Ek olarak çiftçinin açıklamasına (yoksa neden/koşul sorusuna) en yakın 1 parça
        ek = next((d for d in (sonuc or {}).get("documents", [[]])[0] if d not in temel), None)
        parcalar = temel + ([ek] if ek else [])
    except Exception:
        parcalar = []

    if not parcalar:
        # sınıf filtresiyle bulunamadıysa (ör. sınıf adı knowledge/'de yok), filtre olmadan dene
        try:
            sonuc = _collection.query(query_embeddings=embedding, n_results=k)
            parcalar = sonuc.get("documents", [[]])[0]
        except Exception:
            return ""

    return "\n\n---\n\n".join(parcalar)


if __name__ == "__main__":
    ctx = retrieve_context("Tomato___Early_blight")
    print(ctx or "(boş — index oluşturulmamış olabilir, `python rag/build_index.py` çalıştır)")
