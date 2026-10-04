"""
Model servisi (FastAPI)

Görev: CNN modelini HTTP ile sunmak. Proje mimarisinde n8n (local veya n8n Cloud)
bu servise HTTP Request node'uyla görsel gönderir, sınıflandırma + güven döner.

DEMO MODU: model/model.keras bulunamazsa servis çökmez; rastgele bir tahmin üretip
demo_mode=true ile işaretler. Model dosyası yerine konup servis yeniden başlatılınca
gerçek moda geçer.

Çalıştırma:
    .venv\\Scripts\\python.exe -m uvicorn inference.app:app --reload --port 8000

Test:
    http://localhost:8000/docs  (Swagger UI'dan görsel yükleyip dene)
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.image_rag import find_similar, index_var_mi
from agent.rag import retrieve_context
from agent.pdf_rapor import rapor_pdf_olustur

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./model/model.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "./model/class_names.json")
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "./model/model_meta.json")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
IMG_SIZE = 224

# Bitki filtresi (kapalı sınıf sorunu): model tanımadığı bir hastalığı (ör. şeftali yaprak
# bükülmesi) bilinen bir sınıfa yüksek güvenle yakıştırabiliyor (%89 "domates geç yanıklığı").
# Çiftçi fotoğraf açıklamasına bitkiyi yazarsa tahmin SADECE o bitkinin sınıflarıyla
# sınırlanır. Olasılıklar yeniden normalize EDİLMEZ — o bitkiye düşen ham olasılık düşükse
# bu, "görüntü bu bitkinin bilinen sınıflarına benzemiyor" demektir ve uzmana yönlendirilir.
BITKI_ONEKLERI = {
    "domates": "Tomato", "patates": "Potato", "biber": "Pepper,_bell", "elma": "Apple",
    "seftali": "Peach", "kiraz": "Cherry_(including_sour)", "visne": "Cherry_(including_sour)",
    "uzum": "Grape", "misir": "Corn_(maize)", "cilek": "Strawberry", "portakal": "Orange",
    "ahududu": "Raspberry", "soya": "Soybean", "kabak": "Squash",
    "yaban mersini": "Blueberry", "yabanmersini": "Blueberry",
}
BITKI_GORUNEN_AD = {
    "seftali": "Şeftali", "visne": "Vişne", "uzum": "Üzüm", "misir": "Mısır", "cilek": "Çilek",
    "yabanmersini": "Yaban mersini",
}
BITKI_UYUM_ESIGI = 0.50  # o bitkinin sınıflarına düşen toplam olasılık bunun altındaysa uyumsuz

# Otomatik bitki tanıma: çiftçiden bitki adı İSTEMEMEK için fotoğraf önce Pl@ntNet'e
# (my.plantnet.org, ücretsiz plan 500 tanıma/gün) gönderilip tür bulunur. Anahtar yoksa ya da
# servis hata verirse sessizce atlanır — sistem filtresiz (eski) davranışa düşer.
PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY", "")
PLANTNET_MIN_SKOR = float(os.getenv("PLANTNET_MIN_SKOR", "0.20"))
# Bilimsel ad öneki -> BITKI_ONEKLERI'ndeki türkçe anahtar (en uzun önek önce eşleşir)
BILIMSEL_AD_BITKI = {
    "Solanum lycopersicum": "domates", "Solanum tuberosum": "patates",
    "Capsicum": "biber", "Malus": "elma", "Prunus persica": "seftali",
    "Prunus avium": "kiraz", "Prunus cerasus": "visne", "Vitis": "uzum",
    "Zea mays": "misir", "Fragaria": "cilek", "Citrus": "portakal",
    "Rubus idaeus": "ahududu", "Glycine max": "soya", "Cucurbita": "kabak",
    "Vaccinium": "yabanmersini",
}


def _plantnet_tani(raw: bytes) -> Optional[dict]:
    """Pl@ntNet ile türü bulur. Dönüş: {"tur", "yaygin_ad", "skor", "bitki_ad"} (bitki_ad
    desteklenmeyen türde None) ya da anahtar yok/hata/düşük skor durumunda None."""
    if not PLANTNET_API_KEY:
        return None
    try:
        import requests

        r = requests.post(
            "https://my-api.plantnet.org/v2/identify/all",
            params={"api-key": PLANTNET_API_KEY, "lang": "tr", "nb-results": 1},
            files=[("images", ("yaprak.jpg", raw, "image/jpeg"))],
            data={"organs": "leaf"},
            timeout=8,
        )
        if r.status_code == 404:  # Pl@ntNet: "Species not found" -> bitki tanınamadı
            return None
        r.raise_for_status()
        en_iyi = r.json()["results"][0]
    except Exception as e:
        print(f"[UYARI] Pl@ntNet tanıma başarısız, filtresiz devam: {e}")
        return None
    if en_iyi["score"] < PLANTNET_MIN_SKOR:
        return None
    tur = en_iyi["species"]["scientificNameWithoutAuthor"]
    bitki_ad = next(
        (BILIMSEL_AD_BITKI[k] for k in sorted(BILIMSEL_AD_BITKI, key=len, reverse=True) if tur.startswith(k)),
        None,
    )
    yaygin = en_iyi["species"].get("commonNames") or [tur]
    return {"tur": tur, "yaygin_ad": yaygin[0], "skor": round(en_iyi["score"] * 100, 1), "bitki_ad": bitki_ad}


def _bitki_bul(metin: str) -> tuple[Optional[str], Optional[str]]:
    """Serbest metinden (Telegram fotoğraf açıklaması) bitkiyi bulur -> (türkçe ad, sınıf öneki)."""
    # "İ".lower() -> "i" + birleşik nokta (U+0307); önce onu at, sonra Türkçe harfleri sadeleştir
    sade = metin.lower().replace("̇", "").translate(str.maketrans("şçğıöüâ", "scgioua"))
    # "bu mısır değil, domates" -> mısır reddedilmiş, domates alınır (olumsuzlanan ad atlanır)
    for ad in sorted(BITKI_ONEKLERI, key=len, reverse=True):
        # \b: kelime başı olmalı ("akkiraz" kiraz sayılmasın)
        for m in re.finditer(r"\b" + re.escape(ad), sade):
            sonra = sade[m.end():]
            if re.match(r"\w*\s*degil", sonra):
                continue
            # Yer adları bitki sanılmasın: "Elmalı", "Kirazlı", "Üzümlü" (-lı/-lu; ama "elmalığım"
            # bahçe anlamında -lık/-lığ olduğu için bitki sayılır), "Elmadağ", ayrıca "soyadım"
            if re.match(r"l[iu](?![kg])", sonra) or sonra.startswith("dag") or (ad == "soya" and sonra.startswith("d")):
                continue
            return ad, BITKI_ONEKLERI[ad]
    return None, None

# Üç modelin karşılaştırması (notebooks/01_train_model_colab.py çıktısı):
# üretim /predict'ten ayrı: n8n/Telegram akışını etkilemez, sadece Streamlit'in
# "Model Karşılaştırma" sekmesi bunu kullanır.
TUBITAK_MODEL_DIR = os.getenv("TUBITAK_MODEL_DIR", "./model/tubitak")
TUBITAK_MODEL_DEFS = [
    {"ad": "MobileNetV2", "dosya": "model_MobileNetV2.keras"},
    {"ad": "MobileNetV3Small", "dosya": "model_MobileNetV3Small.keras"},
    {"ad": "EfficientNetB0", "dosya": "model_EfficientNetB0.keras"},
]

# Sınıf adı -> sade Türkçe adı (rapor ve arayüz için)
# NOT: Aşağıdaki anahtarlar PlantVillage veri setinin
# BİLİNEN standart klasör adlarıdır. `notebooks/03_efficientnetb0_38_sinif.py` çalıştırılıp
# class_names.json indiğinde, BU SÖZLÜĞÜN ANAHTARLARI o dosyayla birebir karşılaştırılıp
# (özellikle boşluk/virgül/parantez içeren adlarda - örn. "Pepper,_bell", "Spider_mites
# Two-spotted_spider_mite", "Corn_(maize)___Common_rust_") gerekirse düzeltilmelidir —
# TR_ADLAR.get(sinif, sinif) fallback'i sayesinde eşleşmeyen bir anahtar sistemi ÇÖKERTMEZ,
# sadece o sınıf için ham İngilizce adı gösterir (Türkçe ad eksik kalır).
TR_ADLAR = {
    "Tomato___healthy": "Sağlıklı",
    "Tomato___Early_blight": "Erken Yanıklık (Early Blight)",
    "Tomato___Late_blight": "Geç Yanıklık (Late Blight)",
    "Tomato___Bacterial_spot": "Bakteriyel Leke (Bacterial Spot)",
    "Tomato___Septoria_leaf_spot": "Septoria Yaprak Lekesi",
    "Tomato___Leaf_Mold": "Yaprak Küfü (Leaf Mold)",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Kırmızı Örümcek (İki Noktalı)",
    "Tomato___Target_Spot": "Hedef Leke (Target Spot)",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Sarı Yaprak Kıvırcıklığı Virüsü (TYLCV)",
    "Tomato___Tomato_mosaic_virus": "Mozaik Virüsü (ToMV)",
    "Pepper,_bell___Bacterial_spot": "Bakteriyel Leke (Biber)",
    "Pepper,_bell___healthy": "Sağlıklı (Biber)",
    "Potato___Early_blight": "Erken Yanıklık (Patates)",
    "Potato___Late_blight": "Geç Yanıklık (Patates)",
    "Potato___healthy": "Sağlıklı (Patates)",
    "Apple___Apple_scab": "Elma Karalekesi (Apple Scab)",
    "Apple___Black_rot": "Kara Çürüklük (Black Rot)",
    "Apple___Cedar_apple_rust": "Elma Pası (Cedar Apple Rust)",
    "Apple___healthy": "Sağlıklı (Elma)",
    "Blueberry___healthy": "Sağlıklı (Yaban Mersini)",
    "Cherry_(including_sour)___Powdery_mildew": "Külleme (Kiraz)",
    "Cherry_(including_sour)___healthy": "Sağlıklı (Kiraz)",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gri Yaprak Lekesi (Mısır)",
    "Corn_(maize)___Common_rust_": "Yaygın Pas Hastalığı (Mısır)",
    "Corn_(maize)___Northern_Leaf_Blight": "Kuzey Yaprak Yanıklığı (Mısır)",
    "Corn_(maize)___healthy": "Sağlıklı (Mısır)",
    "Grape___Black_rot": "Kara Çürüklük (Üzüm)",
    "Grape___Esca_(Black_Measles)": "Esca (Kara Kızamık)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Yaprak Yanıklığı (Üzüm)",
    "Grape___healthy": "Sağlıklı (Üzüm)",
    "Orange___Haunglongbing_(Citrus_greening)": "Turunçgil Yeşillenmesi (HLB)",
    "Peach___Bacterial_spot": "Bakteriyel Leke (Şeftali)",
    "Peach___healthy": "Sağlıklı (Şeftali)",
    "Raspberry___healthy": "Sağlıklı (Ahududu)",
    "Soybean___healthy": "Sağlıklı (Soya)",
    "Squash___Powdery_mildew": "Külleme (Kabak)",
    "Strawberry___Leaf_scorch": "Yaprak Yanıklığı (Çilek)",
    "Strawberry___healthy": "Sağlıklı (Çilek)",
}
DEMO_CLASS_NAMES = list(TR_ADLAR.keys())

app = FastAPI(title="LeadLeaf AI — Inference Servisi")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # n8n ve arayüz farklı adreslerden çağırabilsin
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_embedding_model = None  # gorsel RAG icin - modelin son katmandan onceki (GAP) ciktisi
_class_names: list[str] = []
DEMO_MODE = True
# model.keras'ın mimarisi model_meta.json'dan okunur; doğru preprocess_input seçilir.
# Dosya yoksa MobileNetV2 varsayılır.
_model_mimari = "MobileNetV2"

_tubitak_models: dict = {}  # ad -> yüklü keras.Model (sadece dosyası bulunanlar)
_tubitak_class_names: list[str] = []


def _strip_quantization_config(yol: str) -> None:
    """Colab'daki Keras, local'den daha yeni olursa .keras'ın config.json'una
    Dense (vb.) katmanlara local'in TANIMADIĞI bir 'quantization_config' alanı
    yazabiliyor ('Unrecognized keyword arguments' hatasıyla yüklemeyi
    çökertiyor). Bu alan sadece quantization-aware eğitim için var, bizde
    None/boş — kaldırmak model ağırlıklarını/davranışını DEĞİŞTİRMEZ, sadece
    eski Keras'in anlamadığı bir serileştirme farkını giderir."""
    import zipfile

    def _temizle(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for v in obj.values():
                _temizle(v)
        elif isinstance(obj, list):
            for item in obj:
                _temizle(item)

    with zipfile.ZipFile(yol, "r") as z:
        isimler = z.namelist()
        config = json.loads(z.read("config.json"))
        digerleri = {n: z.read(n) for n in isimler if n != "config.json"}

    _temizle(config)

    with zipfile.ZipFile(yol, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("config.json", json.dumps(config))
        for n, veri in digerleri.items():
            z.writestr(n, veri)


def _keras_uyumlu_yukle(tf_modul, yol: str):
    """tf.keras.models.load_model'i dener; Keras sürüm uyuşmazlığından
    (quantization_config vb.) patlarsa dosyayı bir kez onarıp tekrar dener."""
    try:
        return tf_modul.keras.models.load_model(yol)
    except TypeError as e:
        if "quantization_config" not in str(e) and "Unrecognized keyword" not in str(e):
            raise
        print(f"[UYARI] {yol}: Keras sürüm uyuşmazlığı tespit edildi, dosya onarılıp tekrar deneniyor...")
        _strip_quantization_config(yol)
        return tf_modul.keras.models.load_model(yol)


def _load_model_if_available() -> None:
    """Model dosyaları varsa yükler; yoksa (veya yükleme başarısız olursa)
    DEMO_MODE'da kalır (servis çökmez)."""
    global _model, _embedding_model, _class_names, DEMO_MODE, _model_mimari

    if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH)):
        print(f"[UYARI] Model bulunamadı ({MODEL_PATH}). DEMO MODU aktif — "
              f"Model dosyası model/ klasörüne konup servis yeniden başlatılmalıdır.")
        _class_names = DEMO_CLASS_NAMES
        DEMO_MODE = True
        return

    # TensorFlow'u sadece gerçekten gerekince import ediyoruz (demo modda hızlı açılış için)
    import tensorflow as tf

    try:
        _model = _keras_uyumlu_yukle(tf, MODEL_PATH)
    except Exception as e:
        print(f"[UYARI] Model yüklenemedi ({MODEL_PATH}): {e}. DEMO MODU aktif.")
        _class_names = DEMO_CLASS_NAMES
        DEMO_MODE = True
        return

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        _class_names = json.load(f)
    DEMO_MODE = False

    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
            _model_mimari = json.load(f).get("mimari", "MobileNetV2")
    print(f"[OK] Model yüklendi: {MODEL_PATH} ({len(_class_names)} sınıf, mimari={_model_mimari})")

    # Görsel RAG için: son Dense (softmax) katmanından ÖNCEKİ (GAP) çıktısını veren
    # ikinci bir model — ayrı bir CLIP modeli indirmeden, sınıflandırıcının kendi
    # öğrenilmiş temsilini yeniden kullanıyoruz (bkz. rag/build_image_index.py).
    if index_var_mi():
        try:
            _embedding_model = tf.keras.Model(
                inputs=_model.input, outputs=_model.layers[-2].output
            )
            print("[OK] Görsel RAG embedding modeli hazır.")
        except Exception as e:
            print(f"[UYARI] Görsel RAG embedding modeli kurulamadı: {e}")


def _load_tubitak_models_if_available() -> None:
    """3 karşılaştırma modelini (varsa) yükler. Kısmi eksik olabilir — eksik
    olan her model /predict_compare içinde tek başına DEMO MODU'na düşer,
    servis çökmez (bkz. inference/app.py'nin genel DEMO MODU felsefesi)."""
    global _tubitak_class_names

    class_names_path = os.path.join(TUBITAK_MODEL_DIR, "class_names.json")
    if not os.path.exists(class_names_path):
        _tubitak_class_names = DEMO_CLASS_NAMES
        print(f"[UYARI] Karşılaştırma modelleri bulunamadı ({TUBITAK_MODEL_DIR}). "
              f"/predict_compare tamamen DEMO MODU'nda çalışacak.")
        return

    import tensorflow as tf

    with open(class_names_path, "r", encoding="utf-8") as f:
        _tubitak_class_names = json.load(f)

    for tanim in TUBITAK_MODEL_DEFS:
        yol = os.path.join(TUBITAK_MODEL_DIR, tanim["dosya"])
        if os.path.exists(yol):
            try:
                _tubitak_models[tanim["ad"]] = _keras_uyumlu_yukle(tf, yol)
                print(f"[OK] Karşılaştırma modeli yüklendi: {tanim['ad']}")
            except Exception as e:
                print(f"[UYARI] Karşılaştırma modeli yüklenemedi ({tanim['ad']}): {e}")
        else:
            print(f"[UYARI] Karşılaştırma modeli eksik, bu model demo modda kalacak: {tanim['ad']} ({yol})")


@app.on_event("startup")
def _startup() -> None:
    _load_model_if_available()
    _load_tubitak_models_if_available()
    # RAG embedding modelini acilista yukle - yoksa ilk Telegram istegi ~17 sn bekliyor.
    retrieve_context("Tomato___healthy")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "num_classes": len(_class_names),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "tubitak_models_yuklu": sorted(_tubitak_models.keys()),
        "tubitak_models_toplam": len(TUBITAK_MODEL_DEFS),
    }


@app.get("/rag-context")
def rag_context(hastalik: str, ek_sorgu: str = "") -> dict:
    """n8n'in Claude'a göndermeden önce çağırdığı RAG bağlamı — agent/rag.py'deki
    Chroma indeksinden bu hastalık için doğrulanmış kaynak metin parçalarını getirir.
    Index yoksa veya sorgu başarısızsa boş metin döner (LLM promptu RAG'siz devam eder)."""
    return {"baglam": retrieve_context(hastalik, ek_sorgu)}


# Telegram'da "PDF ister misiniz?" butonu için: rapor kısa bir numarayla saklanır, numara butonun
# callback_data'sına konur (Telegram sınırı 64 bayt). Çiftçi butona basınca PDF bu numarayla üretilir.
RAPOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raporlar")


# Sohbette düzeltme: çiftçi fotoğraftan sonra sadece "domates" yazarsa, AYNI fotoğraf o bitkiyle
# yeniden değerlendirilir. Bunun için her sohbetin son fotoğrafının Telegram file_id'si saklanır
# (görselin kendisi değil — Telegram dosyayı kendi sunucusunda tutuyor, n8n file_id ile yeniden indirir).
SON_FOTO_YOLU = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "son_fotolar.json")
SON_FOTO_SURESI = 30 * 60  # saniye; daha eski fotoğraf için bitki adı sohbet olarak ele alınır
SON_FOTO_EN_COK_KELIME = 6  # "domates", "bu mısır değil domates" gibi kısa düzeltmeler; uzun soru sohbettir


def _son_fotolar() -> dict:
    try:
        with open(SON_FOTO_YOLU, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _son_foto_kaydet(chat_id: str, file_id: str) -> None:
    kayit = {k: v for k, v in _son_fotolar().items() if time.time() - v["zaman"] < SON_FOTO_SURESI}
    kayit[chat_id] = {"file_id": file_id, "zaman": time.time()}
    os.makedirs(os.path.dirname(SON_FOTO_YOLU), exist_ok=True)
    with open(SON_FOTO_YOLU, "w", encoding="utf-8") as f:
        json.dump(kayit, f)


@app.get("/son-foto/{chat_id}")
def son_foto(chat_id: str, metin: str = "") -> dict:
    """n8n sohbet dalı sorar: bu yazı, son fotoğrafı bir bitki adıyla yeniden değerlendirme isteği mi?
    `yeniden: true` ise n8n, `file_id`'deki fotoğrafı fotoğraf dalından tekrar geçirir."""
    bitki_ad, _ = _bitki_bul(metin)
    kayit = _son_fotolar().get(chat_id)
    if not bitki_ad:
        return {"yeniden": False, "neden": "metinde bitki adı yok"}
    if len(metin.split()) > SON_FOTO_EN_COK_KELIME:
        return {"yeniden": False, "neden": "uzun mesaj, sohbet olarak ele alınır"}
    if not kayit or time.time() - kayit["zaman"] > SON_FOTO_SURESI:
        return {"yeniden": False, "neden": "son 30 dakikada fotoğraf yok"}
    return {"yeniden": True, "file_id": kayit["file_id"],
            "bitki": BITKI_GORUNEN_AD.get(bitki_ad, bitki_ad.capitalize())}


@app.post("/rapor-kaydet")
def rapor_kaydet(veri: dict = Body(...)) -> dict:
    """n8n'den gelen Claude raporu + model çıktısı (ilk3, bitki, ...) saklanır, numarası döner."""
    import uuid

    os.makedirs(RAPOR_DIR, exist_ok=True)
    rapor_id = uuid.uuid4().hex[:12]
    with open(os.path.join(RAPOR_DIR, f"{rapor_id}.json"), "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False)
    return {"rapor_id": rapor_id}


@app.get("/rapor-pdf/{rapor_id}")
def rapor_pdf(rapor_id: str) -> Response:
    if not rapor_id.isalnum() or len(rapor_id) > 32:
        raise HTTPException(400, "Geçersiz rapor numarası.")
    yol = os.path.join(RAPOR_DIR, f"{rapor_id}.json")
    if not os.path.exists(yol):
        raise HTTPException(404, "Rapor bulunamadı.")
    with open(yol, encoding="utf-8") as f:
        veri = json.load(f)
    pdf_bytes = rapor_pdf_olustur(
        veri, hastalik_tr=veri.get("hastalik_tr"), tarih=veri.get("tarih"), ilk3=veri.get("ilk3"),
        bitki=veri.get("bitki"), bitki_uyumu=veri.get("bitki_uyumu"), rag_kullanildi=veri.get("rag_kullanildi"),
    )
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=leadleaf_rapor_{rapor_id}.pdf"})


@app.post("/generate-pdf")
def generate_pdf(rapor: dict = Body(...), hastalik_tr: str = "", tarih: str = "") -> Response:
    """n8n'in Claude'un ürettiği rapor JSON'unu gönderip PDF istediği endpoint —
    dönen binary, Telegram'a "sendDocument" ile doğrudan iletilebilir."""
    pdf_bytes = rapor_pdf_olustur(rapor, hastalik_tr=hastalik_tr or None, tarih=tarih or None)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=leadleaf_rapor.pdf"},
    )


def _demo_predict() -> tuple[str, np.ndarray]:
    """Rastgele ama tutarlı bir softmax benzeri dağılım üretir (sadece demo modu)."""
    n = len(DEMO_CLASS_NAMES)
    secilen = random.randrange(n)
    probs = np.random.dirichlet(np.ones(n) * 0.6)  # tek sınıfa çekik dağılım
    # seçilen sınıfı en yükseğe zorla (demo daha gerçekçi görünsün diye)
    probs[secilen], probs[probs.argmax()] = max(probs[secilen], probs.max()), probs[secilen]
    probs = probs / probs.sum()
    return DEMO_CLASS_NAMES[int(probs.argmax())], probs


def _gercek_predict(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = _tubitak_on_isle_fonksiyonu(_model_mimari)(arr)
    arr = np.expand_dims(arr, axis=0)
    probs = _model.predict(arr, verbose=0)[0]
    return probs


@app.post("/predict")
async def predict(file: UploadFile = File(...), bitki: str = Form(""),
                  chat_id: str = Form(""), file_id: str = Form("")) -> dict:
    """`bitki`: isteğe bağlı serbest metin (Telegram fotoğraf açıklaması, ör. "şeftali ağacım").
    İçinde desteklenen bir bitki adı geçerse tahmin o bitkinin sınıflarıyla sınırlanır.
    `chat_id` + `file_id` (isteğe bağlı, n8n gönderir): sonradan "domates" gibi bir düzeltme
    yazılırsa aynı fotoğrafın bulunabilmesi için saklanır (bkz. /son-foto)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Sadece görsel dosyası kabul edilir (image/*).")
    if chat_id and file_id:
        _son_foto_kaydet(chat_id, file_id)

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Görsel okunamadı: {e}")

    if DEMO_MODE:
        top_class, probs = _demo_predict()
        siniflar = DEMO_CLASS_NAMES
    else:
        probs = _gercek_predict(img)
        siniflar = _class_names
        top_class = siniflar[int(probs.argmax())]

    guven = float(probs.max()) * 100
    hastalik_tr = TR_ADLAR.get(top_class, top_class)
    uzmana_yonlendir = guven < CONFIDENCE_THRESHOLD * 100

    bitki_ad, onek = _bitki_bul(bitki) if bitki and not DEMO_MODE else (None, None)
    bitki_kaynagi = "aciklama" if onek else None
    plantnet = None
    if not onek and not DEMO_MODE:
        plantnet = _plantnet_tani(raw)
        if plantnet and plantnet["bitki_ad"]:
            bitki_ad, onek, bitki_kaynagi = plantnet["bitki_ad"], BITKI_ONEKLERI[plantnet["bitki_ad"]], "plantnet"
        elif plantnet:
            # Tür güvenle tanındı ama modelin bildiği 14 bitkiden biri değil -> teşhis UYDURMA
            top_class = "Desteklenmeyen_bitki"
            hastalik_tr = f"Desteklenmeyen bitki ({plantnet['yaygin_ad']}, {plantnet['tur']})"
            uzmana_yonlendir = True
            bitki_kaynagi = "plantnet"
    bitki_uyumu = None
    if onek:
        idx = [i for i, s in enumerate(siniflar) if s.startswith(onek + "___")]
        bitki_uyumu = float(probs[idx].sum())
        en_iyi = max(idx, key=lambda i: probs[i])
        top_class, guven = siniflar[en_iyi], float(probs[en_iyi]) * 100
        hastalik_tr = TR_ADLAR.get(top_class, top_class)
        uzmana_yonlendir = guven < CONFIDENCE_THRESHOLD * 100
        if bitki_uyumu < BITKI_UYUM_ESIGI:
            # Görüntü bu bitkinin bilinen hiçbir sınıfına benzemiyor -> bilinen bir hastalık
            # adı UYDURMA; RAG de bu sınıf için boş döner, LLM "tanımsız" üzerinden yazar.
            top_class = f"{onek}___Tanimsiz"
            gorunen = BITKI_GORUNEN_AD.get(bitki_ad, bitki_ad.capitalize())
            hastalik_tr = f"{gorunen}: sistemde tanımlı olmayan belirti"
            uzmana_yonlendir = True

    sirali = sorted(zip(siniflar, probs.tolist()), key=lambda x: x[1], reverse=True)
    ilk3 = [
        {"sinif": s, "sinif_tr": TR_ADLAR.get(s, s), "olasilik": round(p * 100, 1)}
        for s, p in sirali[:3]
    ]

    # Görsel RAG: modelin GAP-katmanı embedding'iyle en benzer referans görselleri bul
    benzer_gorseller: list[dict] = []
    if not DEMO_MODE and _embedding_model is not None:
        try:
            arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype("float32")
            arr = np.expand_dims(_tubitak_on_isle_fonksiyonu(_model_mimari)(arr), axis=0)
            emb = _embedding_model.predict(arr, verbose=0)[0]
            benzer_gorseller = find_similar(emb, k=3)
        except Exception as e:
            print(f"[UYARI] Görsel RAG sorgusu başarısız: {e}")

    return {
        "hastalik": top_class,
        "hastalik_tr": hastalik_tr,
        "guven": round(guven, 1),
        "ilk3": ilk3,
        "uzmana_yonlendir": uzmana_yonlendir,
        "bitki": bitki_ad and BITKI_GORUNEN_AD.get(bitki_ad, bitki_ad.capitalize()),
        "bitki_uyumu": None if bitki_uyumu is None else round(bitki_uyumu * 100, 1),
        "bitki_kaynagi": bitki_kaynagi,
        "plantnet": plantnet,
        "demo_mode": DEMO_MODE,
        "model_surumu": "demo" if DEMO_MODE else f"{_model_mimari}-{len(siniflar)}sinif",
        "benzer_gorseller": benzer_gorseller,
    }


def _tubitak_on_isle_fonksiyonu(ad: str):
    """Her mimarinin KENDİ doğru preprocess_input'u — MobileNetV3Small ve
    EfficientNetB0 için bu fonksiyonlar resmi Keras kaynağında pass-through'tur
    (rescaling zaten modelin içinde), MobileNetV2 için piksefli [-1,1]'e ölçekler.
    Notebook'taki "ÇİFT NORMALİZASYON NOTU" ile birebir aynı mantık."""
    from tensorflow.keras.applications import efficientnet, mobilenet_v2, mobilenet_v3

    return {
        "MobileNetV2": mobilenet_v2.preprocess_input,
        "MobileNetV3Small": mobilenet_v3.preprocess_input,
        "EfficientNetB0": efficientnet.preprocess_input,
    }[ad]


def _tubitak_tek_model_tahmin(ad: str, model, img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype("float32")
    arr = _tubitak_on_isle_fonksiyonu(ad)(arr)
    arr = np.expand_dims(arr, axis=0)
    return model.predict(arr, verbose=0)[0]


@app.post("/predict_compare")
async def predict_compare(file: UploadFile = File(...)) -> dict:
    """Aynı fotoğrafı 3 modele (MobileNetV2, MobileNetV3Small, EfficientNetB0)
    birden verir; model karşılaştırması içindir (bkz.
    notebooks/01_train_model_colab.py). Üretim /predict akışını etkilemez.
    Eksik model dosyası varsa o model tek başına DEMO MODU'na düşer."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Sadece görsel dosyası kabul edilir (image/*).")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Görsel okunamadı: {e}")

    sonuclar = []
    for tanim in TUBITAK_MODEL_DEFS:
        ad = tanim["ad"]
        model = _tubitak_models.get(ad)
        if model is not None:
            probs = _tubitak_tek_model_tahmin(ad, model, img)
            siniflar_bu = _tubitak_class_names
            demo = False
        else:
            _, probs = _demo_predict()
            siniflar_bu = DEMO_CLASS_NAMES
            demo = True

        top_idx = int(np.argmax(probs))
        hastalik = siniflar_bu[top_idx]
        guven = float(probs[top_idx]) * 100
        sonuclar.append({
            "model": ad,
            "hastalik": hastalik,
            "hastalik_tr": TR_ADLAR.get(hastalik, hastalik),
            "guven": round(guven, 1),
            "uzmana_yonlendir": guven < CONFIDENCE_THRESHOLD * 100,
            "demo_mode": demo,
        })

    # Model uzlaşması: üçü aynı sınıfı mı buldu (tam), sadece ikisi mi (kısmi),
    # yoksa üçü de farklı mı (yok) — kural tabanlı, ML tahmini DEĞİL.
    tahminler = [s["hastalik"] for s in sonuclar]
    essiz_sayisi = len(set(tahminler))
    if essiz_sayisi == 1:
        uzlasma_durumu = "tam"
    elif essiz_sayisi == len(tahminler):
        uzlasma_durumu = "yok"
    else:
        uzlasma_durumu = "kismi"

    sayac = Counter(tahminler)
    cogunluk_sinif, cogunluk_adet = sayac.most_common(1)[0]
    en_dusuk_guven = min(s["guven"] for s in sonuclar)
    esik_yuzde = CONFIDENCE_THRESHOLD * 100

    if uzlasma_durumu == "tam" and en_dusuk_guven >= esik_yuzde:
        tavsiye = "Üç model de aynı sonuçta hemfikir ve hepsinin güveni eşiğin üzerinde — sonuç güvenilir."
    elif uzlasma_durumu == "yok":
        tavsiye = "Üç model üç farklı sonuç verdi — modeller anlaşamadı, ziraat mühendisine danışın."
    elif en_dusuk_guven < esik_yuzde:
        tavsiye = "En az bir modelin güveni eşiğin altında — teyit için ikinci bir fotoğraf çekin ya da uzmana danışın."
    else:
        tavsiye = "Modeller kısmen hemfikir — çoğunluk sonucu dikkate alınabilir ama teyit önerilir."

    return {
        "sonuclar": sonuclar,
        "uzlasma": {
            "durum": uzlasma_durumu,
            "cogunluk_sinif": cogunluk_sinif,
            "cogunluk_sinif_tr": TR_ADLAR.get(cogunluk_sinif, cogunluk_sinif),
            "cogunluk_adet": cogunluk_adet,
            "toplam_model": len(sonuclar),
            "esik_yuzde": esik_yuzde,
            "en_dusuk_guven": round(en_dusuk_guven, 1),
            "tavsiye": tavsiye,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference.app:app", host="0.0.0.0", port=8000, reload=True)
