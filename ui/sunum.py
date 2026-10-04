"""
LeadLeaf AI — sunum panosu.

Çalıştırma (önce inference servisi açık olmalı, Canlı Demo sayfası onu kullanıyor):
    .venv\\Scripts\\python -m uvicorn inference.app:app --port 8000
    .venv\\Scripts\\python -m streamlit run ui/sunum.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

from ortak import SAYFALAR, STIL, uretim_modeli  # noqa: E402

# Sunumda menü solda hep açık gelsin (Streamlit dar ekran/projektörde kendiliğinden kapatabiliyor)
st.set_page_config(page_title="LeadLeaf AI", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
st.session_state["sunum_modu"] = True  # app.py (Canlı Demo) sadece analiz akışını göstersin
st.logo(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png"), size="large")
st.markdown(STIL, unsafe_allow_html=True)

# Model bir kez, pano açılırken yüklensin; sunum ortasında CNN sayfasında beklemeyelim
with st.spinner("Model yükleniyor..."):
    uretim_modeli()

sayfa = {yol: st.Page(yol, title=baslik, icon=ikon) for yol, baslik, ikon in SAYFALAR}
# Menü, Miuul görev şablonunun üç katmanına göre: DL-Model → LLM-Agent → n8n (Özet sayfasındaki şema)
gruplar = {
    "Proje": ["sayfalar/01_ozet.py"],
    "1 · DL-Model: Algılama": ["sayfalar/02_veri_seti.py", "sayfalar/03_veri_analizi.py", "sayfalar/04_cnn.py",
                               "sayfalar/05_model_secimi.py", "sayfalar/06_fine_tuning.py",
                               "sayfalar/07_test_analizi.py", "sayfalar/08_aciklanabilirlik.py"],
    "2 · LLM-Agent: Karar": ["sayfalar/09_rag.py"],
    "Uçtan Uca": ["app.py", "sayfalar/11_telegram.py"],
    "Sonuç": ["sayfalar/10_sinirliliklar.py"],
}
st.navigation({g: [sayfa[y] for y in yollar] for g, yollar in gruplar.items()}, expanded=True).run()
