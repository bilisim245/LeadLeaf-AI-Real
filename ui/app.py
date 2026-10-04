"""
Streamlit Demo Arayüzü: "Tarla 360" (yerel canlı demo)

İki sekme var:
  1. "Analiz" — asıl akış: çiftçi/tarla bilgisi + görsel → /predict (CNN) →
     agent/report.py (LLM + RAG) → db.py'ye kaydet → hava durumu + PDF rapor.
  2. "Veri Analizi" — notebooks/00_veri_kesfi.py'nin Colab'da ürettiği EDA
     çıktılarını (sınıf dağılımı, boyut istatistiği, bozuk/tekrar eden görsel
     kontrolü) grafiklerle gösterir. "Bootcamp'ten istenen veri analizini nerede
     görüyoruz" sorusunun cevabı burası.

Çalıştırma (önce inference servisini ayrı bir terminalde başlat):
    .venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000
    .venv\\Scripts\\python.exe -m streamlit run ui/app.py
"""
from __future__ import annotations

import hashlib
from datetime import datetime
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from agent.pdf_rapor import rapor_pdf_olustur, sonuc_nasil_olustu
from agent.report import generate_report
from agent.weather import weather_summary
from bot.db import DB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortak import bulgu, telegram_qr  # noqa: E402

load_dotenv()

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")
PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_IMAGES_DIR = os.path.join(PROJE_KOKU, "model", "demo_images")
EDA_DIR = os.path.join(PROJE_KOKU, "report", "eda_ciktilari")
TUBITAK_DIR = os.path.join(PROJE_KOKU, "model", "tubitak")
SINIF38_DIR = os.path.join(PROJE_KOKU, "model", "model_38sinif")


# Bitki önceden SEÇTİRİLMEZ: model önce filtresiz tahmin eder, sonra kullanıcıya "bitki doğru mu?"
# diye sorulur. Kullanıcı düzeltirse bu ad /predict'e `bitki` olarak gider -> aynı fotoğraf o bitkinin
# sınıflarıyla yeniden değerlendirilir (kapalı sınıf sorunu, rapor Adım 37; Telegram'daki
# "Domates" düzeltmesiyle aynı mantık). Adlar inference/app.py BITKI_ONEKLERI ile eşleşir.
BITKILER = ["Domates", "Patates", "Biber", "Elma", "Şeftali", "Kiraz", "Üzüm", "Mısır", "Çilek",
            "Portakal", "Ahududu", "Soya", "Kabak", "Yaban mersini"]
DESTEKLENEN_HASTALIKLAR = {
    "Domates": "erken yanıklık, geç yanıklık, bakteriyel leke, septoria yaprak lekesi, yaprak küfü, "
               "kırmızı örümcek, hedef leke, sarı yaprak kıvırcıklığı virüsü, mozaik virüsü",
    "Patates": "erken yanıklık, geç yanıklık",
    "Biber": "bakteriyel leke",
    "Elma": "karaleke, kara çürüklük, elma pası",
    "Üzüm": "kara çürüklük, esca, yaprak yanıklığı",
    "Mısır": "gri yaprak lekesi, pas, kuzey yaprak yanıklığı",
    "Şeftali": "bakteriyel leke",
    "Kiraz": "külleme",
    "Kabak": "külleme",
    "Çilek": "yaprak yanıklığı",
    "Portakal": "turunçgil yeşillenmesi (HLB)",
    "Yaban mersini, ahududu, soya": "yalnızca sağlıklı yaprak",
}
BITKI_TR = {"Tomato": "Domates", "Potato": "Patates", "Pepper,_bell": "Biber", "Apple": "Elma",
            "Peach": "Şeftali", "Cherry_(including_sour)": "Kiraz", "Grape": "Üzüm",
            "Corn_(maize)": "Mısır", "Strawberry": "Çilek", "Orange": "Portakal",
            "Raspberry": "Ahududu", "Soybean": "Soya", "Squash": "Kabak", "Blueberry": "Yaban mersini"}


@st.cache_data
def _tr_adlar() -> dict:
    """inference/app.py'deki TR_ADLAR sözlüğünü TensorFlow'u import ETMEDEN okur (ast ile)."""
    import ast
    with open(os.path.join(PROJE_KOKU, "inference", "app.py"), encoding="utf-8") as f:
        agac = ast.parse(f.read())
    for dugum in agac.body:
        if isinstance(dugum, ast.Assign) and getattr(dugum.targets[0], "id", "") == "TR_ADLAR":
            return ast.literal_eval(dugum.value)
    return {}


LACIVERT, ACIK_MAVI = "#1B3A6B", "#8FB4E8"
DURUM_RENK = alt.Scale(domain=["Hastalıklı", "Sağlıklı"], range=[LACIVERT, ACIK_MAVI])


def _saglikli(sinif: str) -> bool:
    return sinif.endswith("___healthy")


def _tanimsiz(sinif: str) -> bool:
    """Bitki filtresi 'bilinen sınıflara benzemiyor' dediyse (bkz. inference/app.py)."""
    return sinif.endswith("___Tanimsiz") or sinif == "Desteklenmeyen_bitki"


@st.cache_resource
def get_db() -> DB:
    return DB()  # bot/tarla_defteri.sqlite — yerel dosya, ek kurulum yok


def _demo_user_id(isim: str, il: str, ilce: str) -> int:
    """Telegram user_id yok (bu yerel demo) — isim+konumdan sabit bir id üretir,
    böylece aynı çiftçi tekrar girince geçmişi/trendi görülür."""
    h = hashlib.md5(f"{isim}|{il}|{ilce}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)








# Sunum panosundan (ui/sunum.py) açılınca veri ve model sekmeleri gizlenir; o içerik
# panonun kendi sayfalarında var.
SUNUM_MODU = st.session_state.get("sunum_modu", False)
if not SUNUM_MODU:
    st.set_page_config(page_title="LeadLeaf AI: Tarla 360", page_icon="🌿", layout="wide")

st.title("Canlı Demo: Tarla 360" if SUNUM_MODU else "LeadLeaf AI: Tarla 360")

if SUNUM_MODU:
    tab_analiz = st.container()
else:
    tab_analiz, tab_veri, tab_karsilastirma = st.tabs(
        ["🔎 Analiz", "📊 Veri Analizi", "🔬 Model Karşılaştırması"]
    )

# =====================================================================
# SEKME 1 — ANALİZ (asıl akış)
# =====================================================================
with tab_analiz:
    ust_sol, ust_sag = st.columns([4, 1])
    ust_sol.caption(
        "Yaprak fotoğrafı → model tahmini → bilgi tabanına dayalı Claude raporu → hava durumu bilgisi. "
        "*(Çiftçinin kullandığı arayüz Telegram botudur; bu ekran aynı modelin yerel gösterimidir.)*"
    )
    telegram_qr(ust_sag, genislik=130)  # izleyiciler aynı sistemi telefondan deneyebilsin

    # Sol tarafta sadece menü olsun: konum ve bitki listesi fotoğraf alanının yanında
    st.header("📸 Yaprak Fotoğrafı")
    foto_sol, foto_sag = st.columns([3, 2])
    yuklenen = foto_sol.file_uploader("Yaprak fotoğrafı", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    konum = foto_sag.text_input("🌦️ Konum (isteğe bağlı, hava durumu için)", placeholder="ör. Serik, Antalya")
    with foto_sag.expander("Tanınan Bitki ve Hastalıklar"):
        for b, h in DESTEKLENEN_HASTALIKLAR.items():
            st.markdown(f"**{b}:** {h}")
        st.caption("Sistem 14 bitkide 26 hastalığı ve sağlıklı yaprağı tanır. Listede olmayan hastalıklar tanınamaz.")
    if yuklenen:
        foto_sol.image(yuklenen, caption="Yüklenen fotoğraf", width=240)

    analiz_tiklandi = st.button("🔍 Analiz Et", type="primary", disabled=yuklenen is None)

    # Sonuç oturumda saklanır: "Bitki doğru mu?" sorusuna cevap verilince Streamlit sayfayı baştan
    # çalıştırır; sonuç kaybolmasın ve aynı fotoğraf seçilen bitkiyle yeniden değerlendirilebilsin.
    if analiz_tiklandi and yuklenen is not None:
        st.session_state["analiz"] = {"foto": yuklenen.getvalue(), "dosya": yuklenen.name, "bitki": "",
                                      "onay": False}
    analiz = st.session_state.get("analiz")
    if analiz and (yuklenen is None or yuklenen.name != analiz["dosya"]):
        analiz = st.session_state["analiz"] = None  # fotoğraf kaldırıldı ya da değişti

    if analiz:
        if "cnn" not in analiz:
            with st.spinner("Görsel analiz ediliyor..."):
                try:
                    buf = io.BytesIO()
                    Image.open(io.BytesIO(analiz["foto"])).convert("RGB").save(buf, format="JPEG")
                    buf.seek(0)
                    r = requests.post(
                        f"{INFERENCE_URL}/predict",
                        files={"file": ("yaprak.jpg", buf, "image/jpeg")},
                        data={"bitki": analiz["bitki"]},
                        timeout=30,
                    )
                    r.raise_for_status()
                    analiz["cnn"] = r.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"❌ Inference servisine ulaşılamadı ({INFERENCE_URL}).\n\n"
                        "Önce şunu ayrı bir terminalde çalıştır:\n"
                        "`.venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000`"
                    )
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Hata: {e}")
                    st.stop()
            with st.spinner("Rapor hazırlanıyor (RAG + LLM)..."):
                c = analiz["cnn"]
                analiz["rapor"] = generate_report(c["hastalik"], c["hastalik_tr"], c["guven"])
        cnn, rapor = analiz["cnn"], analiz["rapor"]

        if cnn.get("demo_mode"):
            st.warning(
                "⚠️ **DEMO MODU:** Gerçek model henüz `model/` klasörüne konmadı "
                "Sınıflandırma sonucu rastgele üretilmiştir."
            )

        tanimsiz = _tanimsiz(cnn["hastalik"])
        # Konum sonradan yazılırsa/değişirse sadece hava durumu yeniden alınır, analiz tekrarlanmaz
        if analiz.get("hava_konum") != konum.strip():
            analiz["hava_konum"] = konum.strip()
            analiz["hava"] = None
            if konum.strip():
                with st.spinner("Hava durumu kontrol ediliyor..."):
                    analiz["hava"] = weather_summary(konum.strip())
        hava = analiz["hava"]

        st.divider()
        st.header("🔎 Tespit Sonucu")

        c1, c2, c3 = st.columns(3)
        c1.metric("Hastalık", cnn["hastalik_tr"])
        c2.metric("Model güveni", f"%{cnn['guven']}".replace(".", ","))
        c3.metric("Uzmana yönlendirme", "Evet" if (cnn.get("uzmana_yonlendir") or tanimsiz) else "Hayır")

        # Bitki kullanıcıya SONUÇTAN SONRA sorulur (önceden seçtirmek sonucu yönlendirirdi)
        tahmin_bitki = BITKI_TR.get(cnn["hastalik"].split("___")[0])
        if analiz["bitki"]:
            st.caption(f"🔁 Fotoğraf, kullanıcının belirttiği bitkiye göre yeniden değerlendirilmiştir: "
                       f"**{analiz['bitki']}**. Teşhis yalnızca bu bitkinin sınıfları arasından yapılmıştır; bu "
                       f"sınıfların toplam olasılığı: %{str(cnn.get('bitki_uyumu')).replace('.', ',')}.")
        elif analiz["onay"]:
            st.caption(f"✅ Bitki kullanıcı tarafından doğrulanmıştır: **{tahmin_bitki}**")
        elif tahmin_bitki:
            with st.container(border=True):
                st.markdown(f"**Yaprak, model tarafından _{tahmin_bitki}_ yaprağı olarak değerlendirilmiştir. Doğru mu?**")
                e1, e2, e3 = st.columns([1, 1.4, 1.2], vertical_alignment="bottom")
                if e1.button(f"✅ Evet, {tahmin_bitki}", use_container_width=True):
                    analiz["onay"] = True
                    st.rerun()
                dogru_bitki = e2.selectbox("Hayır, bu bir:", [b for b in BITKILER if b != tahmin_bitki],
                                           index=None, placeholder="Bitkiyi seçin")
                if e3.button("🔁 Bu bitkiyle yeniden değerlendir", disabled=dogru_bitki is None,
                             use_container_width=True):
                    st.session_state["analiz"] = {"foto": analiz["foto"], "dosya": analiz["dosya"],
                                                  "bitki": dogru_bitki, "onay": True}
                    st.rerun()

        if tanimsiz:
            st.error(
                f"**{cnn['hastalik_tr']}**: Yaprak, bu bitkinin sistemde tanımlı sınıflarına "
                "benzememektedir. Tanınmayan bir hastalık söz konusu olabileceği için teşhis konulmamıştır. "
                "⚠️ Bir ziraat mühendisine danışılması önerilir."
            )
            # Model filtresiz bakınca başka bir bitkiye güçlü şekilde benzetiyorsa bunu açıkça söyle
            ilk = cnn["ilk3"][0] if cnn.get("ilk3") else None
            benzettigi = BITKI_TR.get(ilk["sinif"].split("___")[0]) if ilk else None
            if benzettigi and benzettigi != analiz["bitki"] and ilk["olasilik"] >= 50:
                st.info(f"Bitki bilgisi olmadan bakıldığında fotoğraf model tarafından **%{ilk['olasilik']}** "
                        f"olasılıkla **{ilk['sinif_tr']}** olarak sınıflandırılmaktadır. Fotoğraf {benzettigi} "
                        f"yaprağına aitse aşağıdan {benzettigi} olarak yeniden değerlendirilebilir; "
                        f"{analiz['bitki'] or cnn.get('bitki')} yaprağı olduğu kesinse, sistemin tanımadığı bir "
                        "belirti söz konusu olabilir.")
                if st.button(f"🔁 {benzettigi} olarak yeniden değerlendir"):
                    st.session_state["analiz"] = {"foto": analiz["foto"], "dosya": analiz["dosya"],
                                                  "bitki": benzettigi, "onay": True}
                    st.rerun()
        elif cnn.get("uzmana_yonlendir"):
            st.warning(f"**{cnn['hastalik_tr']}**: Sonuç yeterince kesin değildir "
                       f"(güven %70'in altında). ⚠️ Bir ziraat mühendisine danışılması önerilir.")
        elif _saglikli(cnn["hastalik"]):
            st.success(f"**{cnn['hastalik_tr']}**: Yaprakta hastalık belirtisi tespit edilmemiştir.")
        else:
            st.info(f"**{cnn['hastalik_tr']}**: Hastalık belirtisi tespit edilmiştir. Ayrıntılar aşağıdaki raporda yer almaktadır.")

        st.subheader("Açıklama")
        st.write(rapor.get("aciklama", "-"))

        neden = rapor.get("neden")
        if neden:
            st.subheader("Hastalığın Nedeni")
            st.write(neden)

        st.subheader("Yayılmayı Azaltmak İçin Yapılabilecekler")
        onlem = rapor.get("onlem", [])
        if isinstance(onlem, list):
            st.markdown("\n".join(f"- {madde}" for madde in onlem) or "-")
        else:
            st.markdown(onlem or "-")

        st.caption(rapor.get("uyari", "Bu bir ön değerlendirmedir, kesin teşhis değildir."))
        st.download_button(
            "📄 Raporu PDF olarak indir",
            data=rapor_pdf_olustur(rapor, hastalik_tr=cnn["hastalik_tr"],
                                   tarih=datetime.now().strftime("%d.%m.%Y %H:%M"),
                                   ilk3=cnn["ilk3"], bitki=cnn.get("bitki"),
                                   bitki_uyumu=cnn.get("bitki_uyumu"),
                                   rag_kullanildi=rapor.get("_rag_kullanildi")),
            file_name=f"leadleaf_rapor_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
        )
        if rapor.get("_rag_kullanildi"):
            st.caption("🔗 Bu açıklama, doğrulanmış kaynak dokümandan (RAG) getirilen bilgiye dayanmaktadır.")
        elif tanimsiz:
            st.caption("ℹ️ Teşhis konulmadığı için bilgi tabanı (RAG) kullanılmamıştır; rapor genel bilgi içerir.")
        else:
            st.caption("⚠️ RAG bağlamı bulunamadı; `rag/build_index.py` çalıştırılmamış olabilir.")
        if rapor.get("_kaynak") == "sablon":
            st.caption(f"*(Rapor: yerel şablon; {'LLM hatası: ' + rapor['_hata'] if rapor.get('_hata') else '`ANTHROPIC_API_KEY` .env içinde tanımlı değil'}.)*")

        st.subheader("Bitki Bilgisi Olmadan Modelin En Yakın Tahminleri" if tanimsiz
                     else "Modelin Diğer Yakın Olasılıkları")
        for it in cnn["ilk3"]:
            st.progress(it["olasilik"] / 100, text=f"{it['sinif_tr']}: %{str(it['olasilik']).replace('.', ',')}")

        st.subheader("Bu Sonuç Nasıl Oluştu?")
        st.markdown("\n".join(f"- {c}" for c in sonuc_nasil_olustu(
            cnn["ilk3"], cnn.get("bitki"), cnn.get("bitki_uyumu"),
            bool(cnn.get("uzmana_yonlendir") or tanimsiz), rapor.get("_rag_kullanildi"))))

        if hava is not None:
            st.subheader(f"🌦️ {konum.strip()} İçin 3 Günlük Hava Durumu")
            if hava.get("bulundu", True) and hava.get("gunler"):
                st.dataframe(pd.DataFrame([{
                    "Tarih": g["tarih"],
                    "Sıcaklık (°C)": f"{g['tmin']} – {g['tmax']}".replace(".", ","),
                    "Ort. nem (%)": g["nem_ort"],
                    "Yağış (mm)": g["yagis_mm"],
                    "Yağış olasılığı (%)": g["yagis_olasilik"],
                } for g in hava["gunler"]]), hide_index=True, width="stretch")
                seviye = {"dusuk": "düşük", "orta": "orta", "yuksek": "yüksek"}.get(hava.get("mantar_riski"), "bilinmiyor")
                nemler = [g["nem_ort"] for g in hava["gunler"] if g["nem_ort"] is not None]
                ort_nem = sum(nemler) / len(nemler) if nemler else 0
                top_yagis = sum(g["yagis_mm"] or 0 for g in hava["gunler"])
                neden_ = {
                    "yuksek": f"ortalama nem %{ort_nem:.1f} ve toplam yağış {top_yagis:.1f} mm; nem %80 ya da "
                              "yağış 10 mm eşiğine ulaştığı için",
                    "orta": f"ortalama nem %{ort_nem:.1f} ve toplam yağış {top_yagis:.1f} mm; nem %65 ya da "
                            "yağış 2 mm eşiğine ulaştığı, ancak yüksek eşiğin altında kaldığı için",
                    "dusuk": f"ortalama nem %{ort_nem:.1f} (%65 eşiğinin altında) ve toplam yağış {top_yagis:.1f} mm "
                             "(2 mm eşiğinin altında) olduğu için",
                }.get(hava.get("mantar_riski"), "").replace(".", ",")
                sinif = cnn["hastalik"]
                if _saglikli(sinif) or tanimsiz:
                    bulgu(f"Önümüzdeki 3 günde {neden_} nem ve yağışa bağlı hastalıkların yayılma koşulları "
                          f"<b>{seviye}</b> olarak değerlendirilmiştir. Nemli dönemlerde yapraklar daha sık "
                          "kontrol edilmelidir.", "Hava durumu değerlendirmesi")
                elif "Spider_mites" in sinif:
                    bulgu("Kırmızı örümcek bir zararlıdır; nemli havada değil, <b>sıcak ve kuru havada</b> artar. "
                          "Kuru ve sıcak günlerde yaprakların alt yüzü daha sık kontrol edilmelidir.",
                          "Hava durumu değerlendirmesi")
                elif "virus" in sinif.lower() or "Haunglongbing" in sinif:
                    bulgu("Bu hastalığın yayılması nem ve yağışla doğrudan ilişkili değildir; böcekler ve bulaşık "
                          "bitki materyaliyle taşınır. Hava durumu bilgi amaçlı gösterilmektedir.",
                          "Hava durumu değerlendirmesi")
                else:
                    bulgu(f"Bu hastalık nemli ve yağışlı havada daha kolay yayılır. Önümüzdeki 3 günde {neden_} "
                          f"yayılma koşulları <b>{seviye}</b> olarak değerlendirilmiştir. Kaynak: Open-Meteo. "
                          "Genel bilgidir, teşhis değildir.", "Hava durumu değerlendirmesi")
            else:
                st.caption("Hava durumu alınamadı; konum adı kontrol edilmelidir.")

        benzerler = cnn.get("benzer_gorseller", [])
        if benzerler:
            st.subheader("🖼️ Benzer Referans Görseller")
            st.caption("(Görsel RAG: modelin öğrendiği özniteliklere göre en yakın örnekler)")
            cols = st.columns(len(benzerler))
            for col, it in zip(cols, benzerler):
                yol = os.path.join(DEMO_IMAGES_DIR, it["dosya"])
                if os.path.exists(yol):
                    col.image(yol, caption=f"{it['sinif']}: %{it['benzerlik']}")

        with st.expander("🔧 Ham CNN çıktısı (debug)"):
            st.json(cnn)

    elif not yuklenen:
        st.info("👆 Analiz için bir yaprak fotoğrafı yüklenmelidir.")

if not SUNUM_MODU:
    # =====================================================================
    # SEKME 2 — VERİ ANALİZİ (notebooks/00_veri_kesfi.py çıktıları)
    # =====================================================================
    with tab_veri:
        st.caption(
            "Üretimdeki 38 sınıflı modelin eğitildiği veri — PlantVillage (ham, "
            "abdallahalidev/plantvillage-dataset). Sayılar eğitim sırasında kaydedilen "
            "`model/model_38sinif/split_manifest.json` dosyasından okunur: veri VARSAYIMLA "
            "değil SAYIYLA inceleniyor."
        )

        manifest_yolu = os.path.join(SINIF38_DIR, "split_manifest.json")
        if os.path.exists(manifest_yolu):
            with open(manifest_yolu, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            sayilar = manifest["sayilar"]  # {"train": {sinif: n}, "valid": {...}, "test": {...}}
            siniflar = list(sayilar["train"])
            df_s = pd.DataFrame([
                {"Sınıf": s, "Bitki": BITKI_TR.get(s.split("___")[0], s.split("___")[0]),
                 "Durum": "Sağlıklı" if _saglikli(s) else "Hastalıklı",
                 "Eğitim": sayilar["train"].get(s, 0), "Doğrulama": sayilar["valid"].get(s, 0),
                 "Test": sayilar["test"].get(s, 0)}
                for s in siniflar
            ])
            df_s["Toplam"] = df_s[["Eğitim", "Doğrulama", "Test"]].sum(axis=1)
            en_kucuk, en_buyuk = df_s.loc[df_s["Toplam"].idxmin()], df_s.loc[df_s["Toplam"].idxmax()]

            st.subheader("📌 Genel özet — 38 sınıf")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam görsel", f"{int(df_s['Toplam'].sum()):,}".replace(",", "."))
            c2.metric("Bitki / sınıf", f"{df_s['Bitki'].nunique()} / {len(df_s)}")
            c3.metric("Hastalık sınıfı", int((df_s["Durum"] == "Hastalıklı").sum()))
            c4.metric("Dengesizlik (en büyük / en küçük)",
                      f"{en_buyuk['Toplam'] / en_kucuk['Toplam']:.0f}x")

            oranlar = manifest.get("oranlar", {})
            st.info(
                f"**Bölme:** her sınıf kendi içinde %{int(oranlar.get('train', .7) * 100)} eğitim / "
                f"%{int(oranlar.get('valid', .15) * 100)} doğrulama / "
                f"%{int(oranlar.get('test', .15) * 100)} test olarak TEK SEFERDE, sabit tohumla "
                f"(seed={manifest.get('seed')}) bölündü. Test görselleri eğitimde hiç kullanılmadı; "
                "hangi dosyanın nereye düştüğü manifest'te kayıtlı (tekrarlanabilirlik + sızıntı denetimi)."
            )

            st.subheader("🌱 Bitki bazında görsel sayısı")
            df_bitki = df_s.groupby(["Bitki", "Durum"], as_index=False)["Toplam"].sum()
            # Sıralama için bitki toplamını ayrı sütun olarak veriyoruz (yığılmış çubukta "-x" güvenilir değil)
            bitki_sirasi = (df_bitki.groupby("Bitki")["Toplam"].sum()
                            .sort_values(ascending=False).index.tolist())
            st.altair_chart(
                alt.Chart(df_bitki).mark_bar().encode(
                    x=alt.X("Toplam:Q", stack="zero", title="Görsel sayısı"),
                    y=alt.Y("Bitki:N", sort=bitki_sirasi, title=None),
                    color=alt.Color("Durum:N", scale=DURUM_RENK, legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Bitki", "Durum", alt.Tooltip("Toplam:Q", format=",")],
                ).properties(height=420),
                use_container_width=True,
            )
            st.subheader("⚖️ Sınıf dengesizliği")
            tr = _tr_adlar()
            df_s["Etiket"] = [
                f"{b} — {'Sağlıklı' if _saglikli(s) else tr.get(s, s).split(' (')[0]}"
                for s, b in zip(df_s["Sınıf"], df_s["Bitki"])
            ]
            st.altair_chart(
                alt.Chart(df_s).mark_bar(cornerRadiusEnd=3).encode(
                    x=alt.X("Toplam:Q", title="Görsel sayısı"),
                    y=alt.Y("Etiket:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                    color=alt.Color("Durum:N", scale=DURUM_RENK, legend=None),
                    tooltip=["Sınıf", "Durum", alt.Tooltip("Toplam:Q", format=",")],
                ).properties(height=760),
                use_container_width=True,
            )
            st.caption(f"En küçük sınıf: {en_kucuk['Sınıf']} ({en_kucuk['Toplam']}) · "
                       f"en büyük: {en_buyuk['Sınıf']} ({en_buyuk['Toplam']:,})".replace(",", "."))

            with st.expander("📋 Sınıf bazında tam tablo"):
                st.dataframe(df_s.sort_values("Toplam", ascending=False),
                             width="stretch", hide_index=True)

            st.warning(
                "**Veri setinin sınırlılıkları** (rapor Bölüm 5): görseller laboratuvar koşullarında "
                "(tek yaprak, sade arkaplan) çekilmiş — tarladaki başarı ayrıca ölçülmeli. Türkiye'de "
                "önemli birçok hastalık (ör. üzüm mildiyösü, şeftali yaprak kıvırcıklığı) ve bitki "
                "(zeytin, fındık, buğday) veri setinde yok."
            )
        else:
            st.info("38 sınıf eğitim manifest'i bulunamadı (`model/model_38sinif/split_manifest.json`).")

        ozet_json_yolu = os.path.join(EDA_DIR, "veri_ozeti.json")
        if os.path.exists(ozet_json_yolu):
            st.divider()
            st.header("🍅 İlk aşama: domates alt kümesi keşifsel analizi")
            with open(ozet_json_yolu, "r", encoding="utf-8") as f:
                ozet = json.load(f)

            st.subheader("📌 Genel özet")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam sınıf (tüm veri seti)", ozet["toplam_sinif"])
            c2.metric("Toplam görsel", f"{ozet['toplam_gorsel']:,}")
            c3.metric("Dengesizlik oranı (domates)", f"{ozet['dengesizlik_orani']}x")
            c4.metric("Bozuk/açılamayan görsel", ozet["bozuk_gorsel"]["toplam"])

            st.subheader("🍅 Domates alt kümesi — sınıf dağılımı")
            df_sinif = pd.DataFrame(
                [{"Sınıf": k, "Görsel sayısı": v} for k, v in ozet["domates_sayilar"].items()]
            )
            st.bar_chart(df_sinif.set_index("Sınıf"))
            st.dataframe(df_sinif, width="stretch", hide_index=True)

            gb = ozet.get("gorsel_boyutu", {})
            st.subheader("📐 Görsel boyutu")
            if gb.get("sabit"):
                st.success(f"Tüm görseller aynı boyutta: {gb['genislik']}×{gb['yukseklik']}px")
            else:
                st.info("Görsel boyutları değişken (model eğitimi zaten hepsini 224×224'e ölçekliyor).")

            st.subheader("🧹 Bozuk / açılamayan görsel (\"boş veri\" kontrolü)")
            df_bozuk = pd.DataFrame(
                [{"Sınıf": k, "Bozuk görsel": v} for k, v in ozet["bozuk_gorsel"]["sinif_bazli"].items()]
            )
            if ozet["bozuk_gorsel"]["toplam"] > 0:
                st.error(f"Toplam {ozet['bozuk_gorsel']['toplam']} bozuk/açılamayan görsel bulundu — "
                         "eğitimden önce temizlenmesi önerilir.")
            else:
                st.success("Bozuk/açılamayan görsel bulunamadı.")
            st.dataframe(df_bozuk, width="stretch", hide_index=True)

            st.subheader("🔁 Tekrar eden (duplicate) görsel kontrolü")
            st.caption(
                "Ham veri havuzunda aynı/çok benzer görsel var mı — varsa split sırasında "
                "biri train'e biri valid'e düşüp yapay bir sızıntı yaratabilir."
            )
            df_tekrar = pd.DataFrame([
                {"Sınıf": k, "Örneklenen": v["orneklenen"], "Tekrar eden çift": v["tekrar_cift"],
                 "Oran (%)": v["oran_yuzde"]}
                for k, v in ozet.get("tekrar_eden", {}).items()
            ])
            st.dataframe(df_tekrar, width="stretch", hide_index=True)
            yuksek_tekrar = df_tekrar[df_tekrar["Oran (%)"] > 5] if not df_tekrar.empty else df_tekrar
            if not yuksek_tekrar.empty:
                st.warning(f"⚠️ {len(yuksek_tekrar)} sınıfta %5'in üzerinde tekrar oranı — "
                           "split öncesi tekilleştirme (deduplication) düşünülebilir.")
            else:
                st.success("Tüm sınıflarda tekrar oranı düşük (%5 altı).")

            st.subheader("🖼️ Örnek görseller ve grafikler")
            for dosya, baslik in [
                ("sinif_dagilimi.png", "Sınıf dağılımı — tüm veri seti"),
                ("ornek_gorseller_izgara.png", "Domates 5 sınıftan örnekler"),
                ("boyut_dagilimi.png", "Görsel boyutu dağılımı"),
            ]:
                yol = os.path.join(EDA_DIR, dosya)
                if os.path.exists(yol):
                    st.image(yol, caption=baslik, width="stretch")

            with st.expander("📄 Ham özet (veri_ozeti.txt)"):
                txt_yolu = os.path.join(EDA_DIR, "veri_ozeti.txt")
                if os.path.exists(txt_yolu):
                    with open(txt_yolu, "r", encoding="utf-8") as f:
                        st.text(f.read())

    # =====================================================================
    # SEKME 3: MODEL KARŞILAŞTIRMASI (MobileNetV2, MobileNetV3Small, EfficientNetB0)
    # =====================================================================
    with tab_karsilastirma:
        st.caption(
            "İki aşamalı deney: **(1)** üç mimari (MobileNetV2, MobileNetV3Small, EfficientNetB0) "
            "domatesin 5 sınıfında AYNI veri bölmesi ve AYNI eğitim koşullarıyla karşılaştırıldı → "
            "EfficientNetB0 seçildi; **(2)** seçilen mimari PlantVillage'ın 38 sınıfına (14 bitki) "
            "genişletilip üretime alındı. Tüm metrikler eğitimde hiç kullanılmamış bağımsız test "
            "setinde ölçüldü."
        )

        sinif38_csv = os.path.join(SINIF38_DIR, "model_comparison.csv")
        if os.path.exists(sinif38_csv):
            st.subheader("🚀 Üretimdeki model — EfficientNetB0, 38 sınıf")
            m38 = pd.read_csv(sinif38_csv).iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Test doğruluğu", f"%{m38['dogruluk'] * 100:.2f}".replace(".", ","))
            c2.metric("Macro F1", f"%{m38['macro_f1'] * 100:.2f}".replace(".", ","))
            c3.metric("Macro AUC", f"%{m38['macro_auc'] * 100:.2f}".replace(".", ","))
            c4.metric("Model boyutu", f"{m38['model_boyutu_mb']:.1f} MB".replace(".", ","))
            c5.metric("Görüntü başına süre", f"{m38['ort_inference_ms']:.0f} ms")
            col_cm38, col_egri38 = st.columns(2)
            for col, dosya, baslik in [
                (col_cm38, "confusion_matrix_EfficientNetB0_38sinif.png", "Confusion Matrix (bağımsız test seti)"),
                (col_egri38, "ogrenme_egrisi_EfficientNetB0_38sinif.png", "Öğrenme eğrisi (doğruluk + kayıp)"),
            ]:
                yol = os.path.join(SINIF38_DIR, dosya)
                if os.path.exists(yol):
                    col.image(yol, caption=baslik, width="stretch")
            st.caption("⚠️ Bu doğruluk laboratuvar koşullarındaki test görsellerinde ölçüldü; tarla "
                       "fotoğraflarındaki başarı ayrıca ölçülmelidir (rapor 5.3).")
            st.divider()

        karsilastirma_csv = os.path.join(TUBITAK_DIR, "model_comparison.csv")
        if os.path.exists(karsilastirma_csv):
            st.subheader("🏁 Mimari seçimi — 3 model, domates 5 sınıf (aynı koşullar)")
            df_test_sonuc = pd.read_csv(karsilastirma_csv)
            st.bar_chart(df_test_sonuc.set_index("model")[["dogruluk", "macro_f1", "macro_auc"]])
            st.dataframe(
                df_test_sonuc[["model", "dogruluk", "macro_precision", "macro_recall",
                                "macro_f1", "macro_auc", "model_boyutu_mb", "ort_inference_ms"]],
                width="stretch", hide_index=True,
            )
            grafik_yolu = os.path.join(TUBITAK_DIR, "model_karsilastirma_dogruluk.png")
            if os.path.exists(grafik_yolu):
                st.image(grafik_yolu, width="stretch")

            st.subheader("🔍 Model bazında detaylı grafikler")
            MODEL_GRAFIK_SECENEKLERI = {
                "MobileNetV2": "MobileNetV2",
                "MobileNetV3Small": "MobileNetV3Small",
                "EfficientNetB0 (temel tarif)": "EfficientNetB0",
                "EfficientNetB0 (gelişmiş fine-tuning, 5 sınıf — %97,62)": "EfficientNetB0_gelismis",
            }
            secilen_etiket = st.selectbox(
                "Hangi modelin confusion matrix'ini ve öğrenme eğrisini görmek istersin?",
                list(MODEL_GRAFIK_SECENEKLERI.keys()),
                key="model_grafik_secimi",
            )
            secilen_dosya_eki = MODEL_GRAFIK_SECENEKLERI[secilen_etiket]

            col_cm, col_egri = st.columns(2)
            cm_yolu = os.path.join(TUBITAK_DIR, f"confusion_matrix_{secilen_dosya_eki}.png")
            egri_yolu = os.path.join(TUBITAK_DIR, f"ogrenme_egrisi_{secilen_dosya_eki}.png")
            with col_cm:
                if os.path.exists(cm_yolu):
                    st.image(cm_yolu, caption="Confusion Matrix (bağımsız test seti)", width="stretch")
                else:
                    st.info("Bu model için confusion matrix henüz yok.")
            with col_egri:
                if os.path.exists(egri_yolu):
                    st.image(egri_yolu, caption="Öğrenme Eğrisi (doğruluk + kayıp)", width="stretch")
                else:
                    st.info("Bu model için öğrenme eğrisi henüz yok.")

            if secilen_dosya_eki == "EfficientNetB0_gelismis":
                onceki_gelismis_yolu = os.path.join(TUBITAK_DIR, "efficientnetb0_onceki_vs_gelismis.png")
                if os.path.exists(onceki_gelismis_yolu):
                    st.image(onceki_gelismis_yolu, caption="Önceki tarif vs Gelişmiş tarif (5 metrik)",
                              width="stretch")

            st.divider()
        else:
            st.info(
                "📈 Test seti doğruluk karşılaştırması henüz yok — Colab notebook'u "
                "çalışıp `model_comparison.csv` `model/tubitak/`'a konunca burada otomatik görünür."
            )

        st.subheader("🖼️ Tek fotoğrafla canlı karşılaştırma (3 mimari)")
        st.caption("Bu üç model domatesin 5 sınıfıyla (sağlıklı, erken/geç yanıklık, bakteriyel leke, "
                   "septoria) eğitilmiştir — **yalnızca domates yaprağı yüklenmelidir.** Model uzlaşması ve %70 "
                   "güven eşiği kural tabanlı yorumlanır (ML tahmini değil).")
        karsilastirma_foto = st.file_uploader(
            "Yaprak fotoğrafı yükle", type=["jpg", "jpeg", "png"], key="karsilastirma_uploader"
        )
        if karsilastirma_foto:
            st.image(karsilastirma_foto, caption="Yüklenen fotoğraf", width=220)

        karsilastir_tiklandi = st.button(
            "🔬 Üç Modelle Karşılaştır", type="primary", disabled=karsilastirma_foto is None
        )

        if karsilastir_tiklandi and karsilastirma_foto is not None:
            with st.spinner("Üç model de çalıştırılıyor..."):
                try:
                    img = Image.open(karsilastirma_foto)
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG")
                    buf.seek(0)
                    r = requests.post(
                        f"{INFERENCE_URL}/predict_compare",
                        files={"file": ("yaprak.jpg", buf, "image/jpeg")},
                        timeout=60,
                    )
                    r.raise_for_status()
                    karsilastirma = r.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"❌ Inference servisine ulaşılamadı ({INFERENCE_URL}).\n\n"
                        "Önce şunu ayrı bir terminalde çalıştır:\n"
                        "`.venv\\Scripts\\python.exe -m uvicorn inference.app:app --port 8000`"
                    )
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Hata: {e}")
                    st.stop()

            sonuclar = karsilastirma["sonuclar"]
            uzlasma = karsilastirma["uzlasma"]

            if any(s["demo_mode"] for s in sonuclar):
                eksikler = [s["model"] for s in sonuclar if s["demo_mode"]]
                st.warning(
                    f"⚠️ **DEMO MODU:** şu modeller henüz `model/tubitak/` içinde yok, "
                    f"sonuçları RASTGELE üretildi: {', '.join(eksikler)}. Karşılaştırma "
                    f"betiğinin çıktısı oraya kopyalanınca gerçek sonuca döner."
                )

            st.divider()
            st.subheader("📊 Üç modelin sonucu")
            df_karsilastirma = pd.DataFrame([
                {
                    "Model": s["model"],
                    "Tahmin": s["hastalik_tr"],
                    "Güven (%)": s["guven"],
                    "Eşik (%70) durumu": "⚠️ Altında" if s["uzmana_yonlendir"] else "✅ Üstünde",
                    "Mod": "Demo" if s["demo_mode"] else "Gerçek",
                }
                for s in sonuclar
            ])
            st.dataframe(df_karsilastirma, width="stretch", hide_index=True)

            st.subheader("🤝 Model uzlaşması")
            durum_etiket = {"tam": "Tam uzlaşma", "kismi": "Kısmi uzlaşma", "yok": "Uzlaşma yok"}
            c1, c2, c3 = st.columns(3)
            c1.metric("Uzlaşma durumu", durum_etiket.get(uzlasma["durum"], uzlasma["durum"]))
            c2.metric("Çoğunluk tahmini", f"{uzlasma['cogunluk_sinif_tr']} ({uzlasma['cogunluk_adet']}/{uzlasma['toplam_model']})")
            c3.metric("En düşük güven", f"%{uzlasma['en_dusuk_guven']}")

            if uzlasma["durum"] == "tam" and uzlasma["en_dusuk_guven"] >= uzlasma["esik_yuzde"]:
                st.success(f"✅ {uzlasma['tavsiye']}")
            elif uzlasma["durum"] == "yok":
                st.error(f"❌ {uzlasma['tavsiye']}")
            else:
                st.warning(f"⚠️ {uzlasma['tavsiye']}")

            with st.expander("🔧 Ham API çıktısı (debug)"):
                st.json(karsilastirma)
        elif not karsilastirma_foto:
            st.info("👆 Analiz için bir yaprak fotoğrafı yüklenmelidir.")

if SUNUM_MODU:
    from ortak import gezinme
    gezinme(__file__)
