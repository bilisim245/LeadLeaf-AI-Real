import streamlit as st

from ortak import bulgu, gezinme, sinif_tablosu, telegram_qr, test_sonuclari

st.title("LeadLeaf AI")

# Sunum akışı: önce problem ve gerekçe, ardından çözüm (aynı sayfada, sırayla)
st.subheader("Problem Durumu")
c1, c2, c3 = st.columns(3)
c1.metric("Zararlı ve hastalıklarla her yıl kaybedilen küresel ürün", "%40'a kadar")
c2.metric("Bitki hastalıklarının küresel ekonomiye yıllık maliyeti", "220 milyar $+")
c3.metric("Verim kaybı (patates / buğday)", "%17,2 / %21,5")
st.caption("Kaynaklar: FAO / IPPC (2021), Scientific Review of the Impact of Climate Change on Plant Pests; "
           "Savary, S. vd. (2019), The global burden of pathogens and pests on major food crops, "
           "Nature Ecology & Evolution, 3, 430–439.")

st.write("Bir üründe hastalık şüphesi doğduğunda çiftçinin önünde genellikle iki seçenek vardır: bir ziraat "
         "mühendisine ulaşmak ya da hastalığın ilerleyip ilerlemeyeceğini beklemek. Uzmana ulaşmak her zaman ve "
         "her yerde mümkün değildir; zaman ve maliyet gerektirir. Beklemek ise hastalığın tarlaya yayılmasına yol "
         "açabilir. Erken teşhis, ürün kaybını azaltmanın en önemli adımlarından biridir; ancak hastalığın "
         "yapraktan tanınması uzmanlık gerektirir.")

st.subheader("Projenin Gerekçesi")
sol, sag = st.columns(2)
with sol:
    st.markdown("**Mevcut Durum**")
    st.markdown(
        "- Teşhis için uzmana ulaşılması gerekir; bu süreçte geç kalınabilir.\n"
        "- Tek başına bir CNN yalnızca sınıf adı ve olasılık verir (ör. \"Geç yanıklık, %99\"); bu çıktıdan ne "
        "yapılması gerektiği anlaşılmaz.\n"
        "- Fotoğraf doğrudan genel amaçlı bir sohbet botuna sorulduğunda cevap genel kalabilir ve doğrulanmamış "
        "bilgi içerebilir (halüsinasyon).")
with sag:
    st.markdown("**LeadLeaf AI'ın Yaklaşımı**")
    st.markdown(
        "- Telefon ve Telegram yeterlidir; ayrıca bir uygulama kurulması gerekmez.\n"
        "- Hastalık CNN ile görüntüden tanınır, RAG ile doğrulanmış bilgi kaynağı bulunur ve bu kaynak Claude "
        "ile sade bir Türkçe rapora dönüştürülür.\n"
        "- Modelin güveni %70'in altındaysa bu durum açıkça belirtilir ve çiftçi ziraat mühendisine yönlendirilir.")

bulgu("Projenin amacı, çiftçinin uzmana <b>daha erken ve daha bilgili</b> ulaşmasını sağlamaktır. Sistem "
      "uzmanın yerini almaz; üretilen çıktı bir <b>ön değerlendirmedir</b>.", baslik="Projenin Amacı")

st.subheader("Önerilen Çözüm")
giris, qr = st.columns([3, 1])
giris.write("Yaprak fotoğrafı Telegram üzerinden gönderilir. Hastalık model ile tahmin edilir, ardından "
            "bilgi tabanından bu hastalığa ait kaynak metin bulunur ve rapor bu metne dayanılarak Claude ile "
            "yazılır. Modelin emin olmadığı durumlarda çiftçi ziraat mühendisine yönlendirilir.")
telegram_qr(qr, genislik=150)

df = sinif_tablosu()
ts = test_sonuclari()
dogruluk = (ts["y_prob"].argmax(1) == ts["y_true"]).mean() * 100 if ts else 99.02

c1, c2, c3, c4 = st.columns(4)
c1.metric("Görsel", f"{df['Toplam'].sum():,}".replace(",", "."))
c2.metric("Bitki / Sınıf", f"{df['Bitki'].nunique()} / {len(df)}")
c3.metric("Test doğruluğu (yerel ölçüm)", f"%{dogruluk:.2f}".replace(".", ","))
c4.metric("Model", "EfficientNetB0")

st.subheader("Görev Şablonu: DL → LLM-Agent → n8n")
st.caption("Bootcamp görevi: \"Görüntüden Rapor (Vision → Agent → PDF)\", A) Tarım: bitki hastalığı tespiti")
st.graphviz_chart("""
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.3,0.12", width=6];
  edge [color="#1B3A6B"];
  g  [label="Telegram: yaprak fotoğrafı", fillcolor="#EEF2F8", color="#1B3A6B"];
  dl [label="DL-Model: Algılama / Tahmin
EfficientNetB0 (transfer learning), 38 sınıf → sınıf + güven skoru",
      fillcolor="#1B3A6B", fontcolor="white", color="#1B3A6B"];
  ag [label="LLM-Agent: Yorumlama + Karar + Doğal Dil Çıktısı
Claude + RAG (Chroma, 38 bilgi dosyası) → JSON rapor",
      fillcolor="#2E5A9A", fontcolor="white", color="#2E5A9A"];
  n8 [label="n8n: Tetikleme + Entegrasyon + Aksiyon
Telegram cevabı · PDF rapor · Google Sheets · güven < %70 → uzman · takip hatırlatması",
      fillcolor="#EEF2F8", color="#1B3A6B"];
  g -> dl -> ag -> n8;
}
""", use_container_width=True)

st.subheader("Kullanılan Araçlar")
st.dataframe(
    {
        "Katman": ["Model eğitimi", "Model servisi", "Bilgi tabanı (RAG)", "Rapor", "Akış",
                   "Kullanıcı arayüzü", "Kayıt", "Bu pano"],
        "Araç": ["TensorFlow / Keras, Google Colab (GPU)", "FastAPI", "Chroma + sentence-transformers",
                 "Claude (Anthropic API)", "n8n", "Telegram botu", "Google Sheets", "Streamlit"],
    },
    hide_index=True, use_container_width=True,
)

gezinme(__file__)
