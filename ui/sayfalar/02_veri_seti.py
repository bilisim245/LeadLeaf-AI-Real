import os

import streamlit as st

from sklearn.metrics import confusion_matrix

from ortak import (VERI_DIR, bulgu, etiket, gezinme, ornek_dosyalar, sinif_tablosu, test_sonuclari, tr_sirala,
                   veri_var, veri_yok_uyarisi)

st.title("Veri Seti: PlantVillage")
st.write("Veri seti, laboratuvarda sade bir arka plan önünde çekilmiş yaprak fotoğraflarından oluşur. Her "
         "klasör bir sınıfa karşılık gelir ve bitki adı ile hastalık adını (ya da sağlıklı olduğunu) taşır.")

df = sinif_tablosu()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam görsel", f"{df['Toplam'].sum():,}".replace(",", "."))
c2.metric("Bitki", df["Bitki"].nunique())
c3.metric("Sınıf", len(df))
c4.metric("Hastalık sınıfı", int((df["Durum"] == "Hastalıklı").sum()))

if not veri_var():
    veri_yok_uyarisi()
    st.stop()

st.subheader("Veri Setinden Örnek Görseller")
sol, sag = st.columns([1, 3])
with sol:
    bitkiler = sorted(df["Bitki"].unique(), key=tr_sirala)
    secili_bitki = st.selectbox("Bitki", bitkiler, index=bitkiler.index("Domates"))
    alt = df[df["Bitki"] == secili_bitki].sort_values("Toplam", ascending=False)
    # key bitkiye bağlı: bitki değişince önceki bitkinin sınıf seçimi taşınmasın
    secili = st.radio("Sınıf", alt["Sınıf"].tolist(), format_func=lambda s: etiket(s).split(" — ")[1],
                      key=f"sinif_{secili_bitki}")
    if secili not in alt["Sınıf"].values:
        secili = alt["Sınıf"].iloc[0]
    adet = int(alt.loc[alt["Sınıf"] == secili, "Toplam"].iloc[0])
    st.metric("Bu sınıftaki görsel", f"{adet:,}".replace(",", "."))
    if "tohum" not in st.session_state:
        st.session_state.tohum = 0
    if st.button("Başka örnekler göster", use_container_width=True):
        st.session_state.tohum += 1
with sag:
    dosyalar = ornek_dosyalar(secili, 12, st.session_state.tohum)
    for satir in range(3):
        kolonlar = st.columns(4)
        for kolon, yol in zip(kolonlar, dosyalar[satir * 4:(satir + 1) * 4]):
            kolon.image(yol, use_container_width=True)
    st.caption(f"Klasör: {secili} · Görseller 256×256 piksel")

st.subheader("İki Sınıfın Karşılaştırılması")
st.write("Bazı hastalıklar birbirine çok benzer ve model en çok bu sınıflarda zorlanır. Hazır seçenekler, "
         "test setinde en sık karıştırılan üç çifti gösterir.")
siniflar = df.sort_values("Etiket")["Sınıf"].tolist()
ts = test_sonuclari()
karisma = None
if ts:  # testte gerçek karışma sayıları: cm[i, j] = gerçekte i olup j denenler
    cm = confusion_matrix(ts["y_true"], ts["y_prob"].argmax(1), labels=range(len(ts["siniflar"])))
    karisma = {(ts["siniflar"][i], ts["siniflar"][j]): int(cm[i, j])
               for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j]}
    ciftler = sorted({tuple(sorted(k)) for k in karisma},
                     key=lambda c: -(karisma.get(c, 0) + karisma.get(c[::-1], 0)))[:3]
    hazir = {f"{etiket(x)}  ↔  {etiket(y)} ({karisma.get((x, y), 0) + karisma.get((y, x), 0)} hata)": (x, y)
             for x, y in ciftler}
    secim = st.radio("Hazır çift", list(hazir), horizontal=False, label_visibility="collapsed")
    varsayilan = hazir[secim]
else:
    varsayilan = ("Tomato___Early_blight", "Tomato___Target_Spot")
k1, k2 = st.columns(2)
a = k1.selectbox("Birinci sınıf", siniflar, index=siniflar.index(varsayilan[0]), format_func=etiket,
                  key=f"a_{varsayilan}")
b = k2.selectbox("İkinci sınıf", siniflar, index=siniflar.index(varsayilan[1]), format_func=etiket,
                  key=f"b_{varsayilan}")
for kolon, sinif in ((k1, a), (k2, b)):
    ic = kolon.columns(3)
    for k, yol in zip(ic, ornek_dosyalar(sinif, 3, 7)):
        k.image(yol, use_container_width=True)
if karisma is not None and a != b:
    ab, ba = karisma.get((a, b), 0), karisma.get((b, a), 0)
    bulgu(f"Test setinde (8.146 görsel) gerçekte <b>{etiket(a)}</b> olan {ab} görsel <b>{etiket(b)}</b>, "
          f"gerçekte <b>{etiket(b)}</b> olan {ba} görsel ise <b>{etiket(a)}</b> olarak tahmin edilmiştir. "
          + ("Test setinde bu iki sınıf arasında hiç karışma olmamıştır." if ab + ba == 0 else
             "Bu karışmalar Test Sonuçları sayfasındaki karışıklık matrisinde de görülür. Böyle durumlarda "
             "güven düşükse kullanıcı uzmana yönlendirilir."))

with st.expander("Tüm Sınıflar ve Görsel Sayıları"):
    st.dataframe(df[["Etiket", "Sınıf", "Toplam"]].sort_values("Toplam", ascending=False),
                 hide_index=True, use_container_width=True)
    st.caption(f"Kaynak: {os.path.relpath(VERI_DIR)} (GitHub: spMohanty/PlantVillage-Dataset, raw/color)")

gezinme(__file__)
