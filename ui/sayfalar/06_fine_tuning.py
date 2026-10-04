import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ortak import LACIVERT, M38_DIR, TUBITAK_DIR, gezinme, kod_parcasi, nasil_okunur, uretim_modeli

st.title("Fine-Tuning (İnce Ayar)")

st.write("Transfer learning'de önce hazır modelin gövdesi **dondurulur**, yani ağırlıkları "
         "değiştirilmez ve yalnızca en sona eklenen sınıflandırma katmanı eğitilir. Fine-tuning aşamasında ise "
         "gövdenin **son katmanları da eğitime açılır** ve çok küçük adımlarla projenin verisine göre ayarlanır.")

c1, c2 = st.columns(2)
c1.markdown("**Neden Son Katmanlar?**  \nİlk katmanlar kenar ve renk gibi her görselde işe yarayan "
            "özellikleri öğrendiği için bu katmanlar değiştirilmez. Son katmanlar ise ImageNet sınıflarına göre "
            "şekillenmiştir; yaprak hastalıklarına uyarlanması gereken kısım burasıdır.")
c2.markdown("**Neden Küçük Öğrenme Oranı?**  \nİlk aşamada öğrenme oranı 0,001'dir; fine-tuning aşamasında "
            "0,00001'e, yani 100 kat daha küçük bir değere indirilir. Büyük adımlarla ilerlenirse modelin "
            "ImageNet'ten getirdiği bilgi birkaç adımda bozulabilir.")

st.subheader("Uygulanan Yöntem: Kademeli Açma")
asamalar = pd.DataFrame([
    {"Aşama": "0 · Sınıflandırma katmanı", "Açık gövde (%)": 0, "Öğrenme oranı": "0,001", "En fazla epoch": 8},
    {"Aşama": "1 · Son %15 açık", "Açık gövde (%)": 15, "Öğrenme oranı": "0,00001", "En fazla epoch": 6},
    {"Aşama": "2 · Son %30 açık", "Açık gövde (%)": 30, "Öğrenme oranı": "0,00001", "En fazla epoch": 6},
    {"Aşama": "3 · Son %40 açık", "Açık gövde (%)": 40, "Öğrenme oranı": "0,00001", "En fazla epoch": 8},
])
model = uretim_modeli()
govde = model.get_layer("efficientnetb0")
kafa = model.count_params() - govde.count_params()
egitilen = []
for oran in (0, 0.15, 0.30, 0.40):
    n = len(govde.layers)
    acik = govde.layers[int(n * (1 - oran)):] if oran else []
    # Açık katmanların ağırlıkları doğrudan sayılır (BatchNormalization'ın hareketli ortalama/varyansı
    # eğitilmez). Kayıtlı modeldeki trainable bayrağı kullanılmaz: dosya bu bayrağı ilk aşamadaki hâliyle
    # saklamıştır, sonraki aşamalarda açılan katmanları kapalı gösterir.
    egitilen.append(kafa + sum(int(np.prod(w.shape)) for l in acik for w in l.weights if "moving" not in w.name))
asamalar["Eğitilen parametre"] = egitilen

sol, sag = st.columns([3, 2])
with sol:
    st.altair_chart(alt.Chart(asamalar).mark_bar(color=LACIVERT).encode(
        x=alt.X("Eğitilen parametre:Q", title="Eğitilen parametre sayısı"),
        y=alt.Y("Aşama:N", sort=None, title=None),
        tooltip=["Aşama", "Açık gövde (%)", "Öğrenme oranı", alt.Tooltip("Eğitilen parametre:Q", format=",")],
    ).properties(height=220), use_container_width=True)
with sag:
    st.dataframe(asamalar.style.format({"Eğitilen parametre": "{:,.0f}"}), hide_index=True, use_container_width=True)
st.caption("Parametre sayıları üretim modelinin katmanlarından hesaplanmıştır. Modeldeki toplam parametre sayısı: "
           f"{model.count_params():,}".replace(",", ".") + ". Son aşamada katmanların %40'ı açıktır; ancak son katmanlar parametrelerin çoğunu taşıdığı için "
           "parametrelerin yaklaşık %86'sı eğitilmektedir.")
nasil_okunur(
    "Grafikte fine-tuning'in her aşamasında eğitilen (güncellenen) parametre sayısı gösterilmiştir.",
    "Her aşamada gövdenin biraz daha fazlası açıldığı için çubuk uzar. Aşama 0'da yalnızca en sondaki sınıflandırma katmanı eğitilir.",
    "Eğitim kademeli olarak yapılmıştır. Önce yalnızca son katman, ardından sırasıyla gövdenin son %15'i, %30'u ve %40'ı eğitilmiştir. Böylece modelin ImageNet'ten getirdiği bilgi bir anda bozulmamıştır.")

st.markdown("""
Her aşamada iki yardımcı yöntem kullanılmıştır:
- **ReduceLROnPlateau:** Doğrulama kaybı 2 epoch boyunca düzelmezse öğrenme oranı yarıya indirilir.
- **EarlyStopping:** Doğrulama doğruluğu 5 epoch boyunca artmazsa eğitim durdurulur ve en iyi epoch'un ağırlıklarına geri dönülür.

Ayrıca **BatchNormalization** katmanlarına dikkat edilmiştir. Gövde eğitime açıldığında bu katmanlar
istatistiklerini küçük batch'lere göre güncellemeye başlar ve model fark edilmeden bozulabilir. Bunu
önlemek için gövde `training=False` ile çağrılmıştır.
""")

st.subheader("Sonuç: Aynı Model, İki Farklı Eğitim")
ilk = pd.read_csv(os.path.join(TUBITAK_DIR, "model_comparison.csv")).query("model == 'EfficientNetB0'").iloc[0]
son = pd.read_csv(os.path.join(TUBITAK_DIR, "karsilastirma_gelismis.csv")).iloc[0]
kars = pd.DataFrame([
    {"Metrik": ad, "Eğitim": e, "Değer": v}
    for ad, k in (("Doğruluk", "dogruluk"), ("Macro F1", "macro_f1"), ("Macro Recall", "macro_recall"))
    for e, v in (("İlk tarif (son %25, sabit oran)", ilk[k]), ("Kademeli fine-tuning", son[k]))
])
k1, k2, k3 = st.columns(3)
k1.metric("Doğruluk", f"%{son['dogruluk'] * 100:.2f}".replace(".", ","), f"+{(son['dogruluk'] - ilk['dogruluk']) * 100:.2f} puan".replace(".", ","))
k2.metric("Macro F1", f"{son['macro_f1']:.3f}".replace(".", ","), f"+{son['macro_f1'] - ilk['macro_f1']:.3f}".replace(".", ","))
k3.metric("Model boyutu", f"{son['model_boyutu_mb']:.1f} MB".replace(".", ","), f"{son['model_boyutu_mb'] - ilk['model_boyutu_mb']:.1f} MB".replace(".", ","), delta_color="inverse")
st.altair_chart(alt.Chart(kars).mark_bar(clip=True).encode(
    x=alt.X("Değer:Q", scale=alt.Scale(domain=[0.9, 1.0], clamp=True), axis=alt.Axis(format="%"), title=None),
    y=alt.Y("Eğitim:N", title=None, axis=alt.Axis(labelLimit=260)),
    color=alt.Color("Eğitim:N", scale=alt.Scale(domain=["İlk tarif (son %25, sabit oran)", "Kademeli fine-tuning"],
                                                 range=["#8FB4E8", "#2F7D4A"]), legend=None),
    row=alt.Row("Metrik:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
    tooltip=["Metrik", "Eğitim", alt.Tooltip("Değer:Q", format=".2%")],
).properties(height=70, width=620), use_container_width=False)
st.caption("Sonuçlar domatesin 5 sınıflı deneyine aittir. Aynı tarif daha sonra 38 sınıfa uygulanmıştır.")
nasil_okunur(
    "Grafikte aynı EfficientNetB0 modelinin iki farklı eğitim tarifiyle aldığı sonuçlar gösterilmiştir.",
    "Açık mavi ilk tarif (son %25 bir kerede açık, sabit öğrenme oranı), yeşil kademeli fine-tuning. Yeşil çubuk ne kadar uzunsa iyileşme o kadar büyüktür.",
    "Model aynı kalmış, yalnızca eğitim şekli değiştirilmiştir. Bu değişiklikle doğruluk %94,68'den %97,62'ye, macro F1 ise 0,940'tan 0,972'ye yükselmiştir.")

k1, k2 = st.columns(2)
for kolon, yol, baslik in (
    (k1, os.path.join(TUBITAK_DIR, "ogrenme_egrisi_EfficientNetB0_gelismis.png"), "Kademeli fine-tuning (5 sınıf)"),
    (k2, os.path.join(M38_DIR, "ogrenme_egrisi_EfficientNetB0_38sinif.png"), "Aynı tarif, 38 sınıf"),
):
    if os.path.exists(yol):
        kolon.image(yol, caption=baslik, use_container_width=True)
st.caption("Kesikli çizgiler aşama geçişlerini gösterir. Her geçişte küçük bir sıçrama görülmektedir; ancak "
           "eğitim ve doğrulama çizgileri birbirinden ayrılmamaktadır.")
nasil_okunur(
    "Kademeli fine-tuning ile eğitimin epoch bazında ilerleyişi gösterilmiştir. Solda 5 sınıflı deney, sağda 38 sınıflı üretim modeli yer almaktadır.",
    "Mavi eğitim, turuncu doğrulama. Kesikli çizgiler yeni bir aşamanın başladığı yerlerdir. Çizgilerin birlikte yükselmesi iyi, birbirinden ayrılması ezberleme belirtisidir.",
    "Her aşama geçişinde küçük bir sıçrama olmuş, ancak doğrulama doğruluğu istikrarlı biçimde yükselmiştir. Eğitim ve doğrulama çizgileri birbirinden ayrılmadığı için ezberleme belirtisi görülmemektedir.")

with st.expander("Kod: Kademeli Fine-Tuning Döngüsü"):
    st.code(kod_parcasi("notebooks/03_efficientnetb0_38_sinif.py", "for i, (oran, ust_sinir_epoch)",
                        "# %% 8)"), language="python")

gezinme(__file__)
