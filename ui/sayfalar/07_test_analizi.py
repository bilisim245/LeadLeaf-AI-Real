import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ortak import (LACIVERT, M38_DIR, TURUNCU, VERI_DIR, YESIL, etiket, gezinme, nasil_okunur, test_sonuclari,
                   veri_var)

st.title("Test Sonuçları (38 Sınıf)")

ts = test_sonuclari()
if ts is None:
    st.warning("Önce `notebooks/04_yerel_test_degerlendirme.py` çalıştırılmalıdır.")
    st.stop()

siniflar, y_true, y_prob = ts["siniflar"], ts["y_true"], ts["y_prob"]
y_pred, guven = y_prob.argmax(1), y_prob.max(1) * 100
dogru = y_pred == y_true
etiketler = [etiket(s) for s in siniflar]

st.write("Test kümesi eğitimde hiç kullanılmamıştır. Model bu 8.146 görselin tamamında **yerel olarak "
         "yeniden çalıştırılmıştır**; aşağıdaki sonuçların hepsi bu tahminlerden hesaplanmıştır.")

p, r, f1, destek = precision_recall_fscore_support(y_true, y_pred, labels=range(len(siniflar)), zero_division=0)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Test görseli", f"{len(y_true):,}".replace(",", "."))
c2.metric("Doğruluk", f"%{accuracy_score(y_true, y_pred) * 100:.2f}".replace(".", ","),
          "Colab'da %99,02", delta_color="off")
c3.metric("Macro F1", f"{f1.mean():.4f}".replace(".", ","))
c4.metric("Yanlış tahmin", int((~dogru).sum()))
c5.metric("Ortalama güven", f"%{guven.mean():.1f}".replace(".", ","))
st.caption("Colab sonucuyla aradaki küçük fark (birkaç görsel), görseller 224×224 piksele küçültülürken "
           "farklı bir kütüphane kullanılmasından kaynaklanmaktadır.")

t1, t2, t3, t4, t5 = st.tabs(["Karışıklık Matrisi", "Sınıf Bazında", "Güven ve Eşik", "Yanlış Tahminler",
                              "Colab Grafikleri"])

with t1:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(siniflar)))
    sadece_hata = st.toggle("Yalnızca hataları göster (köşegeni gizle)", value=True)
    kayit = [{"Gerçek": etiketler[i], "Tahmin": etiketler[j], "Adet": int(cm[i, j]),
              "Oran": cm[i, j] / max(cm[i].sum(), 1)}
             for i in range(len(siniflar)) for j in range(len(siniflar))
             if cm[i, j] > 0 and not (sadece_hata and i == j)]
    sira = sorted(etiketler)
    st.altair_chart(alt.Chart(pd.DataFrame(kayit)).mark_rect().encode(
        x=alt.X("Tahmin:N", sort=sira, axis=alt.Axis(labelAngle=-60, labelLimit=200, labelFontSize=9)),
        y=alt.Y("Gerçek:N", sort=sira, axis=alt.Axis(labelLimit=220, labelFontSize=9)),
        color=alt.Color("Adet:Q", scale=alt.Scale(scheme="blues" if not sadece_hata else "oranges",
                                                  type="log" if not sadece_hata else "linear"),
                        legend=alt.Legend(title="Görsel")),
        tooltip=["Gerçek", "Tahmin", "Adet", alt.Tooltip("Oran:Q", format=".1%")],
    ).properties(height=760), use_container_width=True)
    hatalar = pd.DataFrame([{"Gerçek": etiketler[i], "Tahmin": etiketler[j], "Adet": int(cm[i, j])}
                            for i in range(len(siniflar)) for j in range(len(siniflar)) if i != j and cm[i, j]])
    st.markdown("**En Sık Karışan Sınıflar**")
    st.dataframe(hatalar.sort_values("Adet", ascending=False).head(10), hide_index=True, use_container_width=True)
    st.caption("Karışmaların çoğu aynı bitkinin hastalıkları arasında görülür. Bitki türü neredeyse hiç "
               "karıştırılmamaktadır; model daha çok birbirine benzeyen lekelerde zorlanmaktadır.")
    nasil_okunur(
        "Karışıklık matrisinde satırlar gerçek sınıfı, sütunlar modelin tahminini gösterir. \"Yalnızca hataları göster\" seçeneği açıkken yalnızca yanlış tahminler çizilir.",
        "Bir kare, o satırdaki gerçek sınıfın o sütundaki sınıfla kaç kez karıştırıldığını gösterir; renk koyulaştıkça karışma artar. Fare kareye getirilince sayı görünür.",
        "8.146 test görselinde 83 hata vardır ve bunların 70'i aynı bitkinin hastalıkları arasındadır. En sık karışma, mısırda gri yaprak lekesi ile kuzey yaprak yanıklığı arasında görülmüştür (iki yönde toplam 17).")

with t2:
    sdf = pd.DataFrame({"Sınıf": etiketler, "Precision": p, "Recall": r, "F1": f1, "Test görseli": destek})
    metrik = st.radio("Metrik", ["F1", "Recall", "Precision"], horizontal=True)
    metrik = metrik or "F1"
    st.altair_chart(alt.Chart(sdf).mark_bar().encode(
        x=alt.X(f"{metrik}:Q", scale=alt.Scale(domain=[0.9, 1.0], clamp=True), axis=alt.Axis(format="%")),
        y=alt.Y("Sınıf:N", sort="x", title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.condition(alt.datum[metrik] < 0.97, alt.value(TURUNCU), alt.value(LACIVERT)),
        tooltip=["Sınıf", alt.Tooltip("Precision:Q", format=".2%"), alt.Tooltip("Recall:Q", format=".2%"),
                 alt.Tooltip("F1:Q", format=".2%"), "Test görseli"],
    ).properties(height=760), use_container_width=True)
    st.caption("Turuncu çubuklar %97'nin altında kalan sınıfları gösterir. Precision, modelin \"bu hastalık\" "
               "dediği görsellerin ne kadarında haklı çıktığını; recall ise o hastalığa sahip görsellerin ne "
               "kadarının yakalandığını ifade eder.")
    nasil_okunur(
        "Grafikte 38 sınıfın her biri için F1 (ya da seçilen metrik) değeri küçükten büyüğe sıralanmıştır.",
        "En üstteki sınıflar modelin en çok zorlandıklarıdır. Turuncu çubuklar %97'nin altında kalan sınıflardır.",
        "Beş sınıf %97'nin altındadır; en düşükleri mısır gri yaprak lekesi (0,889) ve domates erken yanıklığıdır (0,899). Bu sınıflar, benzer lekeler oluşturan hastalıklarla karışmaktadır.")
    st.dataframe(sdf.sort_values(metrik).style.format({"Precision": "{:.2%}", "Recall": "{:.2%}", "F1": "{:.2%}"}),
                 hide_index=True, use_container_width=True)

with t3:
    gdf = pd.DataFrame({"Güven": guven, "Sonuç": np.where(dogru, "Doğru", "Yanlış")})
    sol, sag = st.columns(2)
    with sol:
        st.markdown("**Doğru ve Yanlış Tahminlerde Güven**")
        st.altair_chart(alt.Chart(gdf).mark_bar(opacity=0.85).encode(
            x=alt.X("Güven:Q", bin=alt.Bin(step=5), title="Güven (%)"),
            y=alt.Y("count():Q", stack=None, scale=alt.Scale(type="symlog"), title="Görsel (log ölçek)"),
            color=alt.Color("Sonuç:N", scale=alt.Scale(domain=["Doğru", "Yanlış"], range=[LACIVERT, TURUNCU]),
                            legend=alt.Legend(orient="top", title=None)),
        ).properties(height=320), use_container_width=True)
        st.caption(f"Yanlış tahminlerde ortalama güven %{f'{guven[~dogru].mean():.1f}'.replace('.', ',')}, doğru "
                   f"tahminlerde %{f'{guven[dogru].mean():.1f}'.replace('.', ',')} olarak ölçülmüştür. Model "
                   "yanıldığında genellikle daha az emindir.")
        nasil_okunur(
            "Grafikte modelin verdiği güven değerlerinin dağılımı gösterilmiştir; lacivert doğru, turuncu yanlış tahminleri temsil eder.",
            "Dikey eksen logaritmiktir, az sayıdaki yanlış tahminler de görünsün diye. Çubuklar sağa yığılıyorsa model çoğunlukla çok emindir.",
            "Doğru tahminlerde ortalama güven %99,2, yanlış tahminlerde %71,2'dir. Yine de 20 yanlış tahminde güven %90'ın üzerindedir; bu nedenle güven eşiği tek başına yeterli bir güvence değildir.")
    with sag:
        st.markdown("**Uzmana Yönlendirme Eşiği**")
        esik = st.slider("Güven eşiği (%)", 30, 99, 70)
        ustunde = guven >= esik
        k1, k2 = st.columns(2)
        k1.metric("Eşiğin üstünde (bot cevap verir)", f"%{ustunde.mean() * 100:.1f}".replace(".", ","))
        k2.metric("Bu gruptaki doğruluk", f"%{dogru[ustunde].mean() * 100:.2f}".replace(".", ",") if ustunde.any() else "-")
        k1.metric("Uzmana yönlendirilen", int((~ustunde).sum()))
        k2.metric("Yakalanan yanlış", f"{int((~dogru & ~ustunde).sum())} / {int((~dogru).sum())}")
        esikler = np.arange(30, 100)
        egri = pd.DataFrame({
            "Eşik": np.concatenate([esikler, esikler]),
            "Değer": np.concatenate([[dogru[guven >= e].mean() for e in esikler],
                                     [(guven < e).mean() for e in esikler]]),
            "Ölçü": ["Cevap verilenlerde doğruluk"] * len(esikler) + ["Uzmana giden oranı"] * len(esikler),
        })
        st.altair_chart(alt.Chart(egri).mark_line().encode(
            x="Eşik:Q", y=alt.Y("Değer:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("Ölçü:N", scale=alt.Scale(range=[LACIVERT, YESIL]), legend=alt.Legend(orient="top", title=None)),
        ).properties(height=220) + alt.Chart(pd.DataFrame({"x": [esik]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"),
            use_container_width=True)
        nasil_okunur(
            "Grafikte güven eşiği değiştikçe cevap verilen tahminlerin doğruluğunun (lacivert) ve uzmana yönlendirilen oranın (yeşil) nasıl değiştiği gösterilmiştir.",
            "Kaydırıcı sağa alındıkça lacivert çizgi yükselir ama yeşil çizgi de yükselir. Kesikli çizgi seçili eşiktir.",
            "%70 eşiğinde tahminlerin %98,8'ine cevap verilir ve bu cevapların doğruluğu %99,5'tir. 83 hatanın 42'si ise uzmana yönlendirilerek yakalanır.")
    st.info("Botta eşik %70 olarak belirlenmiştir. Eşik yükseldikçe cevap verilen tahminler daha doğru olur, "
            "ancak daha fazla kullanıcı uzmana yönlendirilir. Eşik bu iki durum arasında denge gözetilerek seçilmiştir.")

with t4:
    yanlis = np.where(~dogru)[0]
    st.write(f"Aşağıda modelin yanıldığı {len(yanlis)} test görseli yer almaktadır. Güveni en yüksek olan "
             "hatalar en tehlikelileridir, çünkü model bu görsellerde yanlış olduğu hâlde emindir.")
    sirala = st.radio("Sıralama", ["Güveni en yüksek", "Güveni en düşük"], horizontal=True)
    yanlis = yanlis[np.argsort(-guven[yanlis] if sirala == "Güveni en yüksek" else guven[yanlis])]
    if not veri_var():
        st.info("Görsellerin gösterilebilmesi için veri seti gereklidir (data/plantvillage).")
    else:
        for bas in range(0, min(len(yanlis), 24), 6):
            kolonlar = st.columns(6)
            for kolon, i in zip(kolonlar, yanlis[bas:bas + 6]):
                kolon.image(os.path.join(VERI_DIR, ts["dosyalar"][i]), use_container_width=True)
                kolon.markdown(f"<small>Gerçek: **{etiketler[y_true[i]]}**<br>Tahmin: {etiketler[y_pred[i]]}"
                               f"<br>Güven: %{guven[i]:.0f}</small>", unsafe_allow_html=True)

with t5:
    k1, k2 = st.columns(2)
    for kolon, dosya, baslik in ((k1, "ogrenme_egrisi_EfficientNetB0_38sinif.png", "Öğrenme eğrisi (eğitim / doğrulama)"),
                                 (k2, "confusion_matrix_EfficientNetB0_38sinif.png", "Karışıklık matrisi (Colab)")):
        yol = os.path.join(M38_DIR, dosya)
        if os.path.exists(yol):
            kolon.image(yol, caption=baslik, use_container_width=True)
    st.caption("Colab eğitiminin son aşamasında doğruluk eğitimde %98,88, doğrulamada %98,92, testte %99,02'dir. Üç değerin "
               "birbirine çok yakın olması modelin ezberlemediğini gösterir.")

gezinme(__file__)
