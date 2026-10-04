import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage

from ortak import (KOK, etiket, gezinme, kod_parcasi, nasil_okunur, ornek_dosyalar, sinif_tablosu, uretim_modeli,
                   veri_var)

st.title("CNN ve Transfer Learning")

siniflar = sinif_tablosu().sort_values("Etiket")["Sınıf"].tolist()


def gorsel_sec(anahtar: str, varsayilan: str = "Tomato___Early_blight") -> Image.Image:
    if veri_var():
        sinif = st.selectbox("Yaprak", siniflar, index=siniflar.index(varsayilan), format_func=etiket, key=anahtar)
        return Image.open(ornek_dosyalar(sinif, 1, 3)[0]).convert("RGB")
    return Image.open(os.path.join(KOK, "model", "demo_images", "Tomato___Early_blight.jpg")).convert("RGB")


t1, t2, t3, t4 = st.tabs(["Evrişim (Convolution)", "Neden Transfer Learning?", "Projedeki Model", "Katmanlar Ne Görüyor?"])

with t1:
    st.write("CNN'in temel işlemi evrişimdir. Küçük bir filtre (3×3'lük bir sayı tablosu) görselin üzerinde "
             "kaydırılır ve her konumda filtredeki sayılar altındaki piksellerle çarpılıp toplanır. Böylece "
             "filtrenin aradığı desenin bulunduğu bölgeler parlak görünür. Aşağıda hazır bir filtre seçilerek ya "
             "da sayılar değiştirilerek sonuç izlenebilir.")
    hazir = {
        "Dikey kenar": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "Yatay kenar": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "Keskinleştirme": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "Bulanıklaştırma": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    }
    sol, orta, sag = st.columns([1, 1, 1])
    with sol:
        img = gorsel_sec("evrisim")
        secim = st.radio("Hazır filtre", list(hazir), horizontal=False)
        filtre = st.data_editor(pd.DataFrame(hazir[secim], columns=["1", "2", "3"]),
                                hide_index=True, key=f"filtre_{secim}", use_container_width=True)
    gri = np.asarray(img.convert("L"), dtype=np.float32)
    k = filtre.to_numpy(dtype=np.float32)
    if secim == "Bulanıklaştırma":
        k = k / max(k.sum(), 1)
    sonuc = ndimage.convolve(gri, k, mode="reflect")
    if secim in ("Dikey kenar", "Yatay kenar"):
        sonuc = np.abs(sonuc)
    sonuc = (sonuc - sonuc.min()) / (np.ptp(sonuc) + 1e-6) * 255
    orta.image(img, caption="Orijinal", use_container_width=True)
    sag.image(sonuc.astype(np.uint8), caption=f"Filtre sonrası: {secim}", use_container_width=True)
    st.caption("Gerçek bir CNN'de bu filtreler elle yazılmaz; model eğitim sırasında bunları kendisi öğrenir. "
               "İlk katmanlar kenar ve renk geçişi gibi basit desenleri, son katmanlar ise leke ve doku gibi "
               "daha karmaşık desenleri öğrenir.")

    nasil_okunur(
        "Ortada seçilen yaprak, sağda ise bu yaprağa 3×3'lük bir filtre uygulanmış hâli görülmektedir.",
        "Sağdaki görselde parlak yerler filtrenin aradığı desenin bulunduğu yerlerdir. Dikey kenar filtresi dikey çizgileri, yatay kenar filtresi yatay çizgileri parlatır.",
        "Küçük bir filtre görselin üzerinde gezdirilerek belirli desenler öne çıkarılır. CNN bu işlemi çok sayıda filtreyle ve art arda gelen katmanlarda tekrarlayarak görüntüyü sınıflandırır.",
        baslik="Ne anlama geliyor?")
with t2:
    st.write("Model için iki yol vardı: sıfırdan bir CNN tasarlayıp eğitmek ya da ImageNet'te önceden "
             "eğitilmiş bir modeli bu veriye uyarlamak (transfer learning). Projede ikinci yol tercih edilmiştir.")
    # İlk sütun satır başlığı (index) yapılır; yoksa tabloda pandas'ın 0, 1, 2… satır numaraları görünür
    st.table(pd.DataFrame({
        "Özellik": ["Başlangıç noktası", "Gereken veri", "Eğitim süresi", "Ezberleme riski",
                    "Bu projeye uygunluğu"],
        "Sıfırdan CNN": ["Rastgele ağırlıklar, öğrenilmiş bilgi yok", "Çok fazla", "Uzun", "Yüksek",
                         "Ücretsiz Colab GPU'su ve kısıtlı süreyle zor"],
        "Transfer learning": ["ImageNet'te (1,28 milyon görsel, 1000 sınıf) eğitilmiş ağırlıklar",
                              "Daha azı yeterlidir", "Kısa", "Daha düşük",
                              "Uygun: kenar, doku, şekil bilgisi hazır gelir"],
    }).set_index("Özellik"))
    st.write("Yaprak da sonuçta bir görseldir. Kenar, doku ve renk geçişi gibi temel özellikler ImageNet'te "
             "öğrenildiği için yaprakta da kullanılabilmektedir. Bu projede önce yalnızca en sondaki "
             "sınıflandırma katmanı eğitilmiş, ardından gövdenin son katmanları da ince ayarla 38 sınıfa "
             "uyarlanmıştır.")
    st.info("Not: Bu projede sıfırdan eğitilen bir CNN ile ayrı bir karşılaştırma yapılmamıştır; bu "
            "karşılaştırma sonraki adımlardan biridir. PlantVillage veri setini tanıtan çalışmada transfer learning "
            "her deneyde sıfırdan eğitimden daha iyi sonuç vermiştir: Ortalama F1, AlexNet'te 0,9782'den 0,9927'ye, "
            "GoogLeNet'te 0,9836'dan 0,9934'e yükselmiştir (Mohanty, Hughes ve Salathé, 2016; renkli görüntüler, "
            "%80 eğitim / %20 test). Fark yaklaşık 1–1,5 puandır; hata (1 − F1) ise AlexNet'te "
            "%2,1'den %0,7'ye, yani yaklaşık üçte birine inmektedir.")
    st.caption("Kaynak: Mohanty, S. P., Hughes, D. P. ve Salathé, M. (2016). Using Deep Learning for "
               "Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419, Tablo 1. "
               "https://doi.org/10.3389/fpls.2016.01419")

with t3:
    model = uretim_modeli()
    govde = model.get_layer("efficientnetb0")
    kafa_param = model.count_params() - govde.count_params()
    c1, c2, c3 = st.columns(3)
    c1.metric("Gövde (EfficientNetB0) katmanı", len(govde.layers))
    c2.metric("Gövde parametresi", f"{govde.count_params():,}".replace(",", "."), "ImageNet'ten hazır", delta_color="off")
    c3.metric("Eklenen sınıflandırma katmanı", f"{kafa_param:,}".replace(",", "."), "38 sınıf için", delta_color="off")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#EEF2F8", color="#1B3A6B", fontname="Helvetica", fontsize=11];
      a [label="Girdi\\n224×224×3"];
      b [label="Veri artırma\\n(çevirme, döndürme,\\nyakınlaştırma, kontrast)\\nyalnızca eğitimde"];
      c [label="EfficientNetB0 gövdesi\\n238 katman\\nImageNet ağırlıkları", fillcolor="#1B3A6B", fontcolor="white"];
      d [label="Global Average\\nPooling\\n7×7×1280 → 1280"];
      e [label="Dropout 0,2"];
      f [label="Dense 38\\nsoftmax\\n(38 olasılık)"];
      a -> b -> c -> d -> e -> f;
    }
    """, use_container_width=True)
    st.write("Softmax ile her sınıf için bir olasılık üretilir; olasılıkların toplamı 1'dir. En yüksek "
             "olasılığa sahip sınıf tahmin, o olasılık da **güven** değeri olarak alınır.")
    with st.expander("Kod: Modelin Kurulduğu Bölüm (notebooks/03_efficientnetb0_38_sinif.py)"):
        st.code(kod_parcasi("notebooks/03_efficientnetb0_38_sinif.py", "inputs = keras.Input", "def parametre_ozeti"),
                language="python")

with t4:
    st.write("Aşağıda bir yaprak modele verilmiş ve gövdenin farklı derinliklerindeki çıktılar gösterilmiştir. "
             "Her küçük kare bir filtrenin tepkisidir; parlak yerler, filtrenin aradığı deseni bulduğu bölgelerdir.")
    img4 = gorsel_sec("katman", "Tomato___Late_blight")
    katmanlar = {"Başta (stem)": "stem_activation", "Ortada (blok 3)": "block3b_activation",
                 "Derinde (blok 6)": "block6d_activation"}

    @st.cache_resource
    def ara_model():
        import keras
        g = uretim_modeli().get_layer("efficientnetb0")
        return keras.Model(g.input, [g.get_layer(n).output for n in katmanlar.values()])

    x = np.expand_dims(np.asarray(img4.resize((224, 224)), dtype=np.float32), 0)
    ciktilar = ara_model().predict(x, verbose=0)
    st.image(img4, width=180)
    for (ad, katman), cikti in zip(katmanlar.items(), ciktilar):
        cikti = cikti[0]
        kanallar = np.argsort(cikti.mean(axis=(0, 1)))[::-1][:8]
        st.markdown(f"**{ad}** · `{katman}` · çıktı boyutu {cikti.shape[0]}×{cikti.shape[1]}, {cikti.shape[2]} filtre")
        kolonlar = st.columns(8)
        for kolon, kanal in zip(kolonlar, kanallar):
            h = cikti[..., kanal]
            h = (h - h.min()) / (np.ptp(h) + 1e-6)
            kolon.image(Image.fromarray((h * 255).astype(np.uint8)).resize((112, 112), Image.NEAREST),
                        use_container_width=True)
    st.caption("Başta çıktı büyüktür (112×112) ve yaprağın şekli seçilebilmektedir. Derine inildikçe çıktı "
               "küçülür (14×14), filtre sayısı artar ve görüntü insan gözüyle zor yorumlanır. Bu katmanlarda şekil "
               "yerine leke ve doku gibi soyut özellikler öne çıkar.")
    nasil_okunur(
        "Aynı yaprağın modelin başındaki, ortasındaki ve derinlerindeki katmanlarda nasıl göründüğü gösterilmiştir. Her kare bir filtrenin çıktısıdır.",
        "Parlak yerler filtrenin bir şey bulduğu yerlerdir. Aşağı inildikçe kareler bulanıklaşır, çünkü çözünürlük düşer (112 → 28 → 14).",
        "Model ilk katmanlarda yaprağın kenarını ve şeklini, derin katmanlarda leke ve doku gibi hastalığa özgü desenleri yakalamaktadır. Karar en derindeki bu desenlere göre verilir.")

gezinme(__file__)
