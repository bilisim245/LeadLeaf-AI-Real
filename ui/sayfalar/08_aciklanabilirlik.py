from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ortak import VERI_DIR, etiket, gezinme, nasil_okunur, test_sonuclari, uretim_modeli, veri_var

st.title("Model Nereye Bakıyor?")
st.write("Grad-CAM yöntemiyle, modelin kararında yaprağın hangi bölgesinin etkili olduğu gösterilmektedir. "
         "Kırmızı bölgeler kararı en çok etkileyen yerlerdir. Model gerçekten lekelere bakıyorsa doğru "
         "özelliği öğrenmiştir; arka plana bakıyorsa bir sorun olabilir.")


def grad_cam(img: Image.Image, sinif_no: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf
    model = uretim_modeli()
    govde = model.get_layer("efficientnetb0")
    gap, dense = model.get_layer("global_average_pooling2d"), model.get_layer("dense")
    x = tf.convert_to_tensor(np.asarray(img.resize((224, 224)), dtype=np.float32)[None])
    with tf.GradientTape() as tape:
        harita = govde(x, training=False)          # 7x7x1280 son evrişim çıktısı
        tape.watch(harita)
        olasilik = dense(gap(harita))
        hedef = int(tf.argmax(olasilik[0])) if sinif_no is None else sinif_no
        skor = olasilik[:, hedef]
    gradyan = tape.gradient(skor, harita)
    agirlik = tf.reduce_mean(gradyan, axis=(1, 2))     # her filtrenin önemi
    cam = tf.nn.relu(tf.reduce_sum(harita * agirlik[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, olasilik[0].numpy()


def ust_uste(img: Image.Image, cam: np.ndarray) -> Image.Image:
    buyuk = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255
    renkli = (matplotlib.colormaps["jet"](buyuk)[..., :3] * 255).astype(np.uint8)
    return Image.blend(img.resize((224, 224)), Image.fromarray(renkli), alpha=0.45)


ts = test_sonuclari()
kaynak = st.radio("Görsel", ["Test kümesinden", "Yüklenen fotoğraftan"], horizontal=True)
img = None
if kaynak == "Test kümesinden":
    if ts is None or not veri_var():
        st.info("Bu bölüm için test sonuçları ve veri seti gereklidir.")
    else:
        dogru = ts["y_prob"].argmax(1) == ts["y_true"]
        k1, k2 = st.columns(2)
        grup = k1.radio("Hangi tahminler?", ["Doğru tahminler", "Yanlış tahminler"], horizontal=True)
        adaylar = np.where(dogru if grup == "Doğru tahminler" else ~dogru)[0]
        siniflar = sorted({ts["y_true"][i] for i in adaylar}, key=lambda i: etiket(ts["siniflar"][i]))
        sinif = k2.selectbox("Gerçek sınıf", siniflar, format_func=lambda i: etiket(ts["siniflar"][i]),
                             key=f"gc_sinif_{grup}")
        secenekler = [i for i in adaylar if ts["y_true"][i] == sinif]
        sira = (st.slider("Örnek", 1, len(secenekler), 1, key=f"gc_ornek_{grup}_{sinif}")
                if len(secenekler) > 1 else 1)
        i = secenekler[sira - 1]
        img = Image.open(os.path.join(VERI_DIR, ts["dosyalar"][i])).convert("RGB")
        st.caption(f"Gerçek sınıf: **{etiket(ts['siniflar'][ts['y_true'][i]])}**")
else:
    yuklenen = st.file_uploader("Yaprak fotoğrafı", type=["jpg", "jpeg", "png"])
    if yuklenen:
        img = Image.open(yuklenen).convert("RGB")

if img is not None:
    cam, olasilik = grad_cam(img)
    k1, k2, k3 = st.columns([1, 1, 1.2])
    k1.image(img.resize((224, 224)), caption="Görsel", use_container_width=True)
    k2.image(ust_uste(img, cam), caption="Grad-CAM", use_container_width=True)
    siniflar_tum = ts["siniflar"] if ts else []
    ilk3 = np.argsort(olasilik)[::-1][:3]
    with k3:
        st.markdown("**Modelin İlk 3 Tahmini**")
        for j in ilk3:
            ad = etiket(siniflar_tum[j]) if siniflar_tum else str(j)
            st.progress(float(olasilik[j]), text=f"{ad}: %{olasilik[j] * 100:.1f}".replace(".", ","))
    st.caption("Isı haritası 7×7'lik son evrişim çıktısından hesaplanıp görsel boyutuna büyütülür. Bu nedenle "
               "haritada ayrıntılı bir sınır yerine kaba bir bölge görülür.")
    nasil_okunur(
        "Solda yaprak, ortada modelin karar verirken en çok baktığı bölgeleri gösteren ısı haritası, sağda ise modelin ilk 3 tahmini yer almaktadır.",
        "Kırmızı ve sarı bölgeler kararı en çok etkileyen, mavi bölgeler en az etkileyen yerlerdir.",
        "Kırmızı bölge lekenin üzerindeyse model doğru bölgeye bakmaktadır. Kırmızı bölge arka planda ya da yaprak sapındaysa model doğru tahmin etse bile bir kısa yol öğrenmiş olabilir. Bu durum, tarla fotoğraflarında başarının düşebileceğine işaret eder.")

st.info("**Bulgu:** Çoğu örnekte modelin lekelere odaklandığı görülmektedir (ör. domates geç yanıklığı, "
        "üzüm esca). Ancak bazı örneklerde, tahmin doğru olduğu hâlde, yaprak sapına ya da kenardaki arka "
        "plana odaklanıldığı görülmüştür (ör. mısır pası, elma kara çürüklüğü). PlantVillage'da aynı sınıfın "
        "fotoğrafları benzer koşullarda çekildiğinden arka plan gibi \"kısa yollar\" da öğrenilebilmektedir; "
        "literatürde bu veri setinde arka plan yanlılığı raporlanmıştır. Bu durum, tarla fotoğraflarında "
        "başarının düşebileceğine dair somut bir işarettir.")

gezinme(__file__)
