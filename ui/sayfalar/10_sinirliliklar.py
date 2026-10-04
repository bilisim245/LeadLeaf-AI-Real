import streamlit as st

from ortak import gezinme

st.title("Sınırlılıklar ve Sonraki Adımlar")

st.subheader("Gerçek Bir Test: Şeftali Yaprak Kıvırcıklığı")
c1, c2, c3 = st.columns(3)
c1.metric("Modelin cevabı", "Domates geç yanıklığı", "%89,1 güven", delta_color="inverse")
c2.metric("Şeftali sınıflarına verdiği olasılık", "%0,1")
c3.metric("Bitki filtresiyle", "Tanımlı olmayan belirti", "uzmana yönlendirilir", delta_color="off")
st.write("Bu hastalık veri setinde bulunmamaktadır. Model \"bilmiyorum\" diyemediği için görüntü, bilinen "
         "sınıflar içinde en çok benzediği sınıfa yüksek güvenle atanmıştır. Çözüm olarak bitki bilgisi "
         "eklenmiştir: Bitki biliniyorsa tahmin yalnızca o bitkinin sınıfları arasından yapılır; hiçbiri "
         "uymuyorsa teşhis konulmaz ve kullanıcı uzmana yönlendirilir.")

st.subheader("Bilinen Sınırlılıklar")
st.dataframe({
    "Konu": ["Laboratuvar verisi", "Eksik hastalıklar", "Bitki bilgisi", "Rapor değerlendirmesi", "Altyapı",
             "Tekrar eden görseller", "İlaç dozu (bilinçli karar)"],
    "Durum": [
        "Görseller sade arka planda çekilmiştir. Grad-CAM'de bazı örneklerde arka plana odaklanıldığı görülmüştür. "
        "Tarladan çekilen fotoğraflarda başarı ayrıca ölçülmelidir.",
        "Üzüm mildiyösü, şeftali yaprak kıvırcıklığı gibi yaygın hastalıklar ve zeytin, fındık gibi bitkiler veri setinde bulunmamaktadır.",
        "Bitki filtresi şu an kullanıcının bitki adını belirtmesine bağlıdır.",
        "Claude'un raporları bir ziraat mühendisi tarafından sistematik olarak incelenmemiştir.",
        "Sistem tek bir bilgisayarda çalışmaktadır; bir sunucuya taşınmalıdır.",
        "3 test görselinin birebir aynısı eğitim kümesinde bulunmaktadır (etkisi %0,04). Tekrarlar bölmeden önce temizlenmelidir.",
        "Görevde istenen doz ve bekleme süresi verilmemektedir; karar ruhsatlı ziraat mühendisine bırakılmıştır.",
    ],
}, hide_index=True, use_container_width=True)

st.subheader("Sonraki Adımlar")
k1, k2, k3 = st.columns(3)
k1.markdown("**Kısa Vade**\n- Telegram botunda konuma göre hava durumu değerlendirmesi\n- Bitkinin yapraktan otomatik tanınması\n- Tarladan toplanmış etiketli test seti\n- Raporların uzmanla değerlendirilmesi")
k2.markdown("**Orta Vade**\n- Türkiye'de yaygın hastalıkların eklenmesi\n- \"Bilmiyorum\" diyebilen bir katman\n- Kullanıcı geri bildirimiyle yeniden eğitim")
k3.markdown("**Uzun Vade**\n- Konum ve mevsim bilgisinin kullanılması\n- Sistemin sunucuya taşınması\n- Mobil kullanım")

gezinme(__file__)
