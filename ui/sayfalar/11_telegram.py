import streamlit as st

from ortak import TELEGRAM_BOT, gezinme, telegram_qr

st.title("Telegram Botu")
st.markdown("#### Çiftçinin Kullandığı Arayüz: @LeadLeafAI_bot")

qr, aciklama = st.columns([1, 2])
telegram_qr(qr, genislik=320, aciklama=False)  # projektörde salondan okutulabilsin diye büyük
aciklama.markdown(f"""
**Nasıl Denenir?**
1. QR kod telefon kamerasıyla okutulur ya da [t.me/LeadLeafAI_bot]({TELEGRAM_BOT}) açılır.
2. Bota bir yaprak fotoğrafı gönderilir.
3. Model tahmini ve bilgi tabanına dayalı Claude raporu kısa süre içinde yanıt olarak gelir.

Aynı model ve rapor akışı burada n8n üzerinden çalışır; çiftçinin ek bir uygulama kurması gerekmez.
""")

gezinme(__file__)
