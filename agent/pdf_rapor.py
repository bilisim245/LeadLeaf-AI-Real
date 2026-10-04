"""
PDF rapor üretimi: n8n, Claude raporunu PDF'e çevirip Telegram'dan belge olarak gönderir.

Neden fpdf2: sistemde ekstra bir kurulum (Pango/Cairo gibi) gerektirmeyen, saf
Python bir kütüphane — Windows'ta sorunsuz çalışır. Türkçe karakterler (ğ, ş,
ı, ç, ö, ü) için sistemin Arial TTF fontu embed ediliyor; fpdf2'nin varsayılan
"core" fontları (Helvetica/Times) bu karakterleri desteklemiyor.
"""
from __future__ import annotations

import os

from fpdf import FPDF

# Windows'un kendi Arial fontu — Türkçe karakterleri tam destekliyor, ekstra
# font dosyası indirmeye/lisanslamaya gerek yok (bu proje sadece bu makinede
# demo edileceği için taşınabilirlik burada öncelik değil).
_FONT_REGULAR = r"C:\Windows\Fonts\arial.ttf"
_FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


_FONT_ADI = "Helvetica"


def _pdf_hazirla() -> FPDF:
    global _FONT_ADI
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    if os.path.exists(_FONT_REGULAR):
        pdf.add_font("Arial", "", _FONT_REGULAR)
        pdf.add_font("Arial", "B", _FONT_BOLD if os.path.exists(_FONT_BOLD) else _FONT_REGULAR)
        _FONT_ADI = "Arial"
    else:
        # Font bulunamazsa (farklı bir işletim sisteminde çalıştırılırsa) çökmek yerine
        # Türkçe karaktersiz bir çekirdek fonta düş — sistem yine de bir PDF üretir.
        _FONT_ADI = "Helvetica"
    pdf.set_font(_FONT_ADI, "", 11)
    return pdf


def _baslik(pdf: FPDF, metin: str) -> None:
    pdf.set_font(_FONT_ADI, "B", 13)
    pdf.multi_cell(0, 8, metin, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(_FONT_ADI, "", 11)
    pdf.ln(1)


def _govde(pdf: FPDF, metin: str) -> None:
    pdf.multi_cell(0, 6, metin, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _yuzde(x) -> str:
    return f"%{float(x):.1f}".replace(".", ",")


def sonuc_nasil_olustu(ilk3: list[dict] | None, bitki: str | None = None, bitki_uyumu=None,
                       uzmana_yonlendir: bool = False, rag_kullanildi: bool | None = None) -> list[str]:
    """Sonucun nasıl oluştuğunu modelin kendi çıktısından (Claude'a yazdırmadan) anlatan cümleler."""
    cumleler: list[str] = []
    # Bitki verildi ama o bitkinin sınıflarına düşen olasılık eşiğin (inference/app.py BITKI_UYUM_ESIGI
    # = %50) altında -> teşhis konmadı. `ilk3` filtresiz tahmindir; bunu ayrıca ve çelişkisiz anlat.
    if bitki and bitki_uyumu is not None and float(bitki_uyumu) < 50 and ilk3:
        bir = ilk3[0]
        cumleler += [
            f"Bitki bilgisi olmadan bakıldığında yaprak en çok \"{bir['sinif_tr']}\" sınıfına "
            f"benzetilmiştir ({_yuzde(bir['olasilik'])}).",
            f"Bitki {bitki} olarak belirtilmiştir; ancak {bitki} sınıflarına verilen toplam olasılık "
            f"yalnızca {_yuzde(bitki_uyumu)} düzeyindedir (eşik %50). Bu nedenle teşhis konulmamış ve "
            f"uzmana yönlendirilmiştir.",
            f"İki olasılık bulunmaktadır: fotoğraf başka bir bitkiye ait olabilir ya da yaprakta {bitki} "
            f"için sistemde tanımlı olmayan bir belirti bulunmaktadır.",
        ]
        if rag_kullanildi:
            cumleler.append("Genel önlem bilgileri proje bilgi tabanına dayanılarak yazılmıştır.")
        cumleler.append("Modelde 14 bitkiye ait 38 sınıf tanımlıdır ve model laboratuvar fotoğraflarıyla "
                        "eğitilmiştir; bu listede olmayan bir hastalık doğru tanınamaz.")
        return cumleler
    if ilk3:
        bir = ilk3[0]
        cumleler.append(f"Yaprak, model tarafından {_yuzde(bir['olasilik'])} olasılıkla "
                        f"\"{bir['sinif_tr']}\" olarak sınıflandırılmıştır.")
        if len(ilk3) > 1 and float(ilk3[1]["olasilik"]) >= 5:
            iki = ilk3[1]
            cumleler.append(f"İkinci en yakın olasılık {_yuzde(iki['olasilik'])} ile \"{iki['sinif_tr']}\" "
                            "sınıfına aittir. Bu iki sınıf arasında kesin bir ayrım yapılamamıştır; benzer "
                            "belirtiler gösterebilmektedirler.")
        else:
            cumleler.append("Diğer sınıfların olasılığı çok düşüktür; sonuç yüksek güvenle elde edilmiştir.")
    if bitki:
        ek = f" (bu bitkinin sınıflarına düşen toplam olasılık {_yuzde(bitki_uyumu)})" if bitki_uyumu is not None else ""
        cumleler.append(f"Bitki bilgisi ({bitki}) kullanılmıştır; tahmin yalnızca bu bitkinin sınıfları "
                        f"arasından yapılmıştır{ek}.")
    if uzmana_yonlendir:
        cumleler.append("Güven %70'in altında olduğu için sonuç kesin kabul edilmemiş ve uzmana yönlendirilmiştir.")
    if rag_kullanildi:
        cumleler.append("Neden ve önlem bilgileri, proje bilgi tabanındaki ilgili hastalık kaynağına dayanılarak yazılmıştır.")
    cumleler.append("Modelde 14 bitkiye ait 38 sınıf tanımlıdır ve model laboratuvar fotoğraflarıyla "
                    "eğitilmiştir; bu listede olmayan bir hastalık doğru tanınamaz.")
    return cumleler


def rapor_pdf_olustur(rapor: dict, hastalik_tr: str | None = None, tarih: str | None = None,
                      ilk3: list[dict] | None = None, bitki: str | None = None, bitki_uyumu=None,
                      rag_kullanildi: bool | None = None) -> bytes:
    """`rapor` = Claude'un ürettiği JSON (hastalik, guven, neden, aciklama, onlem, uzmana_yonlendir,
    uyari). `ilk3` = /predict çıktısındaki en olası 3 sınıf; verilmezse rapor["ilk3"] denenir
    (n8n rapora ekleyip /generate-pdf'e gönderebilir)."""
    ilk3 = ilk3 or rapor.get("ilk3")
    pdf = _pdf_hazirla()

    _baslik(pdf, "LeadLeaf AI — Bitki Hastalığı Ön Değerlendirme Raporu")
    if tarih:
        _govde(pdf, f"Tarih: {tarih}")

    _baslik(pdf, f"Tespit: {rapor.get('hastalik', hastalik_tr or '-')}")
    _govde(pdf, f"Model güveni: %{str(rapor.get('guven', '-')).replace('.', ',')}")

    if ilk3:
        _baslik(pdf, "Modelin Diğer Yakın Olasılıkları")
        for sira, it in enumerate(ilk3, start=1):
            pdf.multi_cell(0, 6, f"{sira}. {it['sinif_tr']} — {_yuzde(it['olasilik'])}",
                           new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    _baslik(pdf, "Bu Sonuç Nasıl Oluştu?")
    for cumle in sonuc_nasil_olustu(ilk3, bitki, bitki_uyumu, bool(rapor.get("uzmana_yonlendir")),
                                    rag_kullanildi):
        pdf.multi_cell(0, 6, f"- {cumle}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if rapor.get("neden"):
        _baslik(pdf, "Hastalığın Nedeni")
        _govde(pdf, rapor["neden"])

    if rapor.get("aciklama"):
        _baslik(pdf, "Değerlendirme")
        _govde(pdf, rapor["aciklama"])

    onlem = rapor.get("onlem") or []
    if onlem:
        _baslik(pdf, "Yayılmayı Azaltmak İçin Yapılabilecekler")
        for madde in onlem:
            pdf.multi_cell(0, 6, f"- {madde}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    if rapor.get("uzmana_yonlendir"):
        _baslik(pdf, "Uzmana Danışın")
        _govde(pdf, "Bu sonuç kesin değil, bir ziraat mühendisine danışmanızı öneririz.")

    if rapor.get("uyari"):
        pdf.set_font(_FONT_ADI, "", 9)
        _govde(pdf, rapor["uyari"])
        pdf.set_font(_FONT_ADI, "", 11)

    return bytes(pdf.output())
