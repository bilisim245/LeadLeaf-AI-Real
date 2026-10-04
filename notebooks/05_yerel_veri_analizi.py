"""
Veri setinin kalite analizi (sunum panosunun "Keşifsel Veri Analizi" sayfası için).

- Tüm 54.305 görselin boyutu (piksel)
- Birebir aynı olan görseller (MD5 özeti ile) — aynı fotoğraf iki kez var mı, farklı
  sınıflarda mı?
- Her sınıftan 60 görselin ortalama parlaklığı ve ortalama R/G/B değeri

Çalıştırma:  .venv\\Scripts\\python notebooks\\05_yerel_veri_analizi.py
Çıktı:       model/model_38sinif/veri_analizi.json
"""
import collections
import hashlib
import json
import os
import random

import numpy as np
from PIL import Image

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERI = os.path.join(KOK, "data", "plantvillage", "raw", "color")
CIKTI = os.path.join(KOK, "model", "model_38sinif", "veri_analizi.json")
ORNEK = 60

siniflar = sorted(os.listdir(VERI))
boyutlar = collections.Counter()
ozetler = collections.defaultdict(list)
renk = []

for sinif in siniflar:
    dosyalar = sorted(os.listdir(os.path.join(VERI, sinif)))
    for d in dosyalar:
        yol = os.path.join(VERI, sinif, d)
        with open(yol, "rb") as f:
            ozetler[hashlib.md5(f.read()).hexdigest()].append(f"{sinif}/{d}")
        with Image.open(yol) as im:
            boyutlar[f"{im.size[0]}x{im.size[1]}"] += 1
    for d in random.Random(42).sample(dosyalar, min(ORNEK, len(dosyalar))):
        with Image.open(os.path.join(VERI, sinif, d)) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
        renk.append({"sinif": sinif, "parlaklik": round(float(a.mean()), 1),
                     "R": round(float(a[..., 0].mean()), 1), "G": round(float(a[..., 1].mean()), 1),
                     "B": round(float(a[..., 2].mean()), 1)})
    print(f"{sinif}: {len(dosyalar)}", flush=True)

tekrarlar = [v for v in ozetler.values() if len(v) > 1]
farkli_sinif = [v for v in tekrarlar if len({p.split('/')[0] for p in v}) > 1]

# Sızıntı kontrolü: aynı görselin kopyaları farklı kümelere (eğitim/doğrulama/test) düşmüş mü?
with open(os.path.join(KOK, "model", "model_38sinif", "split_manifest.json"), encoding="utf-8") as f:
    bolme = json.load(f)["dosyalar"]
kume = {f"{c}/{d}": k for k in bolme for c, ds in bolme[k].items() for d in ds}
capraz = [[{"dosya": p, "kume": kume.get(p, "?")} for p in v] for v in tekrarlar
          if len({kume.get(p) for p in v}) > 1]
egitimde_kopyasi_olan_test = sum(1 for g in capraz for x in g if x["kume"] == "test"
                                 and any(y["kume"] == "train" for y in g))
sonuc = {
    "toplam_gorsel": sum(boyutlar.values()),
    "boyutlar": dict(boyutlar),
    "tekrar_grubu": len(tekrarlar),
    "tekrar_eden_fazla_kopya": sum(len(v) - 1 for v in tekrarlar),
    "farkli_sinifta_tekrar": len(farkli_sinif),
    "tekrar_ornekleri": tekrarlar,
    "farkli_kumede_tekrar": len(capraz),
    "farkli_kumede_ornekler": capraz,
    "egitimde_kopyasi_olan_test": egitimde_kopyasi_olan_test,
    "renk_ornegi": renk,
}
with open(CIKTI, "w", encoding="utf-8") as f:
    json.dump(sonuc, f, ensure_ascii=False, indent=1)
print({k: v for k, v in sonuc.items() if k not in ("renk_ornegi", "tekrar_ornekleri", "farkli_kumede_ornekler")})
