# Geç Yanıklık (Late Blight) — Phytophthora infestans

**Etken:** *Phytophthora infestans* (su küfü / oomycete — klasik anlamda mantar değildir,
ama pratikte "mantar hastalığı" gibi ele alınır).

## Belirtiler
- Yapraklarda düzensiz, sınırları belirsiz, su emmiş görünümlü **büyük** koyu yeşil/kahverengi
  lekeler.
- Nemli havada lekelerin alt yüzeyinde ince, beyazımsı bir küf tabakası (sporlanma) görülebilir.
- Çok hızlı yayılır — uygun koşullarda birkaç gün içinde tüm bitkiyi/parseli etkileyebilir.
- Gövde ve meyvede de kahverengi, sert lezyonlar oluşabilir.

## Uygun koşullar
- Serin-ılık (15–21°C) ve **yüksek nem** (>%90) veya uzun süreli yaprak ıslaklığı.
- Bulutlu, yağmurlu dönemlerde riski en yüksek hastalıklardan biridir.
- Bu proje kapsamındaki hava durumu modülü (`agent/weather.py`), yağış + nem verisine
  bakarak "mantar hastalığı riski"ni tahmin eder — bu risk özellikle geç yanıklık için anlamlıdır.

## Teknik not
Etken sürekli evrim geçiren bir patojendir — Avrupa'da 2022'de ortaya çıkan yeni bir soy
(EU43), yaygın kullanılan bir fungisit grubuna (mandipropamid) TAM DİRENÇLİ bulunmuştur.
Bu, tek bir ilaca güvenmenin neden riskli olduğunu ve entegre (kültürel + rotasyonlu ilaç
kullanımı) bir yaklaşımın neden önemli olduğunu gösterir. Avrupa'da patates geç yanıklığının
yıllık ekonomik maliyeti hâlâ 1 milyar Euro'nun üzerindedir.

## Karıştırılabileceği hastalıklar
- **Erken yanıklık** ile karıştırılabilir — erken yanıklıkta hedef/halka deseni ve daha yavaş
  ilerleme vardır; geç yanıklıkta desen yoktur ve yayılma çok daha hızlıdır.
- Ciddi su stresi/güneş yanığı da benzer solgunluk görüntüsü verebilir, ancak o durumda
  sporlanma (küflü görünüm) olmaz.

## Kültürel/biyolojik önlemler
- **ACİLİYET NOTU:** Geç yanıklık hızlı yayıldığı için şüpheli durumlarda gecikmeden bir
  ziraat mühendisine danışılması özellikle önemlidir.
- Etkilenen bitkileri/yaprakları hemen uzaklaştırıp imha edin (kompostlamayın, sporlar
  yayılabilir).
- Sulamayı azaltın, yaprak ıslaklığını minimize edin, hava sirkülasyonunu artırın.
- Sık yağış bekleniyorsa önleyici tedbirleri (havalandırma, seyreltme) önceden alın.
- Ürün rotasyonu ve sertifikalı/temiz fide kullanımı riski azaltır.

## Kaynak notu
Bu bilgiler genel, doğrulanmış agronomik kaynaklara dayanır; kesin teşhis ve ilaç/doz kararı
için ziraat mühendisine danışılmalıdır.
