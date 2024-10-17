# Video Localization Automation

Bu proje, video yerelleştirme sürecini otomatikleştirmek için tasarlanmış bir Streamlit uygulamasıdır. Uygulama, bir temel video üzerine özelleştirilmiş resimler ve başlıklar ekleyerek çoklu video versiyonları oluşturmanıza olanak tanır.

https://videolocalization.streamlit.app/#video-localization-automation
Bu linkten projenin live versiyonuna ulaşabilirsiniz.

## Proje Amacı

Bu uygulamanın ana amacı, video içerik üreticilerinin ve pazarlama ekiplerinin farklı bölgeler için video içeriklerini hızlı ve etkili bir şekilde uyarlamalarına yardımcı olmaktır. Uygulama şunları yapabilir:

- Temel bir videoya özelleştirilmiş resimler ekler
- Videoya yerelleştirilmiş başlıklar ekler

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız vardır:

- Python 3.7+
- OpenCV (cv2)
- NumPy
- MoviePy
- Pillow (PIL)
- Streamlit
- Pandas
- pillow_avif

Bu gereksinimleri şu komutla kurabilirsiniz:

```
pip install opencv-python numpy moviepy pillow streamlit pandas pillow_avif
```

## Kullanım

1. Uygulamayı başlatmak için terminal veya komut istemcisinde şu komutu çalıştırın:

   ```
   streamlit run script.py
   ```

2. Uygulama arayüzünde:
   - Temel videoyu yükleyin
   - Overlay görüntülerini yükleyin
   - Başlıkları içeren CSV dosyasını yükleyin
   - Çıktı dizinini belirtin
   - Font dosyasını yükleyin
   - Font boyutunu ayarlayın

3. "Process Video" butonuna tıklayın

4. İşlem tamamlandığında, her bir yerelleştirilmiş video gösterilecek ve kaydedilecektir.

## Dosya Gereksinimleri

- **Temel Video**: MP4, MOV veya AVI formatında
- **Overlay Görüntüleri**: PNG, JPG, JPEG, WEBP veya AVIF formatında
- **Başlıklar CSV**: Başlıkları içeren CSV dosyası
- **Font Dosyası**: TTF formatında

## Nasıl Çalışır

1. Uygulama, temel videodaki yeşil dairesel alanları tespit eder.
2. Bu alanlara, yüklenen resimleri yerleştirir.
3. Videonun üst kısmına, CSV dosyasından okunan başlıkları ekler.
4. Her bir overlay ve başlık kombinasyonu için ayrı bir video oluşturur.

## Notlar

- Yeşil daire tespiti, HSV renk uzayında belirli bir yeşil ton aralığını kullanır.
- İşlem süresi, video uzunluğuna ve bilgisayarınızın özelliklerine bağlı olarak değişebilir.