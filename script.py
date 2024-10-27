import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import os
import csv
import streamlit as st
import pandas as pd
import pillow_avif
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1920, 1080))
display.start()



#Yeşil daireyi tespit et
def detect_green_circle(frame):
    # BGR görüntüsünü HSV renk uzayına dönüştürür
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yeşil renk için alt ve üst HSV sınırlarını tanımlar
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Belirlenen yeşil sınırlar arasındaki pikselleri maskeleyen bir maske oluşturur
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Yeşil maskeden dış hatları (contours) bulur
    contours, _ = cv2.findContours(
        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Eğer dış hatlar bulunduysa
    if contours:
        # En büyük dış hattı seçer (maksimum alanlı olan)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Bu dış hat etrafında en küçük çevreleyen çemberi bulur
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Çemberin merkez koordinatlarını ve yarıçapını tamsayıya çevirir
        center = (int(x), int(y))
        radius = int(radius)
        
        # Yeşil çemberin bulunduğunu yazdırır
        print(f"Detected green circle: center={center}, radius={radius}")
        
        # Merkez ve yarıçapı döndürür
        return center, radius
    
    # Eğer yeşil çember bulunmazsa uyarı mesajı verir
    print("No green circle detected")
    
    # Merkez ve yarıçap bulunamazsa None döndürür
    return None, None


#Görüntüyü oku ve numpy dizisine dönüştür
def read_image(image):
    try:
        img = Image.open(image) # Görüntü dosyasını açar
        img = img.convert("RGBA")# Görüntüyü RGBA (Red, Green, Blue, Alpha) formatına dönüştürür
        np_img = np.array(img) # Görüntüyü numpy dizisine dönüştürür
        print(f"Successfully read image: {image.name}") 
        print(f"Image shape: {np_img.shape}")
        return np_img # Dönüştürülen numpy dizisini döndürür
    except Exception as e:  # Görüntü okunurken hata meydana gelirse hata mesajı verir
        print(f"Error reading {image.name}: {e}")
        return None


def create_text_clip(text, size, font_size, color, font_path):
    try:
        # Verilen font dosyasını ve boyutunu kullanarak fontu yükler
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Eğer font dosyası bulunamaz veya okunamazsa hata mesajı verir ve varsayılan fontu kullanır
        print(f"Error: Font file not found or couldn't be read: {font_path}")
        print("Using default font.")
        font = ImageFont.load_default()

    # RGBA (Red, Green, Blue, Alpha) formatında saydam bir görüntü oluşturur
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    
    # Görüntüye çizim yapmak için bir ImageDraw objesi oluşturur
    draw = ImageDraw.Draw(img)

    # Metni satırlara bölmek için kullanılacak liste
    lines = []
    
    # Metni kelimelere böler
    words = text.split()
    
    # İlk kelimeyi satıra ekler
    current_line = words[0]

    # Diğer kelimeleri satıra eklemeye çalışır
    for word in words[1:]:
        test_line = f"{current_line} {word}"
        
        # Kelimelerin bulunduğu satırın genişliğini ölçer
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]
        
        # Eğer satır genişliği, görüntü genişliğinin %90'ını aşmazsa kelimeyi mevcut satıra ekler
        if width <= size[0] * 0.9:
            current_line = test_line
        else:
            # Aşarsa mevcut satırı listeye ekler ve yeni bir satır başlatır
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    # Metnin toplam yüksekliğini hesaplar
    total_height = sum(
        draw.textbbox((0, 0), line, font=font)[3]
        - draw.textbbox((0, 0), line, font=font)[1]
        for line in lines
    )
    
    # Metni görüntü içinde dikey olarak ortalamak için başlangıç y konumunu hesaplar
    y = (size[1] - total_height) / 2

    # Her satırı görüntüye çizer
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Her satır için siyah kontur (outline) ekler
        outline_color = (0, 0, 0, 255)
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                # Kontur çizimlerini yapar
                draw.text(
                    ((size[0] - width) / 2 + dx, y + dy),
                    line,
                    font=font,
                    fill=outline_color,
                )
        
        # Metni çizim alanına belirtilen renk ile çizer
        draw.text(((size[0] - width) / 2, y), line, font=font, fill=color)
        
        # Yüksekliği bir sonraki satır için günceller
        y += height

    # Sonuç olarak numpy dizisine dönüştürülmüş RGBA görüntüsünü döner
    return np.array(img)


def create_circular_mask(h, w, center, radius):
    # Y ve X koordinatlarını oluşturur
    Y, X = np.ogrid[:h, :w]

    # Her pikselin, belirtilen merkez noktasından uzaklığını hesaplar
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    # Merkeze olan mesafesi yarıçapdan küçük veya eşit olan pikselleri seçer
    mask = dist_from_center <= radius

    return mask


def process_frame(
    frame, overlay_image, title, font_path, font_size, frame_number, total_frames
):
    # Orijinal kareyi değiştirmemek için kopyasını oluştur
    frame = frame.copy()

    # Karede yeşil bir çember tespit etmeye çalış
    center, radius = detect_green_circle(frame)

    # Eğer yeşil çember bulunduysa ve yarıçapı sıfırdan büyükse overlay'i uygula
    if center is not None and radius > 0:
        # Overlay görüntüsünü çember boyutuna göre yeniden boyutlandır
        overlay_resized = cv2.resize(overlay_image, (2 * radius, 2 * radius))
        print(f"Resized overlay shape: {overlay_resized.shape}")

        # Çember maskesi oluştur
        circular_mask = create_circular_mask(
            2 * radius, 2 * radius, (radius, radius), radius
        )

        # Çember maskesini RGBA formatına genişlet
        circular_mask_4ch = np.dstack([circular_mask] * 4)

        # Overlay'i çember maskesiyle birleştir (çemberin dışında kalan kısımlar şeffaf olacak)
        masked_overlay = overlay_resized * circular_mask_4ch
        print(f"Masked overlay shape: {masked_overlay.shape}")

        # Çemberin merkez koordinatlarını al
        x, y = center

        # Overlay'in kareye uygulanacağı alanı hesapla (karenin sınırlarını aşmayacak şekilde)
        x_start = max(0, x - radius)
        x_end = min(frame.shape[1], x + radius)
        y_start = max(0, y - radius)
        y_end = min(frame.shape[0], y + radius)

        # Overlay'in uygulanacağı bölgeyi kes
        overlay_section = masked_overlay[0 : (y_end - y_start), 0 : (x_end - x_start)]
        print(f"Overlay section shape: {overlay_section.shape}")

        # Overlay'i ve kareyi alfa karıştırmasıyla birleştir
        alpha_s = overlay_section[:, :, 3] / 255.0  # Overlay'in alfa kanalı
        alpha_l = 1.0 - alpha_s  # Orijinal karenin alfa değeri

        # RGB kanallarını karıştır
        for c in range(3):
            frame[y_start:y_end, x_start:x_end, c] = (
                alpha_s * overlay_section[:, :, c]
                + alpha_l * frame[y_start:y_end, x_start:x_end, c]
            )

        print(
            f"Frame updated with overlay at position: ({x_start}, {y_start}) to ({x_end}, {y_end})"
        )
    else:
        # Eğer yeşil çember tespit edilmediyse overlay'i atla
        print("No green circle detected, skipping overlay")

    # Başlık ekle
    # Başlık görüntüsünü oluştur
    title_img = create_text_clip(
        title,
        (frame.shape[1], int(frame.shape[0] * 0.2)),  # Başlık boyutu kareye göre ayarlandı
        font_size,
        (255, 255, 255, 255),  # Beyaz renkte başlık
        font_path,
    )

    # Başlığın dikey pozisyonunu ayarla (üstten 20 piksel boşluk)
    title_y = 20

    # Başlık için alfa maskesini çıkar
    title_mask = title_img[:, :, 3] / 255.0
    title_mask = np.dstack([title_mask] * 3)  # RGB için maske genişletme
    title_rgb = title_img[:, :, :3]  # Başlık RGB değerlerini al

    # Başlığı kareye alfa karıştırmasıyla uygula
    frame[title_y : title_y + title_img.shape[0], :] = (
        frame[title_y : title_y + title_img.shape[0], :] * (1 - title_mask)
        + title_rgb * title_mask
    )

    # Güncellenmiş kareyi döndür
    return frame



def process_video(
    base_video_path, overlay_images, titles, output_dir, font_path, font_size
):
    # Videoyu yükle
    video = VideoFileClip(base_video_path)
    total_frames = int(video.fps * video.duration)  # Toplam kare sayısını hesapla

    # Overlay ve başlık listelerini en kısa olanın uzunluğuna göre kısalt
    min_length = min(len(overlay_images), len(titles))
    overlay_images = overlay_images[:min_length]
    titles = titles[:min_length]

    # Çıktı videolarının listesini oluştur
    output_videos = []

    # Her bir overlay görüntüsü ve başlık için video işle
    for i, (overlay_image, title_text) in enumerate(zip(overlay_images, titles)):
        print(f"Processing video {i + 1} with overlay image.")

        # Her kareyi işlemek için fonksiyon tanımla
        def process_frame_with_counter(img):
            nonlocal frame_number  # frame_number değişkenini dış fonksiyondan al
            processed = process_frame(
                img,
                overlay_image,
                title_text,
                font_path,
                font_size,
                frame_number,
                total_frames,
            )
            frame_number += 1  # Her kare işlendiğinde frame_number'ı arttır
            return processed

        frame_number = 0  # Başlangıç kare numarasını sıfırla
        processed_video = video.fl_image(process_frame_with_counter)  # Videonun her karesini işle

        # Çıktı videosu için dosya yolu oluştur
        output_path = os.path.join(output_dir, f"localized_video_{i + 1}.mp4")
        processed_video.write_videofile(output_path, codec="libx264")  # Videoyu kaydet
        output_videos.append(output_path)  # Çıktı dosya yolunu listeye ekle

        print(f"Processed and saved video: {output_path}")

    # İşlenmiş videoların listesini döndür
    return output_videos


#CSV'den başlıkları oku
def read_titles_from_csv(title_csv):
    titles = []
    # 'title_csv' UploadedFile nesnesi olduğundan, içeriği bir DataFrame olarak okuyalım
    df = pd.read_csv(title_csv)
    titles = df.iloc[:, 0].tolist()  # İlk sütundan başlıkları alıyoruz
    return titles


def main():
    # Uygulama başlığı
    st.title("Video Localization Automation")

    # Kullanıcıdan video, overlay görüntüleri, başlık CSV'si, çıktı dizini ve font dosyasını yüklemesini iste
    base_video = st.file_uploader("Upload Base Video", type=["mp4", "mov", "avi"])
    overlay_images = st.file_uploader(
        "Upload Images",
        type=["png", "jpg", "jpeg", "webp", "avif"],
        accept_multiple_files=True,  # Birden fazla dosya yüklemeyi etkinleştir
    )
    title_csv = st.file_uploader("Upload Titles CSV", type=["csv"])  # Başlık CSV dosyası yükleme
    output_dir = st.text_input("Output Directory")  # Çıktı dosyalarının kaydedileceği dizini gir
    font_file = st.file_uploader("Upload Font File", type=["ttf"])  # Font dosyası yükleme
    font_size = st.number_input("Font Size", value=70)  # Font boyutu için bir sayı girişi

    # "Process Video" butonuna basıldığında işlem başlat
    if st.button("Process Video"):
        # Tüm gerekli dosyalar yüklendiyse
        if base_video and overlay_images and title_csv and output_dir and font_file:
            # Çıktı dizini yoksa oluştur
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Yüklenen temel videoyu çıktı dizinine kaydet
            base_video_path = os.path.join(output_dir, base_video.name)
            with open(base_video_path, "wb") as f:
                f.write(base_video.getbuffer())

            # Yüklenen font dosyasını çıktı dizinine kaydet
            font_path_saved = os.path.join(output_dir, font_file.name)
            with open(font_path_saved, "wb") as f:
                f.write(font_file.getbuffer())

            # Overlay görüntülerini yükle ve işleme hazır hale getir
            overlay_images_loaded = [read_image(image) for image in overlay_images]

            # Başlıkları CSV dosyasından oku
            titles = read_titles_from_csv(title_csv)
            output_videos = []

            # Her overlay görüntüsü ve başlık için video işle
            for i in range(len(overlay_images_loaded)):
                with st.spinner(f"Processing video {i + 1}..."):  # İşlem sırasında bir spinner göstergesi
                    output_video = process_video(
                        base_video_path,
                        [overlay_images_loaded[i]],  # Tek bir overlay görüntüsü kullan
                        [titles[i]],  # Tek bir başlık kullan
                        output_dir,
                        font_path_saved,
                        font_size,
                    )
                    output_videos.append(output_video[0])  # İşlenmiş video yolunu listeye ekle

                    # İşlenmiş videoyu başarılı bir şekilde işlediğini bildir ve göster
                    st.success(f"Video {i + 1} processed successfully!")
                    st.video(output_video[0])  # İşlenmiş videoyu göster

            # Tüm videolar işlendikten sonra başarı mesajı göster
            st.success("All videos processed!")
        else:
            # Gerekli dosyalar yüklenmediyse uyarı göster
            st.warning("Please upload all required files.")

# Program ana fonksiyon olarak çalıştırıldığında main fonksiyonunu çağır
if __name__ == "__main__":
    main()