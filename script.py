import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import os
import streamlit as st
import pandas as pd

def read_image(image):
    try:
        img = Image.open(image)
        img = img.convert('RGBA')
        np_img = np.array(img)
        print(f"Successfully read image: {image.name}")
        print(f"Image shape: {np_img.shape}")
        return np_img
    except Exception as e:
        print(f"Error reading {image.name}: {e}")
        return None

def create_text_clip(text, size, font_size, color, font_path):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found or couldn't be read: {font_path}")
        print("Using default font.")
        font = ImageFont.load_default()
    
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        test_line = f"{current_line} {word}"
        bbox = draw.textbbox((0, 0), test_line, font=font)  # Calculate text bounding box
        width = bbox[2] - bbox[0]
        if width <= size[0] * 0.9:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    total_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
    y = (size[1] - total_height) / 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        outline_color = (0, 0, 0, 255)  # Black outline color
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                draw.text(((size[0] - width) / 2 + dx, y + dy), line, font=font, fill=outline_color)
        draw.text(((size[0] - width) / 2, y), line, font=font, fill=color)
        y += height

    return np.array(img)

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

def detect_green_circle(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        print(f"Detected green circle: center={center}, radius={radius}")
        return center, radius
    print("No green circle detected")
    return None, None

def process_frame(frame, overlay_image, title, font_path, font_size, frame_number, total_frames):
    frame = frame.copy()
    center, radius = detect_green_circle(frame)

    if center is not None and radius > 0:
        overlay_resized = cv2.resize(overlay_image, (2 * radius, 2 * radius))
        print(f"Resized overlay shape: {overlay_resized.shape}")

        circular_mask = create_circular_mask(2 * radius, 2 * radius, (radius, radius), radius)

        circular_mask_4ch = np.dstack([circular_mask] * 4)

        masked_overlay = overlay_resized * circular_mask_4ch
        print(f"Masked overlay shape: {masked_overlay.shape}")

        x, y = center
        x_start = max(0, x - radius)
        x_end = min(frame.shape[1], x + radius)
        y_start = max(0, y - radius)
        y_end = min(frame.shape[0], y + radius)

        overlay_section = masked_overlay[0:(y_end - y_start), 0:(x_end - x_start)]
        print(f"Overlay section shape: {overlay_section.shape}")

        alpha_s = overlay_section[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            frame[y_start:y_end, x_start:x_end, c] = (
                alpha_s * overlay_section[:, :, c] +
                alpha_l * frame[y_start:y_end, x_start:x_end, c]
            )

        print(f"Frame updated with overlay at position: ({x_start}, {y_start}) to ({x_end}, {y_end})")
    else:
        print("No green circle detected, skipping overlay")

    title_img = create_text_clip(
        title,
        (frame.shape[1], int(frame.shape[0] * 0.2)),
        font_size,
        (255, 255, 255, 255),
        font_path,
    )

    title_y = 20
    title_mask = title_img[:, :, 3] / 255.0
    title_mask = np.dstack([title_mask] * 3)
    title_rgb = title_img[:, :, :3]

    frame[title_y:title_y + title_img.shape[0], :] = (
        frame[title_y:title_y + title_img.shape[0], :] * (1 - title_mask) +
        title_rgb * title_mask
    )

    return frame

def process_video(base_video_path, overlay_images, titles, output_dir, font_path, font_size):
    video = VideoFileClip(base_video_path)
    total_frames = int(video.fps * video.duration)

    min_length = min(len(overlay_images), len(titles))
    overlay_images = overlay_images[:min_length]
    titles = titles[:min_length]

    output_videos = []

    for i, (overlay_image, title_text) in enumerate(zip(overlay_images, titles)):
        print(f"Processing video {i + 1} with overlay image.")

        def process_frame_with_counter(img):
            nonlocal frame_number
            processed = process_frame(img, overlay_image, title_text, font_path, font_size, frame_number, total_frames)
            frame_number += 1
            return processed

        frame_number = 0
        processed_video = video.fl_image(process_frame_with_counter)

        output_path = os.path.join(output_dir, f"localized_video_{i + 1}.mp4")
        processed_video.write_videofile(output_path, codec="libx264")
        output_videos.append(output_path)

        print(f"Processed and saved video: {output_path}")

    return output_videos

def read_titles_from_csv(title_csv):
    titles = []
    df = pd.read_csv(title_csv)
    titles = df.iloc[:, 0].tolist()  # First column for titles
    return titles        

def main():
    st.title("Video Localization Automation")
    
    base_video = st.file_uploader("Upload Base Video", type=["mp4", "mov", "avi"])
    overlay_images = st.file_uploader("Upload Overlay Images", type=["png", "jpg", "jpeg", "webp", "avif"], accept_multiple_files=True)
    title_csv = st.file_uploader("Upload Titles CSV", type=["csv"])
    output_dir = st.text_input("Output Directory")
    font_file = st.file_uploader("Upload Font File", type=["ttf"])
    font_size = st.number_input("Font Size", value=70)

    if st.button("Process Video"):
        if base_video and overlay_images and title_csv and output_dir and font_file:
            # Create directories if not exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the uploaded files
            base_video_path = os.path.join(output_dir, base_video.name)
            with open(base_video_path, "wb") as f:
                f.write(base_video.getbuffer())

            font_path_saved = os.path.join(output_dir, font_file.name)
            with open(font_path_saved, "wb") as f:
                f.write(font_file.getbuffer())

            overlay_images_loaded = [read_image(image) for image in overlay_images]

            titles = read_titles_from_csv(title_csv)
            output_videos = []

            for i in range(len(overlay_images_loaded)):
                with st.spinner(f"Processing video {i + 1}/{len(overlay_images_loaded)}..."):
                    output_videos = process_video(base_video_path, overlay_images_loaded, titles, output_dir, font_path_saved, font_size)

            st.success("Video processing complete!")

            # Add download buttons for each output video
            for video in output_videos:
                with open(video, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    label=f"Download Video {os.path.basename(video)}",
                    data=video_bytes,
                    file_name=os.path.basename(video),
                    mime="video/mp4"
                )
        else:
            st.warning("Please upload all required files.")

if __name__ == "__main__":
    main()
