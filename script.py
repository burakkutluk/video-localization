import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import os
import csv
import streamlit as st
import pandas as pd
import pillow_avif

def read_image(image):
    """Reads an image file and converts it to an RGBA numpy array."""
    try:
        img = Image.open(image)  # Open the image file
        img = img.convert('RGBA')  # Convert to RGBA format
        np_img = np.array(img)  # Convert the image to a numpy array
        print(f"Successfully read image: {image.name}")
        print(f"Image shape: {np_img.shape}")
        return np_img
    except Exception as e:
        print(f"Error reading {image.name}: {e}")
        return None

def create_text_clip(text, size, font_size, color, font_path):
    """Creates a text image with wrapping and returns it as a numpy array."""
    try:
        font = ImageFont.truetype(font_path, font_size)  # Load the specified font
    except IOError:
        print(f"Error: Font file not found or couldn't be read: {font_path}")
        print("Using default font.")
        font = ImageFont.load_default()  # Fall back to the default font if loading fails
    
    # Create a new blank image for the text
    img = Image.new("RGBA", size, (255, 255, 255, 0))  # Transparent background
    draw = ImageDraw.Draw(img)  # Prepare to draw on the image

    # Split text into lines that fit within the specified width
    lines = []
    words = text.split()
    current_line = words[0]

    for word in words[1:]:
        test_line = f"{current_line} {word}"  # Test line with the next word
        bbox = draw.textbbox((0, 0), test_line, font=font)  # Calculate the bounding box for the text
        width = bbox[2] - bbox[0]  # Width of the text
        if width <= size[0] * 0.9:  # Check if it fits within the width limit
            current_line = test_line
        else:
            lines.append(current_line)  # If not, finalize the current line
            current_line = word  # Start a new line with the current word
    lines.append(current_line)  # Add the last line

    # Calculate total height for centering the text
    total_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
    y = (size[1] - total_height) / 2  # Start y position for centering

    # Draw each line on the image
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)  # Get bounding box for the line
        width = bbox[2] - bbox[0]  # Width of the line
        height = bbox[3] - bbox[1]  # Height of the line

        # Draw text outline for better visibility
        outline_color = (0, 0, 0, 255)  # Black outline color
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                draw.text(((size[0] - width) / 2 + dx, y + dy), line, font=font, fill=outline_color)  # Draw outline
        draw.text(((size[0] - width) / 2, y), line, font=font, fill=color)  # Draw main text
        y += height  # Move y position for the next line

    return np.array(img)  # Return the text image as a numpy array

def create_circular_mask(h, w, center, radius):
    """Creates a circular mask for overlaying images."""
    Y, X = np.ogrid[:h, :w]  # Create a grid of coordinates
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)  # Calculate distance from the center
    mask = dist_from_center <= radius  # Create a mask based on the radius
    return mask  # Return the circular mask

def detect_green_circle(frame):
    """Detects a green circle in the given frame and returns its center and radius."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV color space
    lower_green = np.array([40, 40, 40])  # Lower bound for green color
    upper_green = np.array([70, 255, 255])  # Upper bound for green color

    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)  # Create a mask for green color

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Find the largest contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)  # Get the enclosing circle
        center = (int(x), int(y))  # Center of the circle
        radius = int(radius)  # Radius of the circle
        print(f"Detected green circle: center={center}, radius={radius}")
        return center, radius  # Return center and radius
    print("No green circle detected")
    return None, None  # Return None if no circle is detected

def process_frame(frame, overlay_image, title, font_path, font_size, frame_number, total_frames):
    """Processes a single frame by applying the overlay image and title."""
    frame = frame.copy()  # Make a copy of the frame to avoid modifying the original
    center, radius = detect_green_circle(frame)  # Detect green circle

    if center is not None and radius > 0:  # If a circle was detected
        overlay_resized = cv2.resize(overlay_image, (2 * radius, 2 * radius))  # Resize overlay to fit the circle
        print(f"Resized overlay shape: {overlay_resized.shape}")

        circular_mask = create_circular_mask(2 * radius, 2 * radius, (radius, radius), radius)  # Create a circular mask

        circular_mask_4ch = np.dstack([circular_mask] * 4)  # Create a 4-channel mask for RGBA

        masked_overlay = overlay_resized * circular_mask_4ch  # Apply mask to overlay
        print(f"Masked overlay shape: {masked_overlay.shape}")

        x, y = center  # Get the center of the circle
        x_start = max(0, x - radius)  # Calculate start x position
        x_end = min(frame.shape[1], x + radius)  # Calculate end x position
        y_start = max(0, y - radius)  # Calculate start y position
        y_end = min(frame.shape[0], y + radius)  # Calculate end y position

        overlay_section = masked_overlay[0:(y_end - y_start), 0:(x_end - x_start)]  # Get the section of the overlay
        print(f"Overlay section shape: {overlay_section.shape}")

        alpha_s = overlay_section[:, :, 3] / 255.0  # Get alpha channel from overlay
        alpha_l = 1.0 - alpha_s  # Invert alpha for the frame

        # Blend overlay with frame
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
        (frame.shape[1], int(frame.shape[0] * 0.2)),  # Create title image
        font_size,
        (255, 255, 255, 255),  # White color for title
        font_path,
    )

    title_y = 20  # Y position for title
    title_mask = title_img[:, :, 3] / 255.0  # Create mask from alpha channel
    title_mask = np.dstack([title_mask] * 3)  # Convert to 3-channel mask
    title_rgb = title_img[:, :, :3]  # Get RGB channels from title image

    # Blend title with frame
    frame[title_y:title_y + title_img.shape[0], :] = (
        frame[title_y:title_y + title_img.shape[0], :] * (1 - title_mask) +
        title_rgb * title_mask
    )

    return frame  # Return the processed frame

def process_video(base_video_path, overlay_images, titles, output_dir, font_path, font_size):
    """Processes the video frame by frame, applying overlays and titles."""
    video = VideoFileClip(base_video_path)  # Load the base video
    print(f"Loaded video: {base_video_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    frame_count = int(video.fps * video.duration)  # Total number of frames
    print(f"Total frames in video: {frame_count}")

    # Process each frame and create a new video with overlays
    processed_frames = []
    for i, frame in enumerate(video.iter_frames(fps=video.fps, dtype='uint8')):
        print(f"Processing frame {i + 1}/{frame_count}...")
        overlay_image = overlay_images[i % len(overlay_images)]  # Cycle through overlay images
        title = titles[i % len(titles)]  # Cycle through titles

        processed_frame = process_frame(frame, overlay_image, title, font_path, font_size, i, frame_count)  # Process the frame
        processed_frames.append(processed_frame)  # Append processed frame to list

    output_video_path = os.path.join(output_dir, "output_video.mp4")  # Path for output video
    print(f"Saving processed video to: {output_video_path}")

    # Convert processed frames to a VideoFileClip
    output_video_clip = VideoFileClip(np.array(processed_frames), fps=video.fps)
    output_video_clip.write_videofile(output_video_path, codec="libx264")  # Write the video file

    print("Video processing complete.")

# Streamlit UI
def main():
    """Main function for Streamlit app."""
    st.title("Video Localization Automation")  # Title for the app

    # File uploader for video and overlays
    base_video = st.file_uploader("Upload Base Video", type=["mp4"])
    overlay_files = st.file_uploader("Upload Images", type=["avif", "png"], accept_multiple_files=True)

    # Input for titles
    titles_input = st.text_area("Enter Titles (one per line)", "")
    titles = titles_input.splitlines() if titles_input else []  # Split titles into list

    # Font settings
    font_path = "arial.ttf"  # Specify font path
    font_size = st.slider("Font Size", 10, 100, 24)  # Slider for font size

    # Output directory selection
    output_dir = st.text_input("Output Directory", "output")  # Output directory input

    if st.button("Process Video"):
        if base_video is not None and overlay_files and titles:
            overlay_images = [read_image(overlay) for overlay in overlay_files]  # Read overlay images
            print(f"Read {len(overlay_images)} overlay images.")

            if base_video and overlay_images:
                process_video(base_video.name, overlay_images, titles, output_dir, font_path, font_size)  # Process the video
                st.success("Video processed successfully!")
            else:
                st.error("Please upload a valid base video and overlay images.")
        else:
            st.error("Please provide all inputs: video, overlay images, and titles.")

if __name__ == "__main__":
    main()  # Run the Streamlit app