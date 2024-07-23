
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import json
import io
from process import load_seg_model, get_palette, generate_mask, generate_mask_unet, apply_new_colors, draw_color_regions, find_suitable_palette, rgb_to_hex, hex_to_rgb

# Function to convert image to bytes for download
def convert_image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    img_pil = Image.fromarray(image)
    img_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

# Main title
st.title("Cloth Segmentation and Color Modification")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "U2Net Segmentation", "Unet Segmentation"])

# Add some space before the Options section
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")

# Sidebar options
st.sidebar.title("Options")

# Upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Color palettes JSON file path
color_palettes_path = st.sidebar.text_input("Path to color palettes JSON file", "100_five_color_palettes.json")

# Model checkpoint path (for U2Net)
checkpoint_path = st.sidebar.text_input("Path to model checkpoint", "cloth_segm.pth")

# Use CUDA
use_cuda = st.sidebar.checkbox("Use CUDA", False)
device = 'cuda:0' if use_cuda else 'cpu'

if page == "Home":
    st.write("### Marketing Title:")
    st.markdown("**\"Revamp Your Wardrobe with AI: Seamless Cloth Segmentation and Custom Colorization\"**")
    
    st.write("### App Explanation:")
    st.markdown("""
    Experience the future of fashion design with our AI-powered application that transforms how you visualize and customize clothing. Our advanced cloth segmentation technology precisely identifies and isolates different garment sections in any image. Using a user-friendly interface, you can effortlessly apply custom colors to each segment, experiment with suggested palettes, and instantly see your design come to life. Whether you're a designer seeking inspiration or simply want to try new styles, our tool offers endless possibilities for personalization and creativity.
    """)

elif page == "U2Net Segmentation":
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Load color palettes from JSON file
        with open(color_palettes_path, 'r') as file:
            color_palettes = json.load(file)

        # Create an instance of the model
        model = load_seg_model(checkpoint_path, device=device)
        palette = get_palette(4)

        # Placeholder for the text
        text_placeholder = st.empty()
        text_placeholder.text("Generating mask and segmentations...")

        # Generate mask and segmentations
        cloth_seg, output_arr, color_regions, average_colors, len_classes = generate_mask(image, net=model, palette=palette, device=device)

        # Find a suitable color palette
        target_colors = [average_colors[cls] for cls in sorted(average_colors.keys())]
        suitable_palette = find_suitable_palette(target_colors, color_palettes)

        # Clear the placeholder text
        text_placeholder.empty()

        # Display extracted colors and allow user to choose new colors with transparency
        st.write("## Customize Class Colors")

        user_selected_colors = {}

        for cls, avg_color in average_colors.items():
            mask_image_path = f'output/alpha/{cls}.png'
            if len(len_classes)==1:
                cls = len_classes
            alpha_mask_img = Image.open(mask_image_path)
            alpha_mask_img.thumbnail((50, 50))  # Resize for thumbnail display
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                st.image(alpha_mask_img, caption=f"Class {cls}", width=50)
            with col2:
                hex_color = rgb_to_hex(avg_color)
                user_color = st.color_picker(f"Choose color for Class {cls}", hex_color)
                user_selected_colors[cls] = {'color': hex_to_rgb(user_color)}
            with col3:
                transparency = st.slider(f"Transparency for Class {cls}", 0, 100, 50)
                user_selected_colors[cls]['transparency'] = transparency / 100.0  # Normalize to [0, 1]

        # Update the image with user-selected colors
        def apply_user_colors(image, output_arr, user_selected_colors):
            new_image = image.copy()
            for cls, properties in user_selected_colors.items():
                color = properties['color']
                transparency = properties['transparency']
                mask = output_arr[0] == cls
                if np.any(mask):
                    mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    overlay = new_image[mask_resized.astype(bool)]
                    blended_color = (1 - transparency) * overlay + transparency * np.array(color)
                    new_image[mask_resized.astype(bool)] = blended_color.astype(np.uint8)
            return new_image

        user_colored_image = apply_user_colors(img_np, output_arr, user_selected_colors)

        st.write("## Image with User Selected Colors")
        st.image(user_colored_image, caption="User Customized Image", use_column_width=True)
        st.download_button("Download Customized Image", data=convert_image_to_bytes(user_colored_image), file_name="customized_image.png", mime="image/png")

        # Display suitable color palette
        st.write("## Suitable Color Palette")
        if suitable_palette:
            st.image([Image.fromarray(np.full((50, 50, 3), color, dtype=np.uint8)) for color in suitable_palette], caption=[rgb_to_hex(color) for color in suitable_palette], width=50)
        else:
            st.write("No suitable palette found.")

        # Draw color regions
        image_with_boxes1, image_with_boxes2, image_with_boxes3 = draw_color_regions(img_np, output_arr)

        # Display images in the same row with a title above
        st.write("## Image segmentation with New Colors")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image_with_boxes1, caption="Version 1", use_column_width=True)
            st.download_button("Download Version 1", data=convert_image_to_bytes(image_with_boxes1), file_name="version1.png", mime="image/png")
        with col2:
            st.image(image_with_boxes2, caption="Version 2", use_column_width=True)
            st.download_button("Download Version 2", data=convert_image_to_bytes(image_with_boxes2), file_name="version2.png", mime="image/png")
        with col3:
            st.image(image_with_boxes3, caption="Version 3", use_column_width=True)
            st.download_button("Download Version 3", data=convert_image_to_bytes(image_with_boxes3), file_name="version3.png", mime="image/png")

        # Generate and display suggested images based on Suitable Color Palette
        st.write("## Suggested Images Based on Suitable Color Palette")
        suggested_images = []
        class_colors = list(user_selected_colors.keys())

        if len(len_classes) > 1:
            for i, base_color in enumerate(suitable_palette):
                for j, swap_color in enumerate(suitable_palette):
                    if i != j:
                        temp_colors = user_selected_colors.copy()
                        temp_colors[class_colors[0]]['color'] = base_color
                        temp_colors[class_colors[1]]['color'] = swap_color
                        suggested_image = apply_user_colors(img_np, output_arr, temp_colors)
                        suggested_images.append(suggested_image)
        else:
            for color in suitable_palette:
                temp_colors = user_selected_colors.copy()
                temp_colors[class_colors[0]]['color'] = color
                suggested_image = apply_user_colors(img_np, output_arr, temp_colors)
                suggested_images.append(suggested_image)

        cols = st.columns(3)
        for idx, img in enumerate(suggested_images):
            col = cols[idx % 3]
            with col:
                st.image(img, caption=f"Suggested Image {idx + 1}", use_column_width=True)
                st.download_button(f"Download Suggested Image {idx + 1}", data=convert_image_to_bytes(img), file_name=f"suggested_image_{idx + 1}.png", mime="image/png")

elif page == "Unet Segmentation":
    if uploaded_file is not None:
        # Save the uploaded image temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Generate mask and segmentations
        original_image, mask, dst = generate_mask_unet("temp_image.png", device=device)

        # Display original, mask, and blended images
        st.write("## Segmentation Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(mask * 255, caption="Mask", use_column_width=True)
        with col3:
            st.image(dst, caption="Blended Image", use_column_width=True)

        # Allow user to download the mask and blended images
        st.download_button("Download Mask", data=convert_image_to_bytes(mask * 255), file_name="mask.png", mime="image/png")
        st.download_button("Download Blended Image", data=convert_image_to_bytes(dst), file_name="blended_image.png", mime="image/png")
