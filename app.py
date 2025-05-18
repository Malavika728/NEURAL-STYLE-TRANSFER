import streamlit as st
import numpy as np
from PIL import Image
import io
from style_transfer import stylize_image

st.set_page_config(page_title="Neural Style Transfer", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎨 Neural Style Transfer App</h1>", unsafe_allow_html=True)
st.caption("Built with ❤️ by CodTech Intern")

st.sidebar.header("Upload Images")
content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.sidebar.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if st.sidebar.button("Stylize"):
    if content_file and style_file:
        with st.spinner("Generating stylized image..."):
            # Run style transfer
            stylized = stylize_image(content_file, style_file)

            # Convert tensor to image
            stylized_np = np.squeeze(stylized[0].numpy(), axis=0)
            stylized_np = (stylized_np * 255).astype(np.uint8)
            stylized_img = Image.fromarray(stylized_np)

            # Load original images
            content_img = Image.open(content_file).convert("RGB")
            style_img = Image.open(style_file).convert("RGB")

            # Display content and style side-by-side
            st.write("### Uploaded Images")
            col1, col2 = st.columns(2)
            with col1:
                st.image(content_img, caption="Content Image", use_container_width=True)
            with col2:
                st.image(style_img, caption="Style Image", use_container_width=True)

            # Stylized image full width below
            st.write("### Stylized Output")
            st.image(stylized_img, caption="Stylized Image", use_container_width=True)

            # Download button with download icon
            buf = io.BytesIO()
            stylized_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button("⬇️ Download Stylized Image", byte_im, file_name="stylized.png", mime="image/png")
    else:
        st.warning("Please upload both content and style images.")
