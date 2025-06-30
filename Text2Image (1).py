import streamlit as st
import vertexai
from PIL import Image
import io
import base64
from vertexai.preview.vision_models import ImageGenerationModel

st.sidebar.title("Define your Image")

# Initialize Vertex AI
project_id = 'ford-180395bd732cdd9af050c1f7'
vertexai.init(project=project_id, location="us-central1")

# Initialize images variable in session state
if 'images' not in st.session_state:
    st.session_state.images = []

# Sidebar options for user inputs
prompt = st.sidebar.text_input("Enter the prompt for image generation:")
number_of_images = st.sidebar.number_input("Select the number of images to generate", min_value=1, max_value=5, value=1)
aspect_ratio_options = ["1:1", "9:16", "16:9", "3:4", "4:3"]
aspect_ratio = st.sidebar.selectbox("Select the aspect ratio", aspect_ratio_options)

# Create the ImageGenerationModel
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001") #imagegeneration@006

# Generate images based on user inputs
if st.sidebar.button("Generate Images"):
    st.session_state.images = model.generate_images(
        prompt=prompt,
        number_of_images=number_of_images,
        language="en",
        aspect_ratio=aspect_ratio,
        safety_filter_level="block_some",
        person_generation="allow_adult",
    )

# Main body to display the generated images
st.title("Text to Image Generation Model")
for i, image_data in enumerate(st.session_state.images):
    image_bytes = image_data._image_bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    st.image(image, caption=f'Generated Image {i+1}', use_column_width=True)
    st.write(f"Created output image {i+1} using {len(image_bytes)} bytes")
    
    # Download link for the individual image
    image_b64 = base64.b64encode(image_bytes).decode()
    href = f'<a href="data:file/png;base64,{image_b64}" download="generated_image_{i+1}.png">Download Image {i+1}</a>'
    st.markdown(href, unsafe_allow_html=True)
