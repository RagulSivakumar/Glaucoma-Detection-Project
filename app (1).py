
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "/content/drive/MyDrive/Glaucoma_Project/mobilenetv2_model.keras"
    )

model = load_model()

st.title("Glaucoma Detection")
st.write("Upload a retinal fundus image for model prediction.")

uploaded_file = st.file_uploader(
    "Choose an eye image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image", use_container_width=True)

    if st.button("Predict"):
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = float(model.predict(image_array, verbose=0)[0][0])

        if prediction >= 0.5:
            st.success("Model Prediction: Normal")
        else:
            st.warning("Model Prediction: Glaucoma")

        st.caption(
            "Educational project only. This prediction is not a medical diagnosis."
        )
        
