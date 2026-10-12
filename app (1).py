import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Glaucoma Detection AI",
    page_icon="👁️",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.main-title {
    font-size: 38px;
    font-weight: bold;
    color: #17365d;
}
.subtitle {
    font-size: 17px;
    color: #526579;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #dce5ef;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "mobilenetv2_model.keras"

    if not model_path.exists():
        st.error("Model file not found. Check the repository.")
        st.stop()

    return tf.keras.models.load_model(model_path)

model = load_model()

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("👁️ Glaucoma AI")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔬 Image Prediction",
        "📊 Model Comparison",
        "📜 Prediction History",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "AI-assisted retinal image classification "
    "for educational and research purposes."
)

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">Glaucoma Detection System</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Deep Learning Based Retinal Image Classification</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 MobileNetV2")
        st.write(
            "A transfer-learning model used to classify "
            "uploaded retinal fundus images."
        )

    with col2:
        st.markdown("### 🎯 Two Classes")
        st.write("**Glaucoma** and **Normal** retinal images.")

    st.markdown("---")
    st.subheader("How to Use")

    st.write("1. Open **Image Prediction**.")
    st.write("2. Upload a retinal fundus image.")
    st.write("3. Click **Predict Image**.")
    st.write("4. View the model output and prediction score.")
    st.write("5. Download the prediction report if required.")

    st.warning(
        "Educational project only. This system is not a "
        "medical diagnostic tool. Consult an eye-care professional "
        "for medical evaluation."
    )

# ---------------- IMAGE PREDICTION ----------------
elif page == "🔬 Image Prediction":

    st.title("🔬 Retinal Image Prediction")
    st.write("Upload a retinal fundus image in JPG, JPEG or PNG format.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Retinal Image",
                use_container_width=True
            )

        with col2:
            st.markdown("### Image Details")
            st.write(f"Width: {image.width} pixels")
            st.write(f"Height: {image.height} pixels")
            st.write(f"Format: {uploaded_file.type}")

        if st.button("🔍 Predict Image", type="primary"):

            with st.spinner("Analysing image..."):

                resized_image = image.resize((224, 224))

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                ) / 255.0

                image_array = np.expand_dims(image_array, axis=0)

                score = float(
                    model.predict(image_array, verbose=0)[0][0]
                )

                # Model labels: 0 = Glaucoma, 1 = Normal
                if score >= 0.5:
                    prediction = "Normal"
                    confidence = score * 100
                else:
                    prediction = "Glaucoma"
                    confidence = (1 - score) * 100

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                st.session_state.history.append({
                    "Time": timestamp,
                    "Image": uploaded_file.name,
                    "Prediction": prediction,
                    "Model Score (%)": round(score * 100, 2),
                    "Predicted Class Score (%)": round(confidence, 2)
                })

            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction == "Glaucoma":
                st.error("Model Prediction: Glaucoma")
            else:
                st.success("Model Prediction: Normal")

            c1, c2 = st.columns(2)

            c1.metric("Predicted Class", prediction)
            c2.metric("Model Score", f"{confidence:.2f}%")

            st.progress(max(0.0, min(1.0, confidence / 100)))

            st.caption(
                "This percentage is the model's score for its predicted "
                "class, not a clinically validated probability or diagnosis."
            )

            report = pd.DataFrame([{
                "Image": uploaded_file.name,
                "Prediction": prediction,
                "Predicted Class Score (%)": round(confidence, 2),
                "Timestamp": timestamp
            }])

            st.download_button(
                "📥 Download Prediction Report",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="glaucoma_prediction_report.csv",
                mime="text/csv"
            )

            st.warning(
                "This result is for educational purposes only. "
                "It must not be used to diagnose or rule out glaucoma."
            )

# ---------------- MODEL COMPARISON ----------------
elif page == "📊 Model Comparison":

    st.title("📊 CNN vs MobileNetV2")
    st.write("Comparison using the test results from your project.")

    comparison = pd.DataFrame({
        "Model": ["Custom CNN", "MobileNetV2"],
        "Test Accuracy (%)": [89.52, 91.90]
    })

    st.dataframe(comparison, use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(
        comparison["Model"],
        comparison["Test Accuracy (%)"]
    )
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Test Accuracy Comparison")

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "In the recorded test results, MobileNetV2 achieved 91.90% "
        "accuracy, compared with 89.52% for the custom CNN. "
        "These are results from this project's evaluation, "
        "not a guarantee of performance on new data."
    )

    metrics = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score"],
        "Custom CNN (%)": [85.25, 96.30, 90.43],
        "MobileNetV2 (%)": [95.05, 88.89, 91.87]
    })

    st.subheader("Additional Evaluation Metrics")
    st.dataframe(metrics, use_container_width=True)

    st.caption(
        "Precision, recall and F1-score shown here use Normal as the "
        "positive class, matching the recorded overall metrics. "
        "Use the per-class report when evaluating glaucoma detection."
    )

# ---------------- PREDICTION HISTORY ----------------
elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "📥 Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info(
            "No predictions in this session yet. "
            "Go to Image Prediction to test an image."
        )

    st.caption(
        "History is stored only in the current app session and is "
        "not a permanent database."
    )

# ---------------- ABOUT PROJECT ----------------
elif page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.markdown("""
    ### Project Title
    **Glaucoma Detection Using Deep Learning**

    ### Objective
    To develop an educational application that classifies retinal
    fundus images into Glaucoma and Normal classes.

    ### Dataset
    Cataract, DR, Normal and Glaucoma Fundus Images Dataset
    from Kaggle.

    ### Proposed Model
    MobileNetV2 Transfer Learning.

    ### Technologies Used
    - Python
    - TensorFlow / Keras
    - Streamlit
    - NumPy
    - Pandas
    - Matplotlib
    - Pillow

    ### Workflow
    Retinal Image → Preprocessing → MobileNetV2 → Prediction Result

    ### Important Note
    This application is a student research project. Its output is not
    a substitute for professional eye examination or medical diagnosis.
    """)
