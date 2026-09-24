# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ransomwarewatch.data_preprocessing import load_and_preprocess_data
from ransomwarewatch.wcgan import WCGAN_GP
from ransomwarewatch.xgboost_model import RansomwareClassifier
import time

st.set_page_config(page_title="RANSOMEWATCH 2.0", layout="wide")

# Custom CSS for Windows-style animation
st.markdown("""
<style>
.windows-progress {
    width: 100%;
    background-color: #f1f1f1;
    border-radius: 4px;
    margin: 10px 0;
}
.windows-progress-bar {
    height: 24px;
    background-color: #00b4d8;
    border-radius: 4px;
    transition: width 0.3s ease;
    position: relative;
    overflow: hidden;
}
.windows-progress-bar::after {
    content: '';
    position: absolute;
    top: 0;
    left: -50%;
    width: 200%;
    height: 100%;
    background: linear-gradient(
        90deg,
        rgba(255,255,255,0) 25%,
        rgba(255,255,255,0.8) 50%,
        rgba(255,255,255,0) 75%
    );
    animation: shine 2s infinite;
}
@keyframes shine {
    0% { transform: translateX(-50%); }
    100% { transform: translateX(50%); }
}
</style>
""", unsafe_allow_html=True)


def main():
    st.title("ðŸ›¡ï¸ RANSOMEWATCH 2.0 - Ransomware Detection System")

    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = []

    menu = st.sidebar.selectbox("Menu", ["Train Model", "Detect Ransomware"])

    if menu == "Train Model":
        train_interface()
    else:
        detect_interface()


def train_interface():
    st.header("Model Training")

    if st.button("ðŸš€ Start Full Training Pipeline"):
        with st.spinner("Loading and preprocessing data..."):
            X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data()
            st.session_state.scaler = scaler
            st.session_state.feature_names = features

        st.subheader("Adversarial Training (WCGAN-GP)")
        gan = WCGAN_GP(input_dim=X_train.shape[1])

        st.markdown('<div class="windows-progress"><div class="windows-progress-bar" style="width: 0%"></div></div>',
                    unsafe_allow_html=True)

        gan.train(X_train[y_train == 1], epochs=100, batch_size=128)

        st.subheader("Classifier Training (XGBoost)")
        classifier = RansomwareClassifier()
        report = classifier.train(X_train, y_train, X_test, y_test)

        # Performance Metrics
        st.subheader("ðŸ“ˆ Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{report['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{report['1']['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{report['1']['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{report['1']['f1-score']:.2%}")

        st.success("ðŸŽ‰ Training Complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(st.session_state.confusion_matrix)
        with col2:
            st.pyplot(st.session_state.roc_curve)

        # Detailed Report
        with st.expander("ðŸ“Š Detailed Classification Report"):
            st.write(pd.DataFrame(report).transpose())

        with open(st.session_state.pdf_report, "rb") as f:
            st.download_button("ðŸ“¥ Download Full Report", f, file_name="training_report.pdf")

        st.session_state.model = classifier


def detect_interface():
    st.header("Real-time Detection")
    uploaded_file = st.file_uploader("Upload network traffic data (CSV)", type=["csv"])

    if uploaded_file and st.session_state.model:
        df = pd.read_csv(uploaded_file)

        # Preprocess
        cols_to_remove = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns if col in df.columns])
        df = df.select_dtypes(include=['number'])
        df = df[st.session_state.feature_names]

        processed = st.session_state.scaler.transform(df)

        # Animated scanning
        st.subheader("ðŸ› ï¸ Scanning in Progress...")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        predictions = st.session_state.model.predict(processed)

        # Results display
        df_result = df.copy()
        df_result['Prediction'] = ['âš ï¸ Ransomware' if p else 'âœ… Clean' for p in predictions]
        ransomware_count = sum(predictions)

        # Detection Metrics
        st.success(f"Detection Completed! ðŸ›¡ï¸")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Scanned", len(predictions))
        with cols[1]:
            st.metric("Clean Files", sum(p == 0 for p in predictions))
        with cols[2]:
            st.metric("Threats Detected", ransomware_count)
        with cols[3]:
            confidence = ransomware_count / len(predictions) if len(predictions) > 0 else 0
            st.metric("Detection Rate", f"{confidence:.2%}")

        # Threat preview
        if ransomware_count > 0:
            st.subheader("âš¡ Threat Preview (Top 10 Ransomware Samples)")
            st.dataframe(df_result[df_result['Prediction'] == 'âš ï¸ Ransomware'].head(10))
        else:
            st.info("âœ… No ransomware detected!")

        # Download
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("ðŸ“¥ Download Full Result CSV", data=csv, file_name="detection_results.csv", mime='text/csv')

        # SHAP explanation
        if st.checkbox("Show threat explanation"):
            sample_idx = st.selectbox("Select sample to explain", range(len(df)))
            shap_values = st.session_state.model.explainer.shap_values(processed[sample_idx])
            st.pyplot(shap.force_plot(st.session_state.model.explainer.expected_value,
                                      shap_values,
                                      processed[sample_idx]))


if __name__ == "__main__":
    main()
