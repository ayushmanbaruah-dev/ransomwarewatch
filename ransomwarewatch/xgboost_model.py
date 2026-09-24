# xgboost_model.py
from xgboost import XGBClassifier
import streamlit as st
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, RocCurveDisplay
import shap
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import seaborn as sns
import uuid

class RansomwareClassifier:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.explainer = None
        self.is_trained = False

    def train(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        self.explainer = shap.TreeExplainer(self.model)
        self.is_trained = True
        return self._generate_report(X_test, y_test)

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Please train before prediction.")
        return self.model.predict_proba(X)

    def _generate_report(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Enhanced metrics collection
        report = classification_report(y_test, y_pred, output_dict=True)
        report['accuracy'] = accuracy_score(y_test, y_pred)  # Add explicit accuracy
        report['auc'] = roc_auc_score(y_test, y_proba)  # Add AUC to report

        # Visualizations
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_roc_curve(y_test, y_proba)
        self._generate_pdf(report, report['auc'])

        return report

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.session_state.confusion_matrix = plt.gcf()

    def _plot_roc_curve(self, y_true, y_proba):
        plt.figure()
        RocCurveDisplay.from_predictions(y_true, y_proba)
        plt.title('ROC Curve')
        st.session_state.roc_curve = plt.gcf()

    def _generate_pdf(self, report, auc):
        unique_id = str(uuid.uuid4())[:8]
        pdf_path = f"training_report_{unique_id}.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)

        # Add content
        c.drawString(100, 750, "Training Report - Ransomware Detection Model")
        c.drawString(100, 730, f"AUC Score: {auc:.4f}")
        y = 700
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                c.drawString(120, y, f"Class {label}:")
                y -= 20
                for k, v in metrics.items():
                    c.drawString(140, y, f"{k}: {v:.4f}")
                    y -= 20
            else:
                c.drawString(120, y, f"{label}: {metrics:.4f}")
                y -= 20
            y -= 10

        c.save()
        st.session_state.pdf_report = pdf_path