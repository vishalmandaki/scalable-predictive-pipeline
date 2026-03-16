import shap
import joblib
import matplotlib.pyplot as plt

class ModelExplainer:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        print(f"Loaded model from {model_path} for explanation.")

    def explain(self, X):
        """Generate SHAP values for model predictions."""
        # Use TreeExplainer for Random Forest
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # Plot summary
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("feature_importance.png")
        print("SHAP Summary Plot saved as feature_importance.png")
        return shap_values

if __name__ == "__main__":
    # Example usage (assuming model exists)
    # explainer = ModelExplainer("model.joblib")
    # explainer.explain(X_test)
    print("Explainability module ready.")
