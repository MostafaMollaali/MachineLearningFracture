import streamlit as st

# ======================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ======================
st.set_page_config(
    page_title="Material Strength AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ======================
# REST OF IMPORTS
# ======================
import numpy as np
import os
import json
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, LogCosh

# ======================
# CUSTOM STYLING (MOVED AFTER PAGE CONFIG)
# ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;500&display=swap');
    /* Keep the rest of your custom styles here */
</style>
""", unsafe_allow_html=True)

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Material Strength AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM STYLING
# ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;500&display=swap');

    html {
        scroll-behavior: smooth;
    }

    .main {
        background-color: #F5F5F5;
        padding: 2rem;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }

    .prediction-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .feature-range {
        color: #666;
        font-size: 0.9em;
        margin-top: -10px;
    }

    .success-animation {
        animation: bounce 1s;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-15px);}
        60% {transform: translateY(-7px);}
    }
</style>
""", unsafe_allow_html=True)

# ======================
# CONSTANTS & CONFIG
# ======================
categories = ["ANN-Practical Solution", "ANN-MPL", "XGBoost", "LightGBM"]
sheet_names = ["Glass-Tension", "Glass-Flexure", "Ceramic-Flexure", "Ceramic-Tension"]

ceramic_input_columns = ['sqrt(t/R)', 'E*sqrt(t)/Kic', 'ell/R', 'v']
ceramic_input_latex = [r'$\sqrt{\frac{t}{R}}$', r'$\frac{E\sqrt{t}}{K_{\mathrm{Ic}}}$', r'$\frac{\ell}{R}$', r'$\nu$']

glass_input_columns = ['sqrt(t/R)', 'E*sqrt(t)/Kic', 'v']
glass_input_latex = [r'$\sqrt{\dfrac{t}{R}}$', r'$\dfrac{E\sqrt{t}}{K_{\mathrm{Ic}}}$', r'$\nu$']

target_column = 'Sig*sqrt(t)/Kic'
target_latex = r'$\dfrac{\sigma\sqrt{t}}{K_{\mathrm{Ic}}}$'

model_files = {
    "ANN-Practical Solution": {
        "Glass-Tension": "_out/best_model_GLASS-TENSION_tanh_SGD_log_cosh.h5",
        "Glass-Flexure": "_out/best_model_GLASS-FLEXURE_tanh_AdamW_log_cosh.h5",
        "Ceramic-Flexure": "_out/best_model_CERAMIC-FLEXURE_tanh_RMSprop_log_cosh.h5",
        "Ceramic-Tension": "_out/best_model_CERAMIC-TENSION_tanh_RMSprop_mse.h5"
    },
    "ANN-MPL": {
        "Glass-Tension": "_out/best_model_ANN481Glass-Tension_softsign_AdamW_log_cosh_HU6.h5",
        "Glass-Flexure": "_out/best_model_ANN481Glass-Flexure_softsign_RMSprop_mae_HU4.h5",
        "Ceramic-Flexure": "_out/best_model_ANN481Ceramic-Flexure_tanh_RMSprop_log_cosh_HU6.h5",
        "Ceramic-Tension": "_out/best_model_ANN481Ceramic-Tension_swish_AdamW_huber_HU9.h5"
    },
    "XGBoost": {
        "Glass-Tension": "_out/best_xgboost_model_GLASS-TENSION.json",
        "Glass-Flexure": "_out/best_xgboost_model_GLASS-FLEXURE.json",
        "Ceramic-Flexure": "_out/best_xgboost_model_CERAMIC-FLEXURE.json",
        "Ceramic-Tension": "_out/best_xgboost_model_CERAMIC-TENSION.json"
    },
    "LightGBM": {
        "Glass-Tension": "_out/best_lightgbm_model_Glass-Tension.json",
        "Glass-Flexure": "_out/best_lightgbm_model_Glass-Flexure.json",
        "Ceramic-Flexure": "_out/best_lightgbm_model_Ceramic-Flexure.json",
        "Ceramic-Tension": "_out/best_lightgbm_model_Ceramic-Tension.json"
    }
}

normalization_files = {
    "Glass-Tension": "_out/normalization_params_Glass-Tension.json",
    "Glass-Flexure": "_out/normalization_params_Glass-Flexure.json",
    "Ceramic-Flexure": "_out/normalization_params_Ceramic-Flexure.json",
    "Ceramic-Tension": "_out/normalization_params_Ceramic-Tension.json"
}

custom_objects = {
    "mse": MeanSquaredError(),
    "mae": MeanAbsoluteError(),
    "huber": Huber(),
    "log_cosh": LogCosh()
}

# ======================
# HELPER FUNCTIONS
# ======================
def load_normalization_params(sheet):
    with open(normalization_files[sheet], "r") as f:
        return json.load(f)

def normalize_input(user_input, norm_params, input_columns):
    X_min = np.array([norm_params["X_min"][col] for col in input_columns])
    X_max = np.array([norm_params["X_max"][col] for col in input_columns])
    return (np.array(user_input) - X_min) / (X_max - X_min)

def denormalize_output(normalized_pred, norm_params):
    y_min, y_max = norm_params["y_min"], norm_params["y_max"]
    return y_min + normalized_pred * (y_max - y_min)

def load_best_model(category, sheet):
    model_path = model_files[category][sheet]
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: `{model_path}`")
        st.stop()

    if category in ["ANN-Practical Solution", "ANN-MPL"]:
        return load_model(model_path, custom_objects=custom_objects)
    elif category == "XGBoost":
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    elif category == "LightGBM":
        return lgb.Booster(model_file=model_path)

# ======================
# EXPLANATION SECTION
# ======================
def show_parameter_explanations(selected_sheet, norm_params):
    st.markdown("---")
    st.header("📚 Parameter Explanations")
    
    # Display test configuration diagram
    st.image("assets/fs_ceramic.jpg",
             caption="Fig. 1. Material Test Configuration Diagram",
             use_column_width=True)
    
    st.markdown("""
    ### Input Parameters Explanation:
    These dimensionless parameters capture the complex relationships between material properties and geometry:
    """)
    
    # Determine input parameters based on material type
    if "Glass" in selected_sheet:
        params = zip(glass_input_latex, [
            "Square root of thickness-to-flaw radius ratio - governs stress concentration",
            "Elastic modulus scaled by fracture toughness - represents material's brittleness",
            "Poisson's ratio - measures lateral strain response"
        ], glass_input_columns)
    else:
        params = zip(ceramic_input_latex, [
            "Square root of thickness-to-flaw radius ratio",
            "Elastic modulus scaled by fracture toughness",
            "Flaw aspect ratio (length/radius) - accounts for flaw geometry",
            "Poisson's ratio"
        ], ceramic_input_columns)

    # Display parameter details
    for i, (latex, desc, col) in enumerate(params):
        with st.expander(f"Parameter {i+1}: {latex}", expanded=False):
            st.markdown(f"""
            **Physical Meaning:**  
            {desc}
            
            **Mathematical Expression:**  
            {latex}
            
            **Typical Range:**  
            {norm_params['X_min'][col]:.4f} - {norm_params['X_max'][col]:.4f}
            """)

    # Output parameter explanation
    st.markdown("""
    ### Output Parameter Explanation:
    """)
    with st.expander(f"Predicted Strength: {target_latex}", expanded=True):
        st.markdown(f"""
        **Physical Meaning:**  
        Dimensionless strength at failure - critical stress intensity factor scaled by geometry
        
        **Mathematical Expression:**  
        {target_latex}  
        
        Where:  
        - σ = Failure stress (MPa)  
        - t = Material thickness (mm)  
        - K<sub>Ic</sub> = Fracture toughness (MPa·√m)  
        
        **Interpretation:**  
        Higher values indicate greater resistance to fracture propagation. 
        """, unsafe_allow_html=True)

# ======================
# MAIN APP
# ======================
def main():
    st.title("🧪 Material Strength AI Predictor")
    st.markdown("*Leveraging advanced machine learning models to predict material failure thresholds*")

    # Initialize prediction history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Model selection
    col1, col2 = st.columns([3, 2])
    with col1:
        selected_category = st.selectbox("**Select AI Model Type**", categories)
        selected_sheet = st.selectbox("**Material & Test Type**", sheet_names)

    # Load model and normalization parameters
    model = load_best_model(selected_category, selected_sheet)
    norm_params = load_normalization_params(selected_sheet)

    # Show parameter explanations
    show_parameter_explanations(selected_sheet, norm_params)

    # Input configuration
    input_columns = glass_input_columns if "Glass" in selected_sheet else ceramic_input_columns
    input_labels = glass_input_latex if "Glass" in selected_sheet else ceramic_input_latex

    # Sidebar inputs
    with st.sidebar:
        st.header("🧮 Input Parameters")
        user_inputs = []
        out_of_range = False

        for i, label in enumerate(input_labels):
            col = input_columns[i]
            min_val = norm_params["X_min"][col]
            max_val = norm_params["X_max"][col]

            with st.expander(f"Parameter {i+1}: {label}", expanded=True):
                value = st.number_input(
                    f"Value for {label}",
                    value=float((min_val + max_val)/2),
                    step=0.0001,
                    format="%.4f",
                    key=f"input_{i}"
                )
                
                # Range validation
                if value < min_val or value > max_val:
                    st.error(f"⚠️ Out of recommended range ({min_val:.4f} to {max_val:.4f})")
                    out_of_range = True
                else:
                    st.caption(f"Recommended range: {min_val:.4f} to {max_val:.4f}")
                
                user_inputs.append(value)

    # Prediction logic
    if st.button("🚀 Run Prediction", use_container_width=True):
        if out_of_range:
            st.error("⚠️ One or more parameters are outside recommended ranges!")
        else:
            with st.spinner("🧠 Analyzing material properties..."):
                try:
                    # Normalize and predict
                    normalized_input = normalize_input(user_inputs, norm_params, input_columns).reshape(1, -1)
                    
                    if selected_category == "XGBoost":
                        input_df = pd.DataFrame(normalized_input, columns=model.feature_names)
                        dmatrix = xgb.DMatrix(input_df)
                        prediction = model.predict(dmatrix)
                    elif selected_category == "LightGBM":
                        prediction = model.predict(normalized_input)
                    else:  # ANN models
                        prediction = model.predict(normalized_input)

                    # Denormalize and store result
                    real_prediction = denormalize_output(prediction[0], norm_params)
                    if isinstance(real_prediction, np.ndarray):
                        real_prediction = real_prediction.item()

                    # Update history
                    st.session_state.history.insert(0, {
                        "prediction": real_prediction,
                        "model": selected_category,
                        "material": selected_sheet,
                        "inputs": user_inputs
                    })

                    # Display results
                    st.success("✅ Prediction Generated!")
                    col_res1, col_res2 = st.columns([2, 3])
                    
                    with col_res1:
                        st.markdown(f"""
                        <div class="prediction-card success-animation">
                            <h3 style='color: #2ecc71; margin-bottom: 1rem;'>Predicted Strength</h3>
                            <div style='font-size: 2.5rem; font-weight: bold; color: #27ae60;'>
                                {real_prediction:.5f}
                            </div>
                            <div style='margin-top: 1rem;'>
                                {target_latex}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_res2:
                        st.markdown("### Prediction History")
                        for idx, entry in enumerate(st.session_state.history[:5]):
                            st.markdown(f"""
                            <div class="prediction-card">
                                <b>#{idx+1}</b> {entry['material']} ({entry['model']})<br>
                                <span style="color: #3498db;">{entry['prediction']:.5f}</span>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()
