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
# CUSTOM STYLING (MOVED AFTER PAGE CONFIG)
# ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;500&display=swap');
    /* Keep the rest of your custom styles here */
</style>
""", unsafe_allow_html=True)


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
# REST OF IMPORTS
# ======================
import streamlit as st
import numpy as np
import os
import json
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, LogCosh
import joblib
# Debugging: Check LightGBM version after installation
#st.write(f"🔥 LightGBM Version: {lgb.__version__}")

# ======================
# CONSTANTS & CONFIG
# ======================
categories = ["Practical Solution", "MLP Solution", "XGBoost Solution", "LightGBM Solution"]
sheet_names = ["Glass-Tension", "Glass-Flexure", "Ceramic-Flexure", "Ceramic-Tension"]

ceramic_input_columns = ['sqrt(t/R)', 'E*sqrt(t)/Kic', 'ell/R', 'v']
ceramic_input_latex = [r'$\sqrt{\frac{t}{R}}$', r'$\frac{E\sqrt{t}}{K_{\mathrm{Ic}}}$', r'$\frac{\ell}{R}$', r'$\nu$']

glass_input_columns = ['sqrt(t/R)', 'E*sqrt(t)/Kic', 'v']
glass_input_latex = [r'$\sqrt{\dfrac{t}{R}}$', r'$\dfrac{E\sqrt{t}}{K_{\mathrm{Ic}}}$', r'$\nu$']

target_column = 'Sig*sqrt(t)/Kic'
target_latex = r'$\dfrac{\sigma\sqrt{t}}{K_{\mathrm{Ic}}}$'

model_files = {
    "Practical Solution": {
        "Glass-Tension": "_out/best_model_ANN_PracticalSol_Glass-Tension.h5",
        "Glass-Flexure": "_out/best_model_ANN_PracticalSol_Glass-Flexure.h5",
        "Ceramic-Flexure": "_out/best_model_ANN_PracticalSol_Ceramic-Flexure.h5",
        "Ceramic-Tension": "_out/best_model_ANN_PracticalSol_Ceramic-Tension.h5"
    },
    "MLP Solution": {
      "Glass-Tension": "_out/best_model_ANN_MPL_Glass-Tension_OPT-Adam_LR-0.022277_LOSS-mse_HU-7_ACT-softsign.keras",
      "Glass-Flexure": "_out/best_model_ANN_MPL_Glass-Flexure_OPT-Adam_LR-0.010222_LOSS-mse_HU-15_ACT-softsign.keras",
      "Ceramic-Flexure": "_out/best_model_ANN_MPL_Ceramic-Flexure_OPT-Adam_LR-0.019625_LOSS-mse_HU-16_ACT-softsign.keras",
      "Ceramic-Tension": "_out/best_model_ANN_MPL_Ceramic-Tension_OPT-Adam_LR-0.038810_LOSS-mse_HU-26_ACT-softsign.keras"
    },
    "XGBoost Solution": {
        "Glass-Tension": "_out/Glass-Tension_bayesian_xgboost_best_model.json",
        "Glass-Flexure": "_out/Glass-Flexure_bayesian_xgboost_best_model.json",
        "Ceramic-Flexure": "_out/Ceramic-Flexure_bayesian_xgboost_best_model.json",
        "Ceramic-Tension": "_out/Ceramic-Tension_bayesian_xgboost_best_model.json"
    },
    "LightGBM Solution": {
        "Glass-Tension": "_out/Glass-Tension_bayesian_lgbm_best_model.pkl",
        "Glass-Flexure": "_out/Glass-Flexure_bayesian_lgbm_best_model.pkl",
        "Ceramic-Flexure": "_out/Ceramic-Flexure_bayesian_lgbm_best_model.pkl",
        "Ceramic-Tension": "_out/Ceramic-Tension_bayesian_lgbm_best_model.pkl"
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
    # Scale to [0, 1] then to [-1, 1]
    return 2 * (np.array(user_input) - X_min) / (X_max - X_min) - 1


def denormalize_output(normalized_pred, norm_params):
    y_min, y_max = norm_params["y_min"], norm_params["y_max"]
    # Reverse the [-1, 1] normalization back to original scale
    return y_min + ((normalized_pred + 1) / 2) * (y_max - y_min)


import lightgbm as lgb
import joblib

# ======================
# CRITICAL LIGHTGBM FIXES
# ======================

def load_best_model(category, sheet):
    model_path = model_files[category][sheet]
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: `{model_path}`")
        st.stop()

    try:
        if category == "LightGBM Solution":
            model_data = joblib.load(model_path)
            
            # Handle different storage formats
            if isinstance(model_data, dict) and "model" in model_data:
                booster = model_data["model"]
                if isinstance(booster, lgb.Booster):
                    # Reconstruct sklearn wrapper with proper parameters
                    model = lgb.LGBMRegressor()
                    model._Booster = booster
                    model._n_features = booster.num_feature()
                    model.fitted_ = True
                    return model
                return booster
            return model_data

        elif category in ["Practical Solution", "MLP Solution"]:
            return load_model(model_path, custom_objects=custom_objects)

        elif category == "XGBoost":
            model = xgb.Booster()
            model.load_model(model_path)
            return model

    except Exception as e:
        st.error(f"❌ Error loading {category} model: {str(e)}")
        st.stop()




# ======================
# EXPLANATION SECTION
# ======================
def show_parameter_explanations(selected_sheet, norm_params):
    st.markdown("---")
    st.header("📚 Parameter Explanations")
    
    # Display test configuration diagram
    st.image("assets/fs_ceramic.jpg",
             caption="Fig. 1. Material Test Configuration Diagram",
             width=350)
        
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
    # Custom CSS for fonts and spacing
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
            * {
                font-family: 'Inter', sans-serif;
            }
            .stApp {
                margin-top: -2rem;
            }
            .main-column {
                padding-right: 1rem !important;
            }
            .st-emotion-cache-1kyxreq {
                padding: 1rem;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .history-item {
                border-left: 3px solid #2980b9;
                padding: 0.6rem;
                margin: 0.4rem 0;
                font-size: 0.85rem;
            }
            .title-container {
                margin-bottom: 1rem;
            }
            .title-hr {
                border: 1px solid #e0e0e0;
                margin: 4rem 0;
            }
            /* Align sidebar header with main title */
            .sidebar .st-emotion-cache-1v7f65g {
                padding-top: 1.4rem;
            }

            /* Reduce side margins */
            .block-container {
                padding-left: -0.5rem;
                padding-right: -0.5rem;
            }
            /* Make visualization column wider */
            .st-emotion-cache-1n76uvr {
                width: calc(120% - 0.1rem);
                margin-left: 0rem;
            }
            
        </style>
        
        <div class="title-container">
            <h2 style='font-size: 1.5rem; font-weight: 600; color: #2c3e50; margin: 0.2rem 0;'>
                Machine Learning-Based Prediction of Brittle Fracture Strength
            </h2>
            <hr class="title-hr">
        </div>
    """, unsafe_allow_html=True)


    # Initialize prediction history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Main layout columns
    col_main, col_vis = st.columns([35, 65], gap="large")
    
    with col_main:
        selected_category = st.selectbox("**Model Selection**", categories, key="model_type")
        selected_sheet = st.selectbox("**Material-Test Type Selection**", sheet_names, key="material_type")
        
        # Load model and parameters
        model = load_best_model(selected_category, selected_sheet)
        norm_params = load_normalization_params(selected_sheet)
        input_columns = glass_input_columns if "Glass" in selected_sheet else ceramic_input_columns
        input_labels = glass_input_latex if "Glass" in selected_sheet else ceramic_input_latex

    # Sidebar inputs
    with st.sidebar:
        st.header("Input Features", divider="gray")
        user_inputs = []
        out_of_range = False

        for i, label in enumerate(input_labels):
            col = input_columns[i]
            min_val = norm_params["X_min"][col]
            max_val = norm_params["X_max"][col]

            with st.container():
                value = st.number_input(
                    f"${label}$",
                    value=float((min_val + max_val) / 2),
                    min_value=float(min_val * 0.9),
                    max_value=float(max_val * 1.1),
                    step=0.001,
                    format="%.4f",
                    help=f"Recommended range: {min_val:.4f} to {max_val:.4f}"
                )
                
                if value < min_val or value > max_val:
                    out_of_range = True

            user_inputs.append(value)


    with col_main:
        if st.button("Calculate", type="primary", use_container_width=True):
            if out_of_range:
                st.warning("Inputs outside recommended ranges may affect accuracy", icon="⚠️")
            
            with st.spinner("Computing..."):
                try:
                        # ✅ Normalize Input
                        normalized_input = normalize_input(user_inputs, norm_params, input_columns).reshape(1, -1)

                        # ✅ Initialize `prediction` to avoid UnboundLocalError
                        prediction = None

                        # ✅ XGBoost Solution Processing
                        if selected_category == "XGBoost Solution":
                            input_df = pd.DataFrame(normalized_input, columns=input_columns)
                            dmatrix = xgb.DMatrix(input_df, feature_names=input_columns)
                            prediction = model.predict(dmatrix)

                        # ✅ LightGBM Solution Processing
                        elif selected_category == "LightGBM Solution":
                            # Universal input format
                            input_array = np.array(normalized_input, dtype=np.float32).reshape(1, -1)
                            
                            if isinstance(model, lgb.Booster):
                                # Use native booster prediction
                                prediction = model.predict(
                                    input_array,
                                    num_iteration=model.best_iteration,
                                    validate_features=Fals
                                )
                            else:
                                # Use sklearn-style prediction
                                prediction = model.predict(input_array)


                        # ✅ ANN Processing
                        elif selected_category in ["Practical Solution", "MLP Solution"]:
                            prediction = model.predict(normalized_input)

                        # 🚨 Ensure prediction was assigned before proceeding
                        if prediction is None:
                            raise ValueError(f"❌ Unsupported model type: {selected_category}")

                        # ✅ Denormalize & Convert Result
                        real_prediction = denormalize_output(prediction[0], norm_params)
                        if isinstance(real_prediction, np.ndarray):
                            real_prediction = real_prediction.item()

                        # ✅ Store in History
                        st.session_state.history.insert(0, {
                            "prediction": real_prediction,
                            "model": selected_category,
                            "material": selected_sheet,
                            "inputs": user_inputs
                        })
                        
                        # Display results
                        with st.container():
                                st.divider()
                                # Create columns for label + value
                                col_label, col_value = st.columns([2, 3])
                                
                                with col_label:
                                    # LaTeX label with matching font size
                                    st.metric(label=f"${target_latex}$",
                                        value="")
                                
                                with col_value:
                                    # Prediction value display
                                    st.markdown(
                                        f"<div style='font-size: 1.2rem; font-weight: 500; color: #2c3e50; padding: 0.1rem 0;'>"
                                        f"{real_prediction:.5f}"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                                
                                # Prediction history
                                st.subheader("Recent Predictions", divider="gray")
                                for idx, entry in enumerate(st.session_state.history[:5]):
                                    st.markdown(f"""
                                        <div class="history-item" style="padding:0.5rem; margin:0.25rem 0; font-size:0.9rem">
                                            <div style="color:#6c757d">{entry['material']}</div>
                                            <div style="font-weight:600; color:#2c3e50">{entry['prediction']:.5f}</div>
                                        </div>
                                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")
   

    with col_vis:
        # First row of images
        row1 = st.columns(2, gap="small")
        with row1[0]:
            st.image("assets/fs_ceramic.jpg",
                    caption="Ceramic",
                    use_container_width=True)
        with row1[1]:
            st.image("assets/fs_glass.jpg",
                    caption="Glass",
                    use_container_width=True)
        
        # Second row of images
        row2 = st.columns(2, gap="small")
        with row2[0]:
            st.image("assets/3PBT.png",
                    caption="Flexure",
                    use_container_width=True)
        with row2[1]:
            st.image("assets/Tension.png",
                    caption="Tension",
                    use_container_width=True)

if __name__ == "__main__":
    main()










