import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set Page Config (Must be first)
st.set_page_config(page_title="Pediatric CDS", page_icon="🏥", layout="wide")

# --- Custom CSS for Medical UI ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card.danger {
        border-left-color: #dc3545;
        background-color: #fff5f5;
    }
    .metric-card.warning {
        border-left-color: #ffc107;
    }
    .metric-title {
        font-size: 0.9em;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #212529;
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_model():
    # Attempt to load model
    try:
        return joblib.load('api/pediatric_model.joblib')
    except:
        return None

def plot_benchmark(value, label, min_val, max_val, normal_min, normal_max):
    """Creates a visual benchmark of where the patient sits relative to normal."""
    fig, ax = plt.subplots(figsize=(6, 1.5))
    
    # Background range
    ax.barh(0, max_val-min_val, left=min_val, color='#EEEEEE', height=0.5)
    
    # Normal Range (Green Zone)
    ax.barh(0, normal_max-normal_min, left=normal_min, color='#C3E6CB', height=0.5, label='Normal')
    
    # Patient Value
    color = 'red' if (value < normal_min or value > normal_max) else 'green'
    ax.plot(value, 0, marker='o', color=color, markersize=12, markeredgecolor='white', markeredgewidth=2)
    
    # Labels
    ax.text(min_val, -0.4, str(min_val), fontsize=8, color='#666')
    ax.text(max_val, -0.4, str(max_val), fontsize=8, color='#666', ha='right')
    ax.text(value, 0.35, f"{value}", ha='center', fontweight='bold', color=color)
    
    ax.set_yticks([])
    ax.set_title(f"{label} Benchmark", fontsize=10, loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    return fig

# --- Main App ---
# --- Main App ---
model = load_model()

st.markdown("<h1 class='main-header'>🏥 Pediatric Clinical Decision Support</h1>", unsafe_allow_html=True)
st.markdown("---")

# Layout: 2 Columns (Inputs Left, Dashboard Right)
col_nav, col_main = st.columns([1, 2])

with col_nav:
    st.subheader("Patient Vitals")
    with st.container():
        # Update: Age in Years
        age_years = st.number_input("Age (Years)", 0.0, 18.0, 0.75, step=0.1, help="Input age in years (e.g. 1.5 for 18 months)")
        age_months = int(age_years * 12)
        
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female", index=1)
        
        st.markdown("#### Physiology")
        temp = st.number_input("Temperature (C)", 35.0, 43.0, 40.8, step=0.1)
        hr = st.slider("Heart Rate (bpm)", 40, 220, 158)
        rr = st.slider("Respiratory Rate (bpm)", 10, 80, 39)
        spo2 = st.slider("SpO2 (%)", 60, 100, 89)
        
        st.markdown("#### Labs")
        wbc = st.number_input("WBC Count (x10^9/L)", 1.0, 40.0, 24.2, step=0.1)
        glucose = st.number_input("Glucose (mg/dL)", 30.0, 800.0, 100.0, step=1.0)
        crp = st.number_input("CRP (mg/L)", 0.0, 400.0, 62.0, step=1.0)
        cough = st.checkbox("Cough Present")
        
        if st.button("🔎 Analyze Clinical Risk", type="primary", use_container_width=True):
             if model:
                # Prepare Data
                input_data = pd.DataFrame([{
                    "Age_Months": age_months,
                    "Gender": gender,
                    "Heart_Rate": hr,
                    "Respiratory_Rate": rr,
                    "SpO2": spo2,
                    "Temperature": temp,
                    "WBC_Count": wbc,
                    "Glucose": glucose,
                    "CRP": crp,
                    "Cough": int(cough)
                }])
                
                # Predict
                pred_class = model.predict(input_data)[0]
                pred_probs = model.predict_proba(input_data)[0]
                classes = {0: "Healthy", 1: "Sepsis", 2: "Pneumonia", 3: "Diabetes T1"}
                result = classes[pred_class]
                
                # Save to session state
                st.session_state['analyzed'] = True
                st.session_state['result'] = result
                st.session_state['pred_probs'] = pred_probs
                st.session_state['pred_class'] = pred_class
                st.session_state['input_data'] = input_data
                st.session_state['vitals_dict'] = input_data.iloc[0].to_dict()
             else:
                 st.error("Model not loaded.")

with col_main:
    if st.session_state.get('analyzed'):
        result = st.session_state['result']
        pred_probs = st.session_state['pred_probs']
        pred_class = st.session_state['pred_class']
        input_data = st.session_state['input_data']
        vitals_dict = st.session_state['vitals_dict']
        
        # --- Results Header ---
        r1, r2 = st.columns([2, 1])
        with r1:
            # Status Banner
            if result == "Healthy":
                st.success(f"### Assessment: {result}")
            else:
                st.error(f"### ⚠️ Assessment: {result.upper()} DETECTED")
                st.markdown(f"**Confidence**: {pred_probs[pred_class]*100:.1f}%")

        # --- Contextual Analytics Row ---
        st.markdown("### 📊 Physiological Benchmarks")
        b1, b2, b3 = st.columns(3)
        
        # Re-fetch current values for sliders to keep UI consistent, or use stored values
        # Using stored values for the plot specifically
        with b1:
            st.pyplot(plot_benchmark(vitals_dict['Heart_Rate'], "Heart Rate", 40, 220, 80, 130))
        with b2:
            st.pyplot(plot_benchmark(vitals_dict['Temperature'], "Temperature", 35, 43, 36.5, 37.5))
        with b3:
            st.pyplot(plot_benchmark(vitals_dict['SpO2'], "SpO2", 60, 100, 95, 100))
            
        st.markdown("---")
        
        # --- Explainability Row ---
        st.markdown("### 🧠 AI Clinical Reasoning (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            # Check shape again
            vals = None
            if isinstance(shap_values, list):
                vals = shap_values[pred_class][0]
            elif isinstance(shap_values, np.ndarray):
                    vals = shap_values[0, :, pred_class] if shap_values.ndim == 3 else shap_values[0, :]

            feature_names = input_data.columns
            indices = []

            if vals is not None:
                vals = np.array(vals, dtype=float).flatten()
                fig, ax = plt.subplots(figsize=(8, 3))
                top_k = min(5, len(vals))
                indices = np.argsort(np.abs(vals))[-top_k:]
                
                colors = ['#dc3545' if vals[i] > 0 else '#28a745' for i in indices]
                ax.barh(range(top_k), vals[indices], color=colors, align='center', height=0.6)
                ax.set_yticks(range(top_k))
                ax.set_yticklabels([feature_names[i] for i in indices])
                ax.set_xlabel("Impact on Risk Score")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate explanation: {e}")
            indices = [0] # Fallback for generating text

        st.markdown("---")
        
        # --- Clinical Action Protocols ---
        if result != "Healthy":
            st.subheader("📋 Recommended Action Protocol")
            with st.expander(f"Medical Guidelines for {result}", expanded=True):
                if result == "Sepsis":
                    st.markdown("""
                    **The Sepsis 6 (Start within 1 hour):**
                    1.  ✅ **Give Oxygen** to keep saturations > 94%
                    2.  ✅ **Take Blood Cultures**
                    3.  ✅ **Give IV Antibiotics**
                    4.  ✅ **Give Fluid Challenge**
                    5.  ✅ **Measure Lactate**
                    6.  ✅ **Measure Urine Output**
                    """)
                elif result == "Pneumonia":
                        st.markdown("""
                    **Pneumonia Pathway:**
                    1.  Assess oxygenation.
                    2.  Chest X-Ray required.
                    3.  Sputum culture.
                    4.  Initiate antibiotics (Amoxicillin first-line).
                    """)
                elif result == "Diabetes T1":
                        st.markdown("""
                    **DKA Protocol:**
                    1.  Check Ketones.
                    2.  Start IV Fluids (0.9% Saline).
                    3.  Monitor Potassium.
                    4.  Start Insulin infusion *after* fluids.
                    """)
        
        # --- Halo Bot (GenAI Clinical Note) ---
        # "Smart Clinical Assessment" renamed to "Halo Bot"
        
        try:
            import google.generativeai as genai
            
            # Configure API (Ensure this is valid or use fallback)
            genai.configure(api_key="AIzaSyA2JAf-osZjk0KI5bLtMPFtC7AjEv9FP04")
            model_gemini = genai.GenerativeModel('gemini-2.0-flash')
            
            st.markdown("### 🤖 Halo Bot: Clinical Note")
            
            # Only generate once per analysis to save quota and speed
            if 'ai_note' not in st.session_state:
                with st.spinner("Halo Bot is thinking..."):
                    # Construct Prompt
                    key_driver = feature_names[indices[-1]] if len(indices) > 0 else "Unknown"
                    case_desc = f"""
                    Patient: {age_years} year old {'Female' if gender == 1 else 'Male'}.
                    Vitals: HR {hr}, RR {rr}, SpO2 {spo2}%, Temp {temp}C.
                    Labs: WBC {wbc}, Glucose {glucose}, CRP {crp}.
                    Model Prediction: {result} ({pred_probs[pred_class]*100:.1f}% confidence).
                    KEY FINDING: One key driver was {key_driver}.
                    
                    Write a concise, professional medical note acting as a senior pediatric consultant. 
                    Explain WHY this patient is flagged as {result}. 
                    Highlight the critical abnormalities. 
                    Suggest 3 immediate next steps.
                    Keep it under 150 words.
                    """
                    
                    try:
                        response = model_gemini.generate_content(case_desc)
                        st.session_state['ai_note'] = response.text
                    except:
                        st.session_state['ai_note'] = None # Trigger fallback

            if st.session_state.get('ai_note'):
                st.info(st.session_state['ai_note'])
                
        except Exception as e:
            st.session_state['ai_note'] = None

        # Fallback if AI failed or API key invalid
        if not st.session_state.get('ai_note'):
            fallback_notes = {
                "Sepsis": f"**Assessment**: High suspicion of Sepsis based on tachycardia, fever, and elevated inflammatory markers. The patient meets SIRS criteria.\n\n**Plan**:\n1. Immediate septic screen (Blood cultures, Urine, CBP).\n2. Commence IV Ceftriaxone.\n3. Fluid bolus 20ml/kg.",
                "Pneumonia": f"**Assessment**: Clinical presentation consistent with Pneumonia given hypoxia and tachypnea. High risk for bacterial etiology.\n\n**Plan**:\n1. CXR to confirm consolidation.\n2. Supplemental O2.\n3. Start Amoxicillin.",
                "Diabetes T1": f"**Assessment**: Hyperglycemia raises concern for new-onset Type 1 Diabetes/DKA. \n\n**Plan**:\n1. Check urine/blood ketones immediately.\n2. Venous Blood Gas (VBG).\n3. Close monitoring.",
                "Healthy": "**Assessment**: Vitals and labs are within normal limits for age. Low risk of acute deterioration.\n\n**Plan**:\n1. Reassurance.\n2. Discharge with safety netting."
            }
            note = fallback_notes.get(result, "Clinical note unavailable.")
            st.session_state['ai_note'] = note
            st.info(note)
            if result != "Healthy":
                st.caption("⚠️ halo fallback mode active")

        # --- PDF Report Generation ---
        from fpdf import FPDF
        
        def safe_text(text):
            """Sanitize text for FPDF (Latin-1 only)"""
            if not text: return ""
            # Replace common incompatible characters
            replacements = {
                "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"
            }
            for k, v in replacements.items():
                text = text.replace(k, v)
            
            # Encode to latin-1, replacing errors with '?'
            return text.encode('latin-1', 'replace').decode('latin-1')

        def create_report(vitals, prediction, prob, ai_note=None):
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, safe_text("Pediatric Clinical Assessment Report"), 0, 1, 'C')
            
            # Result
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, safe_text(f"Assessment: {prediction.upper()}"), 0, 1)
            pdf.cell(0, 10, safe_text(f"Confidence: {prob:.1f}%"), 0, 1)
            
            # Vitals
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, safe_text("Patient Vitals:"), 0, 1)
            pdf.set_font("Arial", "", 12)
            for k, v in vitals.items():
                pdf.cell(0, 8, safe_text(f"{k}: {v}"), 0, 1)

            # AI Note
            if ai_note:
                pdf.ln(10)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, safe_text("Halo Bot Note:"), 0, 1)
                pdf.set_font("Arial", "I", 10)
                # Remove markdown
                clean_note = ai_note.replace('**', '').replace('__', '').replace('###', '')
                pdf.multi_cell(0, 5, safe_text(clean_note))
            
            return pdf.output(dest='S').encode('latin-1', 'replace')

        st.markdown("---")
        
        # Generate PDF Data
        try:
            pdf_data = create_report(
                vitals_dict, 
                result, 
                pred_probs[pred_class]*100, 
                st.session_state.get('ai_note', '')
            )
            
            st.download_button(
                label="📄 Download Medical Report (PDF)",
                data=pdf_data,
                file_name=f"pediatric_report_{result}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error preparing PDF: {e}")

    else:
        # Empty State
        st.info("👈 Enter patient vitals and click 'Analyze Clinical Risk' to start.")
        st.markdown("#### Quick Reference Ranges")
        st.dataframe(pd.DataFrame({
            "Vital": ["Heart Rate", "Resp Rate", "Temp", "WBC"],
            "Normal (Infant)": ["80-140", "20-40", "36.5-37.5", "5-15"],
            "Alarm": [">160", ">60", ">38.5", ">20"]
        }))
