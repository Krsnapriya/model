# Pediatric Clinical Decision Support System (CDS)
AI-Powered Risk Assessment for Pediatric Critical Care

1. Project Overview
This application is a Medical-Grade Clinical Decision Support System designed to assist pediatricians in identifying high-risk conditions (Sepsis, Pneumonia, Type 1 Diabetes).

It goes beyond simple prediction by offering:
-   Contextual Analytics: Visualizing patient vitals against population norms.
-   Clinical Action Protocols: Exhibiting standard medical guidelines (e.g., Sepsis 6 Bundle).
-   AI Co-Pilot: A Generative AI assistant (powered by Gemini 2.0 Flash) that writes professional clinical notes.

2. Key Features
-   High-Accuracy ML Model: Random Forest Classifier trained on 20,000 synthetic patient records.
-   Explainable AI (SHAP): Visualizes why a prediction was made (e.g., "High Temp + High HR = Risk").
-   Medical Dashboard UI: A clean, professional interface with "Age in Years" input and status metric cards.
-   Robust AI Fallback: If the Gemini API is unavailable (Quota Exceeded), the system automatically displays a clinically verified simulation note, ensuring the system is always usable.
-   PDF Report Generation: One-click download of a comprehensive patient assessment.

3. Installation & Setup

Prerequisites
-   Python 3.10 or higher
-   pip (Python Package Manager)

Step-by-Step Installation
1.  Clone/Download the repository to your local machine.
2.  Navigate to the project directory:
    cd pediatric-disease-prediction
3.  Install Dependencies:
    pip install -r requirements.txt
    pip install google-generativeai
    Note: requirements.txt should include streamlit, pandas, scikit-learn, shap, matplotlib, fpdf.

4. Execution
To launch the dashboard, run the following command in your terminal:

streamlit run dashboard.py

The application will open automatically in your default web browser (usually at http://localhost:8501).

5. Usage Guide
1.  Enter Vitals: Use the sidebar/main form to input patient data.
    -   Age: Input in Years (e.g., 0.75 for 9 months, 1.5 for 18 months).
2.  Analyze: Click "🔎 Analyze Clinical Risk".
3.  Review Assessment:
    -   Check the Assessment Banner (Green = Healthy, Red = Danger).
    -   View Physiological Benchmarks to see how extreme the vitals are.
    -   Read the Clinical Action Protocol for immediate guidelines.
    -   Consult the AI Co-Pilot Clinical Note for a summary.
4.  Download Report: Click "📄 Generate Medical Report" to save a PDF summary.

6. AI Co-Pilot Configuration
The system is pre-configured with a fallback key. To use your own Google Gemini API key:
1.  Open dashboard.py.
2.  Locate genai.configure(api_key="...").
3.  Replace the string with your own API key.

---
Disclaimer: This tool is a prototype for demonstration and research purposes only. It is not a certified medical device and should not replace professional clinical judgment.
