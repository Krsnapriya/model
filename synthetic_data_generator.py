import pandas as pd
import numpy as np
import random

def generate_pediatric_data(n_samples=5000, seed=42):
    """
    Generates a synthetic pediatric EHR dataset.
    
    Features:
    - Age (months): 0 to 216 (18 years)
    - Heart Rate (HR): bpm
    - Respiratory Rate (RR): bpm
    - SpO2: % saturation
    - Temperature: Celsius
    - WBC: White Blood Cell count (x10^9/L)
    - Comorbidity_Score: 0-5 scale
    
    Target:
    - Condition: 0 (Healthy/Other), 1 (Sepsis/Pneumonia/Critical)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    data = []
    
def generate_single_dataset(n_samples=1000, setting="General", seed=42):
    np.random.seed(seed)
    data = []
    
    for _ in range(n_samples):
        # Setting-specific distributions
        if setting == "PICU":
            # PICU: Sicker kids, higher prob of Sepsis/Pneumonia, more interventions
            age_months = np.random.randint(1, 144) # Skew younger
            label_prob = np.random.random()
            # 40% Healthy, 30% Sepsis, 20% Pneumonia, 10% Diabetes
            if label_prob > 0.6: label = 0
            elif label_prob > 0.3: label = 1 # Sepsis
            elif label_prob > 0.1: label = 2 # Pneumonia
            else: label = 3
        elif setting == "Rural":
            # Rural: Older kids, noisy data, less Sepsis
            age_months = np.random.randint(24, 216)
            label_prob = np.random.random()
            # 70% Healthy, 5% Sepsis, 20% Pneumonia, 5% Diabetes
            if label_prob > 0.3: label = 0
            elif label_prob > 0.25: label = 1
            elif label_prob > 0.05: label = 2
            else: label = 3
        else: # General Ward (Baseline)
            age_months = np.random.randint(1, 216)
            label_prob = np.random.random()
            # 60% Healthy, 10% Sepsis, 20% Pneumonia, 10% Diabetes
            if label_prob > 0.4: label = 0
            elif label_prob > 0.3: label = 1
            elif label_prob > 0.1: label = 2
            else: label = 3

        gender = np.random.choice([0, 1])

        # Base vitals logic (same as before but modified by setting)
        if age_months < 12: base_hr, base_rr = 120, 35
        elif age_months < 60: base_hr, base_rr = 100, 25
        else: base_hr, base_rr = 80, 20
        
        hr = int(np.random.normal(base_hr, 15))
        rr = int(np.random.normal(base_rr, 5))
        spo2 = int(np.random.normal(98, 2))
        spo2 = min(spo2, 100)
        temp = round(np.random.normal(37.0, 0.5), 1)
        wbc = round(np.random.normal(8.0, 2.5), 1)
        glucose = round(np.random.normal(90, 15), 1)
        crp = round(np.random.normal(2.0, 1.0), 1)
        cough = 0

        # Disease Signal Injection (Stronger signal in PICU due to advanced monitoring implication)
        signal_strength = 1.2 if setting == "PICU" else 1.0
        
        if label == 1: # Sepsis
            temp += np.random.uniform(1.5, 3.5)
            hr += np.random.randint(20, 50) * signal_strength
            rr += np.random.randint(5, 15)
            wbc += np.random.uniform(10.0, 20.0)
            crp += np.random.uniform(50.0, 150.0)
            spo2 -= np.random.randint(2, 8)
        elif label == 2: # Pneumonia
            cough = 1
            temp += np.random.uniform(1.0, 2.5)
            rr += np.random.randint(10, 25) * signal_strength
            spo2 -= np.random.randint(5, 12) * signal_strength
            wbc += np.random.uniform(5.0, 12.0)
        elif label == 3: # Diabetes
            glucose += np.random.uniform(150.0, 400.0)

        # Setting-specific noise
        if setting == "Rural":
            # Noisier measurements
            hr += np.random.randint(-5, 5)
            temp += np.random.normal(0, 0.2)
        
        data.append({
            "Age_Months": age_months,
            "Gender": gender,
            "Heart_Rate": hr,
            "Respiratory_Rate": rr,
            "SpO2": spo2,
            "Temperature": temp,
            "WBC_Count": wbc,
            "Glucose": glucose,
            "CRP": crp,
            "Cough": cough,
            "Target_Label": label,
            "Hospital_ID": setting
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating Multi-Center Pediatric Data...")
    
    df_ward = generate_single_dataset(n_samples=3000, setting="General", seed=42)
    df_picu = generate_single_dataset(n_samples=1000, setting="PICU", seed=43)
    df_rural = generate_single_dataset(n_samples=1000, setting="Rural", seed=44)
    
    # Save individual datasets
    df_ward.to_csv("pediatric_data_general_ward.csv", index=False)
    df_picu.to_csv("pediatric_data_picu.csv", index=False)
    df_rural.to_csv("pediatric_data_rural.csv", index=False)
    
    # Combined for simplified training if needed
    df_combined = pd.concat([df_ward, df_picu, df_rural])
    df_combined.to_csv("pediatric_ehr_synthetic.csv", index=False)
    
    print(f"Generated 3 datasets. Combined: {len(df_combined)} rows.")
    print("Distribution by Hospital:")
    print(df_combined["Hospital_ID"].value_counts())
