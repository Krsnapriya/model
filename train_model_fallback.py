import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.model_selection import train_test_split

def train_and_save():
    print("Loading Multi-Class data...")
    df = pd.read_csv('data/pediatric_ehr_synthetic.csv')
    
    X = df.drop('Target_Label', axis=1)
    y = df['Target_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest model (Fallback)...")
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f"Model Accuracy: {acc:.4f}")
    
    output_path = "api/pediatric_model.joblib"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_and_save()
