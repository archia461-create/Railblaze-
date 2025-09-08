import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor
import os

# ------------------ Load Data ------------------
def load_data(file_path):
    """
    Loads the CSV dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at {file_path}. Please check the path.")
    
    df = pd.read_csv(file_path, low_memory=False)
    print("✅ Loaded data with columns:", df.columns.tolist())

    # Ensure 'Current_Delay(mins)' exists
    if "Current_Delay(mins)" not in df.columns:
        print("⚠️ No 'Current_Delay(mins)' column found. Adding random delays...")
        df["Current_Delay(mins)"] = np.random.randint(0, 30, size=len(df))
    
    return df

# ------------------ Feature Engineering ------------------
def preprocess(df):
    features = ["Train_Type", "Passenger_Load", "Distance", 
                "Weather", "Congestion_Level", "Maintenance_Risk"]
    target = "Current_Delay(mins)"

    X = df[features].copy()
    y = df[target]

    # Encode categorical features
    encoders = {}
    for col in ["Train_Type", "Passenger_Load", "Weather", "Congestion_Level", "Maintenance_Risk"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, encoders

# ------------------ Train Model ------------------
def train_model(file_path, model_path="scripts/delay_predictor.pkl"):
    df = load_data(file_path)
    X, y, encoders = preprocess(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost Regressor
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1]
    }

    # Grid Search CV
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        cv=3, 
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 Model Evaluation:")
    print(f"   MAE: {mae:.2f} mins")
    print(f"   R² Score: {r2:.2f}")

    # Ensure scripts folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model + encoders
    joblib.dump({"model": best_model, "encoders": encoders}, model_path)
    print(f"\n✅ Model + encoders saved to {model_path}")

    return best_model

# ------------------ Run Example ------------------
if __name__ == "__main__":
    # Correct relative path from scripts/ to data/ folder at project root
    file_path = r"../data/train_schedule_upgraded.csv"

    # Train the model and save it
    train_model(file_path)
