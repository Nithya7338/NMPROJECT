import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Dataset/RealData/real_2016.csv')

# Extract features (X) and target (y)
X = df.iloc[:, :-1].values  # All columns except the last
y = df.iloc[:, -1].values   # Last column (AQI value)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store model performance metrics
model_metrics = {}

# Define and train models
models = {
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'linear_regression': LinearRegression(),
    'ridge_regression': Ridge(alpha=1.0),
    'lasso_regression': Lasso(alpha=0.1),
    'svr': SVR(kernel='rbf')
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Save metrics
    model_metrics[name] = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    print(f"  Training RMSE: {train_rmse:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    # Save the model
    model_path = f"models/{name}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to {model_path}")

# Save model metrics for reference
with open('models/model_metrics.pkl', 'wb') as f:
    pickle.dump(model_metrics, f)

print("\nAll models have been built and saved in the 'models' directory.")
print("Model performance metrics have been saved to 'models/model_metrics.pkl'.")

# Print summary of best mormance Summary (sorted by Test R²):")
sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['test_r2'], reverse=True)
for i, (name, metrics) in enumerate(sorted_models, 1):
    print(f"{i}. {name}: Test R² = {metrics['test_r2']:.4f}, Test RMSE = {metrics['test_rmse']:.2f}")
#dels based on test R²
print("\nModel Performance summary completed")