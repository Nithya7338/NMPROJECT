# AQI Prediction

A machine learning application that predicts Air Quality Index based on weather parameters using multiple regression models.

## About The Project

Human health is greatly influenced by air quality. A wide range of health problems, particularly in children, are brought on by air quality degradation. This project uses machine learning approaches to construct models that predict air quality scores based on meteorological parameters.

## Project Structure

```
AQI-main/
├── app.py                      # Flask application
├── build_model.py              # Script to train and evaluate multiple ML models
├── Dataset/                    # Organized data directory
│   ├── AQI/                    # AQI data files
│   └── RealData/               # Weather data files
├── models/                     # Saved ML models directory
├── static/                     # Static assets for web app
│   └── css/                    # CSS stylesheets
├── templates/                  # HTML templates
│   ├── compare_results.html    # Template for comparing model predictions
│   ├── home.html               # Home page with input form
│   ├── result.html             # Result page for individual model prediction
│   ├── train_model.html        # Template for training custom models
│   └── training_result.html    # Template for displaying training results
├── uploads/                    # Directory for user-uploaded datasets
└── requirements.txt            # Python dependencies
```

## How to run the application

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Build the models (optional, sample models included):
   ```
   python build_model.py
   ```
3. Run the Flask application:
   ```
   python app.py
   ```
4. Open your web browser and go to: http://127.0.0.1:5000

## Features

### AQI Prediction

- Input form for 8 weather parameters
- Multiple prediction models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Support Vector Regression (SVR)
- Option to select a specific model or compare all models
- Sample data button to populate the form with real data
- Detailed results with AQI category and health implications
- Model performance metrics

### Custom Model Training

- Train your own machine learning models using built-in or custom datasets
- Choose from 6 different regression algorithms
- Customize hyperparameters for each algorithm:
  - **Random Forest**: Number of trees, maximum depth, minimum samples split
  - **Gradient Boosting**: Number of boosting stages, learning rate, maximum depth
  - **Ridge Regression**: Regularization strength (alpha)
  - **Lasso Regression**: Regularization strength (alpha)
  - **SVR**: Kernel type, C parameter, epsilon
- Automatically calculates and displays model performance metrics:
  - R² score (coefficient of determination)
  - Root Mean Squared Error (RMSE)
- Feature importance visualization for tree-based models
- Use trained models immediately for predictions

## Technologies used

- Python 3.13
- Flask for web framework
- scikit-learn for machine learning models
- NumPy and Pandas for data manipulation
- Pickle for model serialization

## Model Evaluation

The application includes a model evaluation system that:

1. Trains multiple regression models on historical data
2. Evaluates each model using train/test split methodology
3. Reports key metrics (RMSE and R²) for each model
4. Sorts models by performance for easy selection
5. Allows comparing predictions from all models side-by-side
