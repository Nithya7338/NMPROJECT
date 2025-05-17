from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import numpy as np
import pandas as pd
import pickle
import random
import os
import time
import datetime
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages
app.config['UPLOAD_FOLDER'] = 'upload'
from werkzeug.utils import secure_filename

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the dataset
dataset = pd.read_csv('Dataset/RealData/real_2016.csv')

# Load available models from the models directory
def load_available_models():
	models = {}
	metrics = {}
	
	# Load model metrics if available
	try:
		with open('models/model_metrics.pkl', 'rb') as f:
			metrics = pickle.load(f)
	except:
		pass
	
	# Check for models in the models directory
	if os.path.exists('models'):
		for filename in os.listdir('models'):
			if filename.endswith('_model.pkl'):
				model_name = filename.replace('_model.pkl', '')
				try:
					with open(f'models/{filename}', 'rb') as f:
						models[model_name] = pickle.load(f)
				except:
					continue
	
	# If no models are found in the models directory, load the original model
	if not models and os.path.exists('random_forest_regression_model.pkl'):
		with open('random_forest_regression_model.pkl', 'rb') as f:
			models['random_forest'] = pickle.load(f)
	
	return models, metrics

# Load the models
models, model_metrics = load_available_models()

# Prepare model names and descriptions for dropdown
def prepare_model_options():
	model_options = []
	for name in models.keys():
		display_name = name.replace('_', ' ').title()
		
		# Add performance metrics if available
		description = ""
		if name in model_metrics:
			r2 = model_metrics[name]['test_r2']
			rmse = model_metrics[name]['test_rmse']
			description = f" (R² = {r2:.4f}, RMSE = {rmse:.2f})"
		
		model_options.append({
			'id': name,
			'name': display_name,
			'description': description
		})
	
	# Sort models by performance (if metrics available)
	if model_metrics:
		model_options.sort(key=lambda x: model_metrics.get(x['id'], {}).get('test_r2', 0), reverse=True)
	
	return model_options

@app.route('/')
def home():
	model_options = prepare_model_options()
	return render_template('home.html', model_options=model_options)

@app.route('/predict', methods=['POST'])
def predict():
	# Get user input values from the form
	features = []
	for i in range(8):  # The model expects 8 features
		feature_val = float(request.form.get(f'feature_{i}', 0))
		features.append(feature_val)
	
	# Get selected model (default to first available model)
	selected_model_name = request.form.get('model_select', list(models.keys())[0])
	if selected_model_name not in models:
		selected_model_name = list(models.keys())[0]
	
	selected_model = models[selected_model_name]
	
	# Convert to numpy array and reshape for prediction
	input_data = np.array(features).reshape(1, -1)
	
	# Check if user wants to compare all models
	compare_all = request.form.get('compare_all') == 'true'
	
	if compare_all:
		# Make predictions with all models
		predictions = {}
		for name, model in models.items():
			prediction = model.predict(input_data)
			predictions[name] = round(float(prediction[0]), 2)
		
		# Sort predictions by model performance if metrics available
		sorted_predictions = []
		for name, value in predictions.items():
			display_name = name.replace('_', ' ').title()
			
			# Add model metrics if available
			description = ""
			if name in model_metrics:
				description = f" (R² = {model_metrics[name]['test_r2']:.4f})"
				
			sorted_predictions.append({
				'name': display_name,
				'value': value,
				'description': description
			})
		
		# Sort by model performance if metrics available
		if model_metrics:
			sorted_predictions.sort(key=lambda x: model_metrics.get(x['name'].lower().replace(' ', '_'), {}).get('test_r2', 0), reverse=True)
		
		return render_template('compare_results.html', predictions=sorted_predictions)
	else:
		# Make prediction with the selected model
		prediction = selected_model.predict(input_data)
		prediction_value = round(float(prediction[0]), 2)
		
		# Get model metrics
		model_metrics_info = {}
		if selected_model_name in model_metrics:
			model_metrics_info = model_metrics[selected_model_name]
		
		return render_template('result.html', 
							  prediction=prediction_value, 
							  model_name=selected_model_name.replace('_', ' ').title(),
							  model_metrics=model_metrics_info)

@app.route('/sample_data', methods=['GET'])
def sample_data():
	# Select a random row from the dataset
	if len(dataset) > 0:
		random_index = random.randint(1, len(dataset) - 1)  # Skip header row
		sample = dataset.iloc[random_index]
		
		# Convert to a dictionary, handling NaN values
		sample_dict = {}
		# Only include the first 8 features (which is what the model expects)
		for i in range(8):
			value = sample[i] if i < len(sample) else 0
			sample_dict[i] = float(value) if not pd.isna(value) else 0.0
		
		return jsonify(sample_dict)
	else:
		return jsonify({"error": "Dataset is empty"})

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
	if request.method == 'POST':
		print("Form submitted via POST")
		print(f"Form data: {request.form}")
		
		# Get basic parameters
		model_type = request.form.get('model_type')
		custom_name = request.form.get('custom_name')
		dataset_choice = request.form.get('dataset_choice')
		test_size = int(request.form.get('test_size', 20)) / 100  # Convert percentage to fraction

		# Validate parameters
		if model_type not in ['rf', 'gb', 'lr', 'ridge', 'lasso', 'svr']:
			message = "Invalid model type selected"
			return render_template('train_model.html', message=message, message_type="message-error")

		# Load dataset
		try:
			if dataset_choice == 'built_in':
				built_in_dataset = request.form.get('built_in_dataset')
				if built_in_dataset not in ['real_2016.csv', 'real_2015.csv', 'real_2014.csv', 'real_2013.csv']:
					message = "Invalid built-in dataset selected"
					return render_template('train_model.html', message=message, message_type="message-error")
				
				df = pd.read_csv(f'Dataset/RealData/{built_in_dataset}')
			else:  # Upload custom dataset
				if 'custom_dataset' not in request.files:
					message = "No file uploaded"
					return render_template('train_model.html', message=message, message_type="message-error")
				
				file = request.files['custom_dataset']
				if file.filename == '':
					message = "No file selected"
					return render_template('train_model.html', message=message, message_type="message-error")
				
				if not file.filename.endswith('.csv'):
					message = "Only CSV files are supported"
					return render_template('train_model.html', message=message, message_type="message-error")
				
				# Save and load the file
				filename = secure_filename(file.filename)
				filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
				file.save(filepath)
				df = pd.read_csv(filepath)
		
			# Extract features and target
			X = df.iloc[:, :-1].values  # All columns except the last
			y = df.iloc[:, -1].values   # Last column (AQI value)
			
			# Get column names for feature importance
			feature_names = df.columns[:-1].tolist()

			# Split the data
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
			
			# Create model based on type and parameters
			model_params = {}
			if model_type == 'rf':
				n_estimators = int(request.form.get('rf_n_estimators', 100))
				min_samples_split = int(request.form.get('rf_min_samples_split', 2))
				
				max_depth_input = request.form.get('rf_max_depth', '')
				max_depth = None if max_depth_input == '' or max_depth_input == 'None' else int(max_depth_input)
				
				model = RandomForestRegressor(
					n_estimators=n_estimators,
					max_depth=max_depth,
					min_samples_split=min_samples_split,
					random_state=42
				)
				
				model_params = {
					'Number of Trees': n_estimators,
					'Maximum Depth': 'Unlimited' if max_depth is None else max_depth,
					'Minimum Samples Split': min_samples_split
				}
				
			elif model_type == 'gb':
				n_estimators = int(request.form.get('gb_n_estimators', 100))
				learning_rate = float(request.form.get('gb_learning_rate', 0.1))
				max_depth = int(request.form.get('gb_max_depth', 3))
				
				model = GradientBoostingRegressor(
					n_estimators=n_estimators,
					learning_rate=learning_rate,
					max_depth=max_depth,
					random_state=42
				)
				
				model_params = {
					'Number of Boosting Stages': n_estimators,
					'Learning Rate': learning_rate,
					'Maximum Depth': max_depth
				}
				
			elif model_type == 'lr':
				model = LinearRegression()
				model_params = {'Algorithm': 'Ordinary Least Squares'}
				
			elif model_type == 'ridge':
				alpha = float(request.form.get('ridge_alpha', 1.0))
				
				model = Ridge(alpha=alpha, random_state=42)
				
				model_params = {'Alpha (Regularization Strength)': alpha}
				
			elif model_type == 'lasso':
				alpha = float(request.form.get('lasso_alpha', 0.1))
				
				model = Lasso(alpha=alpha, random_state=42)
				
				model_params = {'Alpha (Regularization Strength)': alpha}
				
			elif model_type == 'svr':
				kernel = request.form.get('svr_kernel', 'rbf')
				C = float(request.form.get('svr_c', 1.0))
				epsilon = float(request.form.get('svr_epsilon', 0.1))
				
				model = SVR(kernel=kernel, C=C, epsilon=epsilon)
				
				model_params = {
					'Kernel': kernel,
					'C (Regularization Parameter)': C,
					'Epsilon': epsilon
				}
			
			# Train the model
			model.fit(X_train, y_train)
			
			# Make predictions
			y_train_pred = model.predict(X_train)
			y_test_pred = model.predict(X_test)
			
			# Calculate metrics
			train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
			test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
			train_r2 = r2_score(y_train, y_train_pred)
			test_r2 = r2_score(y_test, y_test_pred)
			
			metrics = {
				'train_rmse': train_rmse,
				'test_rmse': test_rmse,
				'train_r2': train_r2,
				'test_r2': test_r2
			}
			
			# Generate model name if custom name is not provided
			if not custom_name:
				timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
				model_name = f"{model_type}_{timestamp}"
			else:
				model_name = custom_name
			
			# Ensure name is suitable for filename
			model_name = secure_filename(model_name)
			
			# Save the model
			os.makedirs('models', exist_ok=True)
			model_path = f"models/{model_name}_model.pkl"
			with open(model_path, 'wb') as f:
				pickle.dump(model, f)
			
			# Update model metrics
			try:
				with open('models/model_metrics.pkl', 'rb') as f:
					model_metrics = pickle.load(f)
			except:
				model_metrics = {}
			
			model_metrics[model_name] = metrics
			
			with open('models/model_metrics.pkl', 'wb') as f:
				pickle.dump(model_metrics, f)
			
			# Reload models
			global models
			models, model_metrics = load_available_models()
			
			# Get feature importances if available
			feature_importances = []
			if hasattr(model, 'feature_importances_'):
				importances = model.feature_importances_
				# Sort and normalize importances
				max_importance = max(importances)
				for i, importance in enumerate(importances):
					feature_importances.append({
						'name': feature_names[i] if i < len(feature_names) else f"Feature {i}",
						'importance': importance,
						'percentage': (importance / max_importance) * 100
					})
				feature_importances.sort(key=lambda x: x['importance'], reverse=True)
			
			# Display the model training results
			display_name = model_name.replace('_', ' ').title()
			return render_template(
				'training_result.html',
				model_name=display_name,
				metrics=metrics,
				parameters=model_params,
				feature_importances=feature_importances
			)
			
		except Exception as e:
			message = f"An error occurred: {str(e)}"
			return render_template('train_model.html', message=message, message_type="message-error")
	
	# GET request - show the training form
	return render_template('train_model.html')

@app.route('/test')
def test():
	return render_template('test.html')

if __name__ == '__main__':
	app.run(debug=True, port=5000)