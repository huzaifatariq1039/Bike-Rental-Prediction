import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the dataset
file_path = 'C:/Users/OSL/Downloads/bike+sharing+dataset/hour.csv'
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please ensure the file exists at the specified path.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Exploratory Data Analysis (EDA)
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Drop unnecessary columns, including 'dteday'
try:
    data = data.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)
except KeyError as e:
    print(f"Error: One or more columns to drop not found: {e}")
    exit()

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 3. Feature Engineering
# Handle outliers: Cap target ('cnt') at 95th percentile
data['cnt'] = np.clip(data['cnt'], 0, data['cnt'].quantile(0.95))

# Add interaction term and peak hour feature
data['temp_hum_interaction'] = data['temp'] * data['hum']
data['is_peak_hour'] = data['hr'].isin([7, 8, 17, 18]).astype(int)

# 4. Preprocess the data
# Features (X) and target (y)
X = data.drop('cnt', axis=1)
y = data['cnt']

# Convert categorical variables to dummy variables
try:
    X = pd.get_dummies(X, columns=['season', 'weathersit', 'weekday', 'hr'], drop_first=True)
except KeyError as e:
    print(f"Error: One or more categorical columns not found: {e}")
    exit()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
except Exception as e:
    print(f"Error during scaling: {e}")
    exit()

# 5. Train models
print("\nStarting model training...")
try:
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    print("Linear Regression trained successfully.")

    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_test_scaled)
    print("Ridge Regression trained successfully.")

    # Decision Tree with GridSearchCV
    dt_params = {'max_depth': [3, 5, 7, 10]}
    dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=5, scoring='r2')
    dt_grid.fit(X_train_scaled, y_train)
    dt_pred = dt_grid.predict(X_test_scaled)
    print(f"Best Decision Tree Depth: {dt_grid.best_params_}")

    # Random Forest with GridSearchCV
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10, 15]}
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='r2')
    rf_grid.fit(X_train_scaled, y_train)
    rf_pred = rf_grid.predict(X_test_scaled)
    print(f"Best Random Forest Params: {rf_grid.best_params_}")
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# 6. Cross-Validation
print("\nStarting cross-validation...")
models = {
    'Linear Regression': lr_model,
    'Ridge Regression': ridge_model,
    'Decision Tree': dt_grid.best_estimator_,
    'Random Forest': rf_grid.best_estimator_
}
for name, model in models.items():
    try:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"{name} Cross-Validation R²: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    except Exception as e:
        print(f"Error during cross-validation for {name}: {e}")

# 7. Evaluate models
print("\nEvaluating models on test set...")
results = {}
for name, pred in {
    'Linear Regression': lr_pred,
    'Ridge Regression': ridge_pred,
    'Decision Tree': dt_pred,
    'Random Forest': rf_pred
}.items():
    try:
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        results[name] = {'MSE': mse, 'R²': r2, 'MAE': mae}
        print(f"\n{name}:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
    except Exception as e:
        print(f"Error during evaluation for {name}: {e}")

# 8. Visualize Model Comparison
try:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    models_names = list(results.keys())
    mse_values = [results[model]['MSE'] for model in models_names]
    r2_values = [results[model]['R²'] for model in models_names]
    mae_values = [results[model]['MAE'] for model in models_names]

    # MSE Plot
    ax1.bar(models_names, mse_values, color='#36A2EB')
    ax1.set_title('Mean Squared Error Comparison')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)

    # R² Plot
    ax2.bar(models_names, r2_values, color='#4BC0C0')
    ax2.set_title('R² Score Comparison')
    ax2.set_ylabel('R²')
    ax2.tick_params(axis='x', rotation=45)

    # MAE Plot
    ax3.bar(models_names, mae_values, color='#FFCE56')
    ax3.set_title('Mean Absolute Error Comparison')
    ax3.set_ylabel('MAE')
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
except Exception as e:
    print(f"Error during model comparison visualization: {e}")

# 9. Visualize Predictions vs Actual (Random Forest)
try:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, rf_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Rentals')
    plt.ylabel('Predicted Rentals')
    plt.title('Random Forest: Predicted vs Actual Bicycle Rentals')
    plt.tight_layout()
    plt.savefig('bike_rental_predictions.png')
    plt.close()
except Exception as e:
    print(f"Error during predictions vs actual visualization: {e}")

# 10. Visualize Feature Importance (Random Forest)
try:
    feature_importance = pd.Series(rf_grid.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title('Top 10 Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
except Exception as e:
    print(f"Error during feature importance visualization: {e}")

# 11. Visualize Residuals (Random Forest)
try:
    residuals = y_test - rf_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, color='purple', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Rentals')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Random Forest: Residual Plot')
    plt.tight_layout()
    plt.savefig('residual_plot.png')
    plt.close()
except Exception as e:
    print(f"Error during residual visualization: {e}")

print("\nVisualizations saved as 'model_comparison.png', 'bike_rental_predictions.png', 'feature_importance.png', and 'residual_plot.png'")