# Bike Rental Prediction

A machine learning project to predict hourly bike rental counts using weather and seasonal data from Washington D.C.'s Capital Bikeshare system.

## About This Project

I built this regression model to analyze and predict bike rental patterns using 2 years of historical data. The goal was to understand how weather conditions, time of day, and seasonal factors affect bike sharing demand.

## Why Bike Sharing?

Bike sharing systems have become really popular in cities worldwide. They're automated rental systems where you can pick up a bike at one station and drop it off at another. What makes them interesting for data science is that they generate detailed records of every trip - when it started, ended, weather conditions, etc. 

This data can help cities optimize bike distribution, predict maintenance needs, and understand urban mobility patterns.

## The Data

I used the Capital Bikeshare dataset from Washington D.C., covering 2011-2012 with 17,379 hourly records. The dataset includes:

**Weather & Environment:**
- Temperature and "feels like" temperature
- Humidity and wind speed  
- Weather conditions (clear, misty, rainy, etc.)

**Time Information:**
- Hour of day, day of week, month
- Season, year, holidays, working days

**Target Variable:**
- Total bike rentals per hour (what we're trying to predict)

## My Approach

### Data Preprocessing
- Checked for missing values (none found)
- Removed outliers by capping extreme rental counts at 95th percentile
- Created new features:
  - Temperature × humidity interaction
  - Peak hour indicator (rush hours: 7-8am, 5-6pm)
- Converted categorical variables to dummy variables
- Standardized numerical features

### Models Tested
I compared four different algorithms:

1. **Linear Regression** - Simple baseline model
2. **Ridge Regression** - Linear model with regularization  
3. **Decision Tree** - Non-linear model that can capture complex patterns
4. **Random Forest** - Ensemble of multiple decision trees

For the tree models, I used grid search to find the best parameters.

### Evaluation
- Split data 80/20 for training/testing
- Used 5-fold cross-validation
- Measured performance with R², MSE, and MAE

## Results

Here's how each model performed:

| Model | R² Score | Mean Absolute Error |
|-------|----------|-------------------|
| Linear Regression | 0.70 | 66.8 rentals |
| Ridge Regression | 0.70 | 66.7 rentals |  
| Decision Tree | 0.75 | 54.6 rentals |
| **Random Forest** | **0.88** | **37.8 rentals** |

Random Forest was clearly the winner, explaining 88% of the variance in bike rentals. On average, its predictions were off by about 38 bikes per hour.

**Best hyperparameters found:**
- Decision Tree: max_depth = 10
- Random Forest: 100 trees, max_depth = 15

## Visualizations

The code generates several plots to understand the results:

- `model_comparison.png` - Bar charts comparing all models
- `bike_rental_predictions.png` - Scatter plot of predicted vs actual rentals
- `feature_importance.png` - Which features matter most for predictions
- `residual_plot.png` - Analysis of prediction errors

## What I Learned

Some interesting insights from the feature importance analysis:
- Hour of day is crucial (rush hours drive demand)
- Temperature and season have major impacts
- Weather conditions significantly affect rentals
- The interaction between temperature and humidity helps predictions

## Running the Code

**Requirements:**
```
pandas
numpy  
scikit-learn
matplotlib
seaborn
```

**Setup:**
1. Download the bike sharing dataset (`hour.csv`)
2. Update the file path in the script
3. Run: `python bike_rental_prediction.py`

The script will train all models, print results, and save the visualization plots.

## Potential Applications

This type of model could help:
- Bike sharing companies optimize bike distribution
- City planners understand transportation patterns  
- Predict maintenance needs during low-demand periods
- Improve user experience by ensuring bike availability

## Files

- `bike_rental_prediction.py` - Main analysis script
- `hour.csv` - Dataset (17,379 records)
- Generated plots saved as PNG files

## Dataset Citation

This project uses data from:

Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg.

## Next Steps

Some ideas for improving this project:
- Try more advanced models like XGBoost or neural networks
- Add external data like local events or holidays
- Build a simple web app for making predictions
- Analyze daily patterns instead of hourly

---

Feel free to reach out if you have questions about the analysis or want to discuss the results!
