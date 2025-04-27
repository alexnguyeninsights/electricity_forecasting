# ⚡ Electricity Consumption Forecasting

## 1. Project Overview
This project aims to forecast short-term electricity consumption at the appliance level using time series data collected every 10 minutes. It applies statistical and machine learning techniques including Seasonal Naïve, AutoARIMA, and LSTM models to predict future energy usage patterns.

The project showcases skills in time series preprocessing, feature engineering, model building, and evaluation, designed for practical applications such as smart homes and energy management systems.


## 2. Dataset
* Source: energydata_complete.csv
* Description: Records indoor temperatures, humidity across different rooms, outdoor weather conditions, light usage, and appliance energy consumption.
* Period Covered: January 2016 – May 2016
* Frequency: Every 10 minutes


## 3. Tools and Libraries
* Python 3: Pandas, NumPy, Seaborn, Matplotlib
* Stats models: ARIMA, Seasonal Decompose, PMDARIMA (AutoARIMA), Scikit-learn (StandardScaler, PCA), TensorFlow/Keras (LSTM)


## 4. Methodology

### 4.1 Data Preprocessing
* Handled missing values and removed irrelevant features (rv1, rv2).
* Renamed columns for clarity (e.g., T1 → temp_kitchen).
* Engineered seconds_since_midnight to capture daily cyclic patterns.
* Set the datetime index and sorted chronologically.

### 4.2 Exploratory Data Analysis (EDA)
* Visualized appliance consumption trends and seasonality.
* Decomposed series into trend, seasonality, and residual components.
* Examined correlations among sensor features, weather features, and target variable.
* Identified multicollinearity among predictors.

### 4.3 Feature Engineering
* Scaled numerical features using StandardScaler.
* Reduced feature dimensions using Principal Component Analysis (PCA) (retaining 11 principal components).

### 4.4 Modelling
* Seasonal Naïve Model: Baseline forecasting using lag from the previous week.
* AutoARIMA: ARIMA model enhanced with Fourier terms for strong daily seasonality.
* LSTM Neural Network: Deep learning model using 144 timesteps (1 day) as input sequence length.

### 4.5 Model Evaluation
* Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) - Compared models on test data performance.
* Results:

| Model           | MAE                         | RMSE                        |
|-----------------|------------------------------|------------------------------|
| Seasonal Naïve  | ~61.7                        | ~124.1                      |
| AutoARIMA       | ~56.4                        | ~91.2                       |
| LSTM            | ~92.4                        | ~129.9                      |


=> Daily seasonality was dominant in appliance energy consumption. AutoARIMA and LSTM captured complex patterns better than simple baselines.


### 4.6 Key Takeaways
* Daily cyclicality is the strongest predictor of appliance usage.
* Scaling and dimensionality reduction (PCA) significantly improved model efficiency.
* Both traditional time series models and LSTM deep learning models have strengths depending on the forecast horizon and noise level.


### 4.7 Future Improvements
* Fine-tune LSTM hyperparameters for better accuracy and generalisation.
* Experiment with additional models like Prophet, XGBoost for time series.
* Incorporate external factors (e.g., public holidays, events, extended weather forecasts).
* Build a real-time deployment-ready dashboard for energy consumption prediction.


### 4.8 How to Run
* Clone this repository.
* Install required libraries: pip install pandas numpy seaborn matplotlib statsmodels scikit-learn pmdarima tensorflow
* Run the notebook cells sequentially to: Preprocess data, Visualize EDA, Train and test models, Evaluate performance


### 4.9 Author
📧 alexnguyen.insights@gmail.com
🔗 LinkedIn: AlexNguyenInsights
