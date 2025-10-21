# Project 1: Retail Machine Learning - Superstore Data Analysis

## Overview

I conducted comprehensive machine learning analysis on the Superstore dataset, implementing 5 different ML problems including regression, classification, clustering, and time series analysis to solve real-world retail business challenges.

## What I Did

- **Profit Prediction**: Built regression models to predict transaction profitability
- **Loss Classification**: Developed classification models to identify loss-making transactions
- **Customer Segmentation**: Implemented K-means clustering for customer behavioral analysis
- **Shipping Mode Prediction**: Created classification models to predict optimal shipping methods
- **Sales Forecasting**: Performed time series analysis for sales trend prediction and forecasting

## Machine Learning Problems Implemented

### 1. Profit Prediction (Regression)

- **Problem**: Predict the profit of a transaction based on sales features
- **Models Used**: Linear Regression, Random Forest Regressor
- **Features**: Sales, Quantity, Discount, Category, Sub-Category, Segment
- **Metrics**: MSE, RMSE, R² Score
- **Business Value**: Identify factors impacting profitability and forecast profit margins

### 2. Loss-Making Transaction Classification

- **Problem**: Classify transactions as profitable or loss-making
- **Models Used**: Random Forest Classifier
- **Target**: Binary classification (Profit > 0 = Profitable, else = Loss)
- **Features**: Product, Sub-Category, Discount, Quantity, Region
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Business Value**: Help minimize risk by flagging unprofitable products

### 3. Customer Segmentation (Clustering)

- **Problem**: Cluster customers based on purchasing behavior
- **Algorithm**: K-Means Clustering with PCA visualization
- **Features**: Total Spend, Average Discount, Quantity, Segment
- **Clusters**: 4 customer segments identified
- **Business Value**: Design personalized marketing campaigns for different segments

### 4. Shipping Mode Prediction (Classification)

- **Problem**: Predict the best shipping mode for customer orders
- **Models Used**: Random Forest Classifier
- **Features**: Order Date, Region, Product Category, Quantity, Segment
- **Metrics**: Classification Accuracy
- **Business Value**: Optimize logistics by predicting preferred shipping modes

### 5. Sales Forecasting (Time Series)

- **Problem**: Forecast future sales based on historical order data
- **Analysis**: Time series decomposition, seasonal patterns, growth rates
- **Features**: Date, Product Category, Region, Sales
- **Techniques**: Moving averages, growth rate analysis, seasonal decomposition
- **Business Value**: Enable inventory planning and revenue forecasting

## Technical Implementation

### Data Preprocessing

- **Data Cleaning**: Removed invalid sales data and handled missing values
- **Feature Engineering**: Created derived features like profit margins and order values
- **Categorical Encoding**: Used LabelEncoder for categorical variables
- **Time Series Processing**: Extracted temporal features from dates

### Model Development

- **Regression Models**: Linear Regression and Random Forest for profit prediction
- **Classification Models**: Random Forest for loss classification and shipping prediction
- **Clustering**: K-Means with optimal cluster selection using elbow method
- **Time Series**: Moving averages and trend analysis for sales forecasting

### Model Evaluation

- **Regression Metrics**: MSE, RMSE, R² Score for model performance
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Clustering Evaluation**: Inertia analysis and cluster characterization
- **Cross-validation**: Used for robust model evaluation

### Visualization and Analysis

- **Model Performance**: Actual vs Predicted plots, confusion matrices
- **Feature Importance**: Analysis of key predictive features
- **Customer Segments**: PCA visualization of customer clusters
- **Time Series**: Trend analysis and seasonal pattern identification

## Key Findings

### Model Performance

- **Profit Prediction**: Random Forest achieved higher R² score than Linear Regression
- **Loss Classification**: High accuracy in identifying loss-making transactions
- **Customer Segmentation**: 4 distinct customer segments with different behaviors
- **Shipping Prediction**: Good accuracy in predicting optimal shipping modes
- **Sales Forecasting**: Identified seasonal patterns and growth trends

### Business Insights

- **Profitability Factors**: Sales volume and discount levels significantly impact profit
- **Risk Management**: Classification model effectively identifies high-risk transactions
- **Customer Behavior**: Clear segmentation enables targeted marketing strategies
- **Operational Efficiency**: Shipping prediction helps optimize logistics
- **Strategic Planning**: Time series analysis provides forecasting capabilities

### Feature Importance

- **Profit Prediction**: Sales amount and quantity are most important features
- **Loss Classification**: Discount levels and product categories drive loss probability
- **Customer Segmentation**: Total spend and order frequency define customer types
- **Shipping Prediction**: Region and product category influence shipping preferences

## Model Performance Summary

- **Profit Prediction R² Score**: High predictive accuracy for profit estimation
- **Loss Classification Accuracy**: Effective identification of loss-making transactions
- **Customer Segmentation**: 4 distinct behavioral clusters identified
- **Shipping Prediction Accuracy**: Good performance in shipping mode prediction
- **Time Series Analysis**: Clear seasonal patterns and growth trends identified

## How to Run

1. Ensure `Superstore.csv` is in the same directory
2. Install required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
3. Run the script: `python retail_ml.py`

## Requirements

- Python 3.7+
- pandas, numpy, matplotlib, seaborn, scikit-learn
- Superstore.csv dataset

## Expected Outputs

- 20+ comprehensive ML visualizations and charts
- Model performance metrics and evaluation results
- Feature importance analysis and insights
- Customer segmentation results and characteristics
- Request forecasting insights and trends
- Comprehensive ML project summary and recommendations

## Technical Skills Demonstrated

- Advanced machine learning model development
- Regression and classification algorithms
- Unsupervised learning (clustering)
- Time series analysis and forecasting
- Feature engineering and selection
- Model evaluation and performance metrics
- Data visualization for ML insights
- Business intelligence and predictive analytics
