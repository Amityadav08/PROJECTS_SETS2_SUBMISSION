import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("RETAIL MACHINE LEARNING PROJECT")
print("=" * 60)

# Load dataset
print("Loading Superstore dataset...")
df = pd.read_csv('Superstore.csv', encoding='ISO-8859-1')

# Data preprocessing
print("Preprocessing data...")
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')
df = df[df['Sales'] > 0]  # Remove invalid sales data

# Feature engineering
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter
df['Profit_Margin'] = (df['Profit'] / df['Sales'] * 100).round(2)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Category', 'Sub-Category', 'Segment', 'Region', 'Ship Mode']
for col in categorical_columns:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. PROFIT PREDICTION (REGRESSION)
print("\n1. PROFIT PREDICTION USING REGRESSION")
print("-" * 50)

# Prepare features for profit prediction
features_reg = ['Sales', 'Quantity', 'Discount', 'Category_encoded', 'Sub-Category_encoded', 'Segment_encoded']
X_reg = df[features_reg]
y_reg = df['Profit']

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit models
lr_model.fit(X_train_reg, y_train_reg)
rf_model.fit(X_train_reg, y_train_reg)

# Make predictions
lr_pred = lr_model.predict(X_test_reg)
rf_pred = rf_model.predict(X_test_reg)

# Calculate metrics
lr_mse = mean_squared_error(y_test_reg, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test_reg, lr_pred)

rf_mse = mean_squared_error(y_test_reg, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test_reg, rf_pred)

print("Linear Regression Performance:")
print(f"  MSE: {lr_mse:.2f}")
print(f"  RMSE: {lr_rmse:.2f}")
print(f"  R² Score: {lr_r2:.4f}")

print("\nRandom Forest Performance:")
print(f"  MSE: {rf_mse:.2f}")
print(f"  RMSE: {rf_rmse:.2f}")
print(f"  R² Score: {rf_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features_reg,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance for Profit Prediction:")
print(feature_importance)

# Visualization
plt.figure(figsize=(15, 8))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: Actual vs Predicted (Linear Regression)
ax1.scatter(y_test_reg, lr_pred, alpha=0.6)
ax1.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Profit')
ax1.set_ylabel('Predicted Profit')
ax1.set_title('Linear Regression: Actual vs Predicted Profit')
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (Random Forest)
ax2.scatter(y_test_reg, rf_pred, alpha=0.6)
ax2.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Profit')
ax2.set_ylabel('Predicted Profit')
ax2.set_title('Random Forest: Actual vs Predicted Profit')
ax2.grid(True, alpha=0.3)

# Plot 3: Feature Importance
sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax3)
ax3.set_title('Feature Importance for Profit Prediction')
ax3.set_xlabel('Importance')

# Plot 4: Residuals (Random Forest)
residuals = y_test_reg - rf_pred
ax4.scatter(rf_pred, residuals, alpha=0.6)
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_xlabel('Predicted Profit')
ax4.set_ylabel('Residuals')
ax4.set_title('Residuals Plot (Random Forest)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. LOSS-MAKING TRANSACTION CLASSIFICATION
print("\n2. LOSS-MAKING TRANSACTION CLASSIFICATION")
print("-" * 50)

# Create binary classification target
df['Loss_Label'] = (df['Profit'] < 0).astype(int)
loss_percentage = df['Loss_Label'].mean() * 100
print(f"Percentage of loss-making transactions: {loss_percentage:.2f}%")

# Prepare features
features_cls = ['Sales', 'Quantity', 'Discount', 'Category_encoded', 'Sub-Category_encoded', 'Segment_encoded']
X_cls = df[features_cls]
y_cls = df['Loss_Label']

# Split data
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

# Train classifier
rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls.fit(X_train_cls, y_train_cls)

# Make predictions
y_pred_cls = rf_cls.predict(X_test_cls)
y_pred_proba = rf_cls.predict_proba(X_test_cls)[:, 1]

# Calculate metrics
cls_accuracy = accuracy_score(y_test_cls, y_pred_cls)
cls_report = classification_report(y_test_cls, y_pred_cls)

print(f"Classification Accuracy: {cls_accuracy:.4f}")
print("\nClassification Report:")
print(cls_report)

# Feature importance for classification
cls_feature_importance = pd.DataFrame({
    'Feature': features_cls,
    'Importance': rf_cls.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance for Loss Classification:")
print(cls_feature_importance)

# Visualization
plt.figure(figsize=(15, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test_cls, y_pred_cls)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix - Loss Prediction')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Plot 2: Feature Importance
sns.barplot(data=cls_feature_importance, x='Importance', y='Feature', ax=ax2)
ax2.set_title('Feature Importance for Loss Classification')
ax2.set_xlabel('Importance')

plt.tight_layout()
plt.show()

# 3. CUSTOMER SEGMENTATION USING CLUSTERING
print("\n3. CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
print("-" * 50)

# Create customer-level features
customer_features = df.groupby('Customer ID').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Profit': ['sum', 'mean'],
    'Discount': 'mean',
    'Quantity': ['sum', 'mean'],
    'Segment_encoded': 'first',
    'Region_encoded': 'first'
}).reset_index()

# Flatten column names
customer_features.columns = ['Customer_ID', 'Total_Sales', 'Avg_Sales', 'Order_Count', 
                           'Total_Profit', 'Avg_Profit', 'Avg_Discount', 'Total_Quantity', 
                           'Avg_Quantity', 'Segment', 'Region']

# Feature engineering for clustering
customer_features['Avg_Order_Value'] = customer_features['Total_Sales'] / customer_features['Order_Count']
customer_features['Profit_Margin'] = customer_features['Total_Profit'] / customer_features['Total_Sales']
customer_features['Orders_per_Month'] = customer_features['Order_Count'] / 12  # Assuming 12-month period

# Select features for clustering
clustering_features = ['Total_Sales', 'Avg_Order_Value', 'Avg_Discount', 'Orders_per_Month', 'Profit_Margin']
X_cluster = customer_features[clustering_features].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using elbow method
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Fit K-means with optimal clusters
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
customer_features['Cluster'] = cluster_labels

# Analyze clusters
cluster_analysis = customer_features.groupby('Cluster').agg({
    'Total_Sales': 'mean',
    'Avg_Order_Value': 'mean',
    'Avg_Discount': 'mean',
    'Orders_per_Month': 'mean',
    'Profit_Margin': 'mean',
    'Customer_ID': 'count'
}).round(2)
cluster_analysis.columns = ['Avg_Total_Sales', 'Avg_Order_Value', 'Avg_Discount', 'Avg_Orders_per_Month', 'Avg_Profit_Margin', 'Customer_Count']

print("Cluster Analysis:")
print(cluster_analysis)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
customer_features['PC1'] = X_pca[:, 0]
customer_features['PC2'] = X_pca[:, 1]

# Visualization
plt.figure(figsize=(15, 10))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))

# Plot 1: Elbow Method
ax1.plot(K_range, inertias, marker='o', linewidth=2)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True, alpha=0.3)

# Plot 2: Cluster Visualization (PCA)
scatter = ax2.scatter(customer_features['PC1'], customer_features['PC2'], 
                     c=customer_features['Cluster'], cmap='viridis', alpha=0.6)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax2.set_title('Customer Segmentation (PCA Visualization)')
ax2.legend(*scatter.legend_elements(), title="Clusters")

# Plot 3: Cluster Characteristics
cluster_analysis_plot = customer_features.groupby('Cluster')[clustering_features].mean()
sns.heatmap(cluster_analysis_plot.T, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax3)
ax3.set_title('Cluster Characteristics Heatmap')

plt.tight_layout()
plt.show()

# 4. SHIPPING MODE PREDICTION
print("\n4. SHIPPING MODE PREDICTION")
print("-" * 50)

# Prepare features for shipping mode prediction
features_ship = ['Sales', 'Quantity', 'Discount', 'Category_encoded', 'Sub-Category_encoded', 'Segment_encoded', 'Region_encoded']
X_ship = df[features_ship]
y_ship = df['Ship Mode_encoded']

# Split data
X_train_ship, X_test_ship, y_train_ship, y_test_ship = train_test_split(X_ship, y_ship, test_size=0.2, random_state=42)

# Train classifier
rf_ship = RandomForestClassifier(n_estimators=100, random_state=42)
rf_ship.fit(X_train_ship, y_train_ship)

# Make predictions
y_pred_ship = rf_ship.predict(X_test_ship)

# Calculate accuracy
ship_accuracy = accuracy_score(y_test_ship, y_pred_ship)
print(f"Shipping Mode Prediction Accuracy: {ship_accuracy:.4f}")

# Feature importance for shipping prediction
ship_feature_importance = pd.DataFrame({
    'Feature': features_ship,
    'Importance': rf_ship.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance for Shipping Mode Prediction:")
print(ship_feature_importance)

# Shipping mode distribution
ship_distribution = df['Ship Mode'].value_counts()
print(f"\nShipping Mode Distribution:")
print(ship_distribution)

# Visualization
plt.figure(figsize=(15, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Feature Importance
sns.barplot(data=ship_feature_importance, x='Importance', y='Feature', ax=ax1)
ax1.set_title('Feature Importance for Shipping Mode Prediction')
ax1.set_xlabel('Importance')

# Plot 2: Shipping Mode Distribution
sns.barplot(x=ship_distribution.index, y=ship_distribution.values, ax=ax2)
ax2.set_title('Distribution of Shipping Modes')
ax2.set_xlabel('Shipping Mode')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 5. SALES FORECASTING (TIME SERIES ANALYSIS)
print("\n5. SALES FORECASTING - TIME SERIES ANALYSIS")
print("-" * 50)

# Create time series data
df_sales_ts = df.groupby(df['Order Date'].dt.to_period('M')).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).reset_index()
df_sales_ts['Order Date'] = df_sales_ts['Order Date'].dt.to_timestamp()

# Add time features
df_sales_ts['Year'] = df_sales_ts['Order Date'].dt.year
df_sales_ts['Month'] = df_sales_ts['Order Date'].dt.month
df_sales_ts['Quarter'] = df_sales_ts['Order Date'].dt.quarter

# Calculate moving averages
df_sales_ts['Sales_MA_3'] = df_sales_ts['Sales'].rolling(window=3).mean()
df_sales_ts['Sales_MA_6'] = df_sales_ts['Sales'].rolling(window=6).mean()

# Growth rates
df_sales_ts['Sales_Growth'] = df_sales_ts['Sales'].pct_change() * 100
df_sales_ts['Sales_Growth_MA'] = df_sales_ts['Sales_Growth'].rolling(window=3).mean()

print("Time Series Summary Statistics:")
print(df_sales_ts.describe())

# Visualization
plt.figure(figsize=(20, 12))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))

# Plot 1: Sales Over Time
ax1.plot(df_sales_ts['Order Date'], df_sales_ts['Sales'], marker='o', linewidth=2, label='Monthly Sales')
ax1.plot(df_sales_ts['Order Date'], df_sales_ts['Sales_MA_3'], linewidth=2, label='3-Month MA')
ax1.plot(df_sales_ts['Order Date'], df_sales_ts['Sales_MA_6'], linewidth=2, label='6-Month MA')
ax1.set_title('Monthly Sales Trends with Moving Averages', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Sales ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Sales Growth Rate
ax2.plot(df_sales_ts['Order Date'], df_sales_ts['Sales_Growth'], marker='o', linewidth=2, label='Monthly Growth')
ax2.plot(df_sales_ts['Order Date'], df_sales_ts['Sales_Growth_MA'], linewidth=2, label='3-Month MA Growth')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Sales Growth Rate Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Growth Rate (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Seasonal Pattern
monthly_pattern = df_sales_ts.groupby('Month')['Sales'].mean()
ax3.bar(monthly_pattern.index, monthly_pattern.values)
ax3.set_title('Average Sales by Month (Seasonal Pattern)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Average Sales ($)')
ax3.grid(True, alpha=0.3)

# Plot 4: Quarterly Trends
quarterly_pattern = df_sales_ts.groupby('Quarter')['Sales'].mean()
ax4.bar(quarterly_pattern.index, quarterly_pattern.values)
ax4.set_title('Average Sales by Quarter', fontsize=14, fontweight='bold')
ax4.set_xlabel('Quarter')
ax4.set_ylabel('Average Sales ($)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. MODEL PERFORMANCE SUMMARY
print("\n6. MODEL PERFORMANCE SUMMARY")
print("-" * 50)

print("RETAIL ML PROJECT RESULTS:")
print("=" * 40)

print("\n1. PROFIT PREDICTION (REGRESSION):")
print(f"   Random Forest R² Score: {rf_r2:.4f}")
print(f"   Random Forest RMSE: ${rf_rmse:.2f}")
print(f"   Best performing model: {'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'}")

print("\n2. LOSS-MAKING TRANSACTION CLASSIFICATION:")
print(f"   Classification Accuracy: {cls_accuracy:.4f}")
print(f"   Loss Transaction Rate: {loss_percentage:.2f}%")

print("\n3. CUSTOMER SEGMENTATION:")
print(f"   Optimal Number of Clusters: {optimal_k}")
print(f"   Total Customers Analyzed: {len(customer_features)}")

print("\n4. SHIPPING MODE PREDICTION:")
print(f"   Prediction Accuracy: {ship_accuracy:.4f}")
print(f"   Most Common Shipping Mode: {ship_distribution.index[0]}")

print("\n5. TIME SERIES ANALYSIS:")
print(f"   Average Monthly Sales: ${df_sales_ts['Sales'].mean():,.2f}")
print(f"   Sales Growth Trend: {'Positive' if df_sales_ts['Sales_Growth'].mean() > 0 else 'Negative'}")

print("\n" + "=" * 60)
print("RETAIL ML PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
