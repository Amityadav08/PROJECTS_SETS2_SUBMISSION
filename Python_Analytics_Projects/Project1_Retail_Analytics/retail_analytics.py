import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading Superstore dataset...")
df = pd.read_csv('Superstore.csv', encoding='ISO-8859-1')

# Data preprocessing
print("Preprocessing data...")
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')
df['Delivery Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Profit Margin'] = (df['Profit'] / df['Sales'] * 100).round(2)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("RETAIL ANALYTICS - SUPERSTORE DATA ANALYSIS")
print("=" * 60)

# 1. Profit margins by category and sub-category
print("\n1. PROFIT MARGINS BY CATEGORY AND SUB-CATEGORY")
print("-" * 50)
category_profit = df.groupby(['Category', 'Sub-Category'])[['Sales', 'Profit', 'Profit Margin']].sum().reset_index()
category_profit = category_profit.sort_values('Profit', ascending=False)
print(category_profit.head(10))

plt.figure(figsize=(15, 8))
sns.barplot(data=category_profit.head(15), x='Sub-Category', y='Profit', hue='Category')
plt.title('Top 15 Sub-Categories by Profit', fontsize=16, fontweight='bold')
plt.xlabel('Sub-Category', fontsize=12)
plt.ylabel('Total Profit ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Sales and profit by state
print("\n2. TOP STATES BY SALES AND PROFIT")
print("-" * 50)
state_analysis = df.groupby('State').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).reset_index()
state_analysis.columns = ['State', 'Total_Sales', 'Total_Profit', 'Order_Count']
state_analysis['Avg_Order_Value'] = (state_analysis['Total_Sales'] / state_analysis['Order_Count']).round(2)
state_analysis = state_analysis.sort_values('Total_Sales', ascending=False)
print(state_analysis.head(10))

plt.figure(figsize=(15, 8))
top_states = state_analysis.head(15)
sns.barplot(data=top_states, x='State', y='Total_Sales')
plt.title('Top 15 States by Total Sales', fontsize=16, fontweight='bold')
plt.xlabel('State', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Discount impact on profit
print("\n3. DISCOUNT IMPACT ON PROFITABILITY")
print("-" * 50)
discount_bins = pd.cut(df['Discount'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+'])
df['Discount_Range'] = discount_bins
discount_impact = df.groupby('Discount_Range').agg({
    'Sales': 'mean',
    'Profit': 'mean',
    'Profit Margin': 'mean',
    'Order ID': 'count'
}).reset_index()
discount_impact.columns = ['Discount_Range', 'Avg_Sales', 'Avg_Profit', 'Avg_Profit_Margin', 'Order_Count']
print(discount_impact)

plt.figure(figsize=(12, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Average Profit by Discount Range
sns.barplot(data=discount_impact, x='Discount_Range', y='Avg_Profit', ax=ax1)
ax1.set_title('Average Profit by Discount Range', fontsize=14, fontweight='bold')
ax1.set_xlabel('Discount Range', fontsize=12)
ax1.set_ylabel('Average Profit ($)', fontsize=12)

# Plot 2: Profit Margin by Discount Range
sns.barplot(data=discount_impact, x='Discount_Range', y='Avg_Profit_Margin', ax=ax2)
ax2.set_title('Average Profit Margin by Discount Range', fontsize=14, fontweight='bold')
ax2.set_xlabel('Discount Range', fontsize=12)
ax2.set_ylabel('Average Profit Margin (%)', fontsize=12)

plt.tight_layout()
plt.show()

# 4. Loss-making products
print("\n4. TOP 10 LOSS-MAKING PRODUCTS")
print("-" * 50)
product_analysis = df.groupby(['Product Name', 'Category', 'Sub-Category']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum',
    'Order ID': 'count'
}).reset_index()
product_analysis['Profit_per_Unit'] = (product_analysis['Profit'] / product_analysis['Quantity']).round(2)
loss_products = product_analysis[product_analysis['Profit'] < 0].sort_values('Profit').head(10)
print(loss_products[['Product Name', 'Category', 'Sub-Category', 'Sales', 'Profit', 'Quantity']])

# 5. Average delivery time by ship mode
print("\n5. DELIVERY PERFORMANCE BY SHIPPING MODE")
print("-" * 50)
delivery_analysis = df.groupby('Ship Mode').agg({
    'Delivery Time': ['mean', 'median', 'std'],
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).reset_index()
delivery_analysis.columns = ['Ship_Mode', 'Avg_Delivery_Time', 'Median_Delivery_Time', 'Std_Delivery_Time', 'Total_Sales', 'Total_Profit', 'Order_Count']
delivery_analysis['Avg_Order_Value'] = (delivery_analysis['Total_Sales'] / delivery_analysis['Order_Count']).round(2)
print(delivery_analysis)

plt.figure(figsize=(12, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Delivery Time by Ship Mode
sns.barplot(data=delivery_analysis, x='Ship_Mode', y='Avg_Delivery_Time', ax=ax1)
ax1.set_title('Average Delivery Time by Shipping Mode', fontsize=14, fontweight='bold')
ax1.set_xlabel('Shipping Mode', fontsize=12)
ax1.set_ylabel('Average Delivery Time (Days)', fontsize=12)

# Plot 2: Profit by Ship Mode
sns.barplot(data=delivery_analysis, x='Ship_Mode', y='Total_Profit', ax=ax2)
ax2.set_title('Total Profit by Shipping Mode', fontsize=14, fontweight='bold')
ax2.set_xlabel('Shipping Mode', fontsize=12)
ax2.set_ylabel('Total Profit ($)', fontsize=12)

plt.tight_layout()
plt.show()

# 6. Customer segment analysis
print("\n6. CUSTOMER SEGMENT ANALYSIS")
print("-" * 50)
segment_analysis = df.groupby('Segment').agg({
    'Sales': ['sum', 'mean'],
    'Profit': ['sum', 'mean'],
    'Order ID': 'count',
    'Customer ID': 'nunique'
}).reset_index()
segment_analysis.columns = ['Segment', 'Total_Sales', 'Avg_Sales', 'Total_Profit', 'Avg_Profit', 'Order_Count', 'Unique_Customers']
segment_analysis['Orders_per_Customer'] = (segment_analysis['Order_Count'] / segment_analysis['Unique_Customers']).round(2)
segment_analysis['Avg_Order_Value'] = (segment_analysis['Total_Sales'] / segment_analysis['Order_Count']).round(2)
print(segment_analysis)

# 7. Top profitable customers
print("\n7. TOP 10 PROFITABLE CUSTOMERS")
print("-" * 50)
customer_analysis = df.groupby(['Customer ID', 'Customer Name', 'Segment']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Discount': 'mean'
}).reset_index()
customer_analysis.columns = ['Customer_ID', 'Customer_Name', 'Segment', 'Total_Sales', 'Total_Profit', 'Order_Count', 'Avg_Discount']
customer_analysis['Avg_Order_Value'] = (customer_analysis['Total_Sales'] / customer_analysis['Order_Count']).round(2)
top_customers = customer_analysis.sort_values('Total_Profit', ascending=False).head(10)
print(top_customers[['Customer_Name', 'Segment', 'Total_Sales', 'Total_Profit', 'Order_Count', 'Avg_Order_Value']])

# 8. Monthly trends analysis
print("\n8. MONTHLY SALES AND PROFIT TRENDS")
print("-" * 50)
monthly_trends = df.groupby(['Year', 'Month']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Discount': 'mean'
}).reset_index()
monthly_trends['Date'] = pd.to_datetime(monthly_trends[['Year', 'Month']].assign(day=1))
monthly_trends['Profit_Margin'] = (monthly_trends['Profit'] / monthly_trends['Sales'] * 100).round(2)
print(monthly_trends.head(10))

plt.figure(figsize=(15, 8))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Plot 1: Sales Trend
sns.lineplot(data=monthly_trends, x='Month', y='Sales', hue='Year', marker='o', ax=ax1)
ax1.set_title('Monthly Sales Trend by Year', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Total Sales ($)', fontsize=12)
ax1.legend(title='Year')

# Plot 2: Profit Trend
sns.lineplot(data=monthly_trends, x='Month', y='Profit', hue='Year', marker='s', ax=ax2)
ax2.set_title('Monthly Profit Trend by Year', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Total Profit ($)', fontsize=12)
ax2.legend(title='Year')

plt.tight_layout()
plt.show()

# 9. Regional analysis
print("\n9. REGIONAL PERFORMANCE ANALYSIS")
print("-" * 50)
regional_analysis = df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Customer ID': 'nunique',
    'Delivery Time': 'mean'
}).reset_index()
regional_analysis.columns = ['Region', 'Total_Sales', 'Total_Profit', 'Order_Count', 'Unique_Customers', 'Avg_Delivery_Time']
regional_analysis['Avg_Order_Value'] = (regional_analysis['Total_Sales'] / regional_analysis['Order_Count']).round(2)
regional_analysis['Profit_Margin'] = (regional_analysis['Total_Profit'] / regional_analysis['Total_Sales'] * 100).round(2)
regional_analysis = regional_analysis.sort_values('Total_Sales', ascending=False)
print(regional_analysis)

# 10. Product performance analysis
print("\n10. SUB-CATEGORY PERFORMANCE ANALYSIS")
print("-" * 50)
subcategory_analysis = df.groupby('Sub-Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum',
    'Order ID': 'count'
}).reset_index()
subcategory_analysis['Sales_per_Unit'] = (subcategory_analysis['Sales'] / subcategory_analysis['Quantity']).round(2)
subcategory_analysis['Profit_per_Unit'] = (subcategory_analysis['Profit'] / subcategory_analysis['Quantity']).round(2)
subcategory_analysis['Profit_Margin'] = (subcategory_analysis['Profit'] / subcategory_analysis['Sales'] * 100).round(2)
subcategory_analysis = subcategory_analysis.sort_values('Profit', ascending=False)
print(subcategory_analysis.head(10))

# 11. Predictive Model for Profit
print("\n11. PROFIT PREDICTION MODEL")
print("-" * 50)

# Prepare features for modeling
X = df[['Sales', 'Quantity', 'Discount']].copy()
y = df['Profit'].copy()

# Remove any rows with missing values
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

print(f"\nFeature Importance (Coefficients):")
feature_names = ['Sales', 'Quantity', 'Discount']
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")

# 12. Key Insights Summary
print("\n" + "=" * 60)
print("KEY INSIGHTS SUMMARY")
print("=" * 60)

print(f"\n📊 DATASET OVERVIEW:")
print(f"Total Records: {len(df):,}")
print(f"Total Sales: ${df['Sales'].sum():,.2f}")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")
print(f"Overall Profit Margin: {(df['Profit'].sum() / df['Sales'].sum() * 100):.2f}%")

print(f"\n🏆 TOP PERFORMERS:")
best_category = category_profit.iloc[0]
print(f"Best Sub-Category: {best_category['Sub-Category']} (Profit: ${best_category['Profit']:,.2f})")
best_state = state_analysis.iloc[0]
print(f"Best State: {best_state['State']} (Sales: ${best_state['Total_Sales']:,.2f})")
best_customer = top_customers.iloc[0]
print(f"Best Customer: {best_customer['Customer_Name']} (Profit: ${best_customer['Total_Profit']:,.2f})")

print(f"\n⚠️ AREAS OF CONCERN:")
worst_category = category_profit.iloc[-1]
print(f"Worst Sub-Category: {worst_category['Sub-Category']} (Profit: ${worst_category['Profit']:,.2f})")
print(f"Loss-making Products: {len(loss_products)} products generating losses")

print(f"\n📈 BUSINESS RECOMMENDATIONS:")
print("1. Focus on high-profit sub-categories for inventory optimization")
print("2. Review loss-making products for pricing strategy adjustments")
print("3. Analyze discount impact - higher discounts may reduce profitability")
print("4. Optimize shipping modes based on delivery time vs. cost analysis")
print("5. Develop targeted strategies for top-performing customer segments")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
