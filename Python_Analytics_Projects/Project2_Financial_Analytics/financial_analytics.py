import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading Financial Management dataset...")
df = pd.read_csv('Financial_Management_Dataset.csv')

# Data preprocessing
print("Preprocessing data...")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
df['Net_Flow'] = df['Credit'] - df['Debit']
df['Transaction_Amount'] = df[['Credit', 'Debit']].max(axis=1)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("FINANCIAL MANAGEMENT ANALYTICS")
print("=" * 60)

# 1. Monthly trends in spending and income across departments
print("\n1. MONTHLY SPENDING AND INCOME TRENDS BY DEPARTMENT")
print("-" * 60)
monthly_dept = df.groupby(['Year', 'Month', 'Department']).agg({
    'Credit': 'sum',
    'Debit': 'sum',
    'Net_Flow': 'sum'
}).reset_index()
monthly_dept['Date'] = pd.to_datetime(monthly_dept[['Year', 'Month']].assign(day=1))

# Display summary
dept_summary = df.groupby('Department').agg({
    'Credit': 'sum',
    'Debit': 'sum',
    'Net_Flow': 'sum',
    'Transaction ID': 'count'
}).reset_index()
dept_summary.columns = ['Department', 'Total_Credit', 'Total_Debit', 'Net_Flow', 'Transaction_Count']
dept_summary['Avg_Transaction'] = (dept_summary[['Total_Credit', 'Total_Debit']].max(axis=1) / dept_summary['Transaction_Count']).round(2)
print(dept_summary)

plt.figure(figsize=(15, 10))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: Monthly Credit Trends by Department
for dept in df['Department'].unique():
    dept_data = monthly_dept[monthly_dept['Department'] == dept]
    ax1.plot(dept_data['Date'], dept_data['Credit'], marker='o', label=dept, linewidth=2)
ax1.set_title('Monthly Credit Trends by Department', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Total Credit ($)', fontsize=12)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Monthly Debit Trends by Department
for dept in df['Department'].unique():
    dept_data = monthly_dept[monthly_dept['Department'] == dept]
    ax2.plot(dept_data['Date'], dept_data['Debit'], marker='s', label=dept, linewidth=2)
ax2.set_title('Monthly Debit Trends by Department', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Total Debit ($)', fontsize=12)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: Net Flow by Department
for dept in df['Department'].unique():
    dept_data = monthly_dept[monthly_dept['Department'] == dept]
    ax3.plot(dept_data['Date'], dept_data['Net_Flow'], marker='^', label=dept, linewidth=2)
ax3.set_title('Monthly Net Flow by Department', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12)
ax3.set_ylabel('Net Flow ($)', fontsize=12)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 4: Department-wise Total Transactions
sns.barplot(data=dept_summary, x='Department', y='Transaction_Count', ax=ax4)
ax4.set_title('Total Transactions by Department', fontsize=14, fontweight='bold')
ax4.set_xlabel('Department', fontsize=12)
ax4.set_ylabel('Transaction Count', fontsize=12)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 2. Budget analysis by department and transaction category
print("\n2. BUDGET ANALYSIS BY DEPARTMENT AND CATEGORY")
print("-" * 60)
budget_analysis = df.groupby(['Department', 'Category']).agg({
    'Debit': 'sum',
    'Credit': 'sum',
    'Transaction ID': 'count'
}).reset_index()
budget_analysis['Net_Amount'] = budget_analysis['Credit'] - budget_analysis['Debit']

# Calculate budget utilization (assuming budgets based on historical averages)
dept_avg_spending = df.groupby('Department')['Debit'].mean()
budget_analysis['Avg_Monthly_Spending'] = budget_analysis['Department'].map(dept_avg_spending)
budget_analysis['Budget_Utilization'] = (budget_analysis['Debit'] / budget_analysis['Avg_Monthly_Spending']).round(2)

print("Top spending categories by department:")
top_spending = budget_analysis.sort_values('Debit', ascending=False).head(10)
print(top_spending[['Department', 'Category', 'Debit', 'Budget_Utilization']])

plt.figure(figsize=(15, 8))
# Budget utilization heatmap
pivot_budget = budget_analysis.pivot_table(
    values='Budget_Utilization', 
    index='Department', 
    columns='Category', 
    fill_value=0
)
sns.heatmap(pivot_budget, annot=True, fmt='.2f', cmap='RdYlBu_r', center=1)
plt.title('Budget Utilization Heatmap by Department and Category', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 3. Cash flow and liquidity analysis
print("\n3. CASH FLOW AND LIQUIDITY ANALYSIS")
print("-" * 60)
cash_flow = df.groupby(['Year', 'Month']).agg({
    'Credit': 'sum',
    'Debit': 'sum',
    'Net_Flow': 'sum'
}).reset_index()
cash_flow['Date'] = pd.to_datetime(cash_flow[['Year', 'Month']].assign(day=1))
cash_flow['Cumulative_Flow'] = cash_flow['Net_Flow'].cumsum()

# Category-wise cash flow analysis
category_flow = df.groupby('Category').agg({
    'Credit': 'sum',
    'Debit': 'sum',
    'Net_Flow': 'sum'
}).reset_index()
category_flow = category_flow.sort_values('Net_Flow', ascending=False)

print("Category-wise Cash Flow Analysis:")
print(category_flow)

plt.figure(figsize=(15, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Monthly Cash Flow Trend
ax1.plot(cash_flow['Date'], cash_flow['Net_Flow'], marker='o', linewidth=2, label='Monthly Net Flow')
ax1.plot(cash_flow['Date'], cash_flow['Cumulative_Flow'], marker='s', linewidth=2, label='Cumulative Flow')
ax1.set_title('Monthly Cash Flow Trends', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Amount ($)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 2: Category-wise Net Flow
sns.barplot(data=category_flow.head(10), x='Net_Flow', y='Category', ax=ax2)
ax2.set_title('Top 10 Categories by Net Cash Flow', fontsize=14, fontweight='bold')
ax2.set_xlabel('Net Flow ($)', fontsize=12)
ax2.set_ylabel('Category', fontsize=12)

plt.tight_layout()
plt.show()

# 4. Performance monitoring and cost control
print("\n4. PERFORMANCE MONITORING AND COST CONTROL")
print("-" * 60)

# Transaction category trends
category_trends = df.groupby(['Category', 'Month']).agg({
    'Debit': 'sum',
    'Transaction ID': 'count'
}).reset_index()

# Top approvers analysis
approver_analysis = df.groupby('Approver').agg({
    'Debit': 'sum',
    'Credit': 'sum',
    'Transaction ID': 'count',
    'Transaction_Amount': 'mean'
}).reset_index()
approver_analysis.columns = ['Approver', 'Total_Debit', 'Total_Credit', 'Transaction_Count', 'Avg_Transaction_Amount']
approver_analysis = approver_analysis.sort_values('Total_Debit', ascending=False)

print("Top Approvers by Volume and Amount:")
print(approver_analysis.head(10))

# Non-operational expenses analysis
operational_categories = ['Salary', 'Utilities', 'Rent', 'Insurance']
non_operational = df[~df['Category'].isin(operational_categories)]
non_operational_analysis = non_operational.groupby('Category').agg({
    'Debit': 'sum',
    'Transaction ID': 'count'
}).reset_index()
non_operational_analysis['Percentage_of_Total'] = (non_operational_analysis['Debit'] / df['Debit'].sum() * 100).round(2)
non_operational_analysis = non_operational_analysis.sort_values('Debit', ascending=False)

print("\nNon-Operational Expenses Analysis:")
print(non_operational_analysis)

plt.figure(figsize=(15, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Top Approvers by Transaction Volume
sns.barplot(data=approver_analysis.head(10), x='Transaction_Count', y='Approver', ax=ax1)
ax1.set_title('Top 10 Approvers by Transaction Volume', fontsize=14, fontweight='bold')
ax1.set_xlabel('Transaction Count', fontsize=12)
ax1.set_ylabel('Approver', fontsize=12)

# Plot 2: Non-Operational Expenses Distribution
sns.barplot(data=non_operational_analysis.head(10), x='Percentage_of_Total', y='Category', ax=ax2)
ax2.set_title('Non-Operational Expenses (% of Total)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Percentage of Total Expenses (%)', fontsize=12)
ax2.set_ylabel('Category', fontsize=12)

plt.tight_layout()
plt.show()

# 5. Fraud and anomaly detection
print("\n5. FRAUD AND ANOMALY DETECTION")
print("-" * 60)

# Department-wise anomaly detection
dept_stats = df.groupby('Department')['Debit'].agg(['mean', 'std']).reset_index()
dept_stats['upper_bound'] = dept_stats['mean'] + (2 * dept_stats['std'])
dept_stats['lower_bound'] = dept_stats['mean'] - (2 * dept_stats['std'])

# Identify anomalies
df_anomalies = df.merge(dept_stats, on='Department')
anomalies = df_anomalies[
    (df_anomalies['Debit'] > df_anomalies['upper_bound']) | 
    (df_anomalies['Debit'] < df_anomalies['lower_bound'])
]

print(f"Total Anomalous Transactions Detected: {len(anomalies)}")
if len(anomalies) > 0:
    print("\nTop Anomalous Transactions:")
    anomaly_summary = anomalies.groupby(['Department', 'Category']).agg({
        'Debit': ['count', 'sum', 'mean'],
        'Transaction ID': 'count'
    }).reset_index()
    anomaly_summary.columns = ['Department', 'Category', 'Anomaly_Count', 'Total_Amount', 'Avg_Amount', 'Transaction_Count']
    print(anomaly_summary.sort_values('Total_Amount', ascending=False).head(10))

# Transaction frequency analysis
frequency_analysis = df.groupby(['Department', 'Category']).agg({
    'Transaction ID': 'count',
    'Debit': 'sum',
    'Date': ['min', 'max']
}).reset_index()
frequency_analysis.columns = ['Department', 'Category', 'Transaction_Count', 'Total_Debit', 'First_Date', 'Last_Date']
frequency_analysis['Date_Range'] = (pd.to_datetime(frequency_analysis['Last_Date']) - pd.to_datetime(frequency_analysis['First_Date'])).dt.days
frequency_analysis['Transactions_per_Day'] = (frequency_analysis['Transaction_Count'] / frequency_analysis['Date_Range']).round(2)

print("\nUnusual Transaction Frequencies:")
unusual_freq = frequency_analysis[frequency_analysis['Transactions_per_Day'] > 1].sort_values('Transactions_per_Day', ascending=False)
print(unusual_freq.head(10))

# 6. Transaction behavior and forecasting
print("\n6. TRANSACTION BEHAVIOR AND FORECASTING")
print("-" * 60)

# Monthly spending patterns
monthly_patterns = df.groupby('Month').agg({
    'Debit': 'sum',
    'Credit': 'sum',
    'Net_Flow': 'sum'
}).reset_index()

# Calculate month-over-month growth
monthly_patterns['MoM_Debit_Growth'] = monthly_patterns['Debit'].pct_change() * 100
monthly_patterns['MoM_Credit_Growth'] = monthly_patterns['Credit'].pct_change() * 100

print("Monthly Transaction Patterns:")
print(monthly_patterns.round(2))

# Account-wise analysis
account_analysis = df.groupby('Account').agg({
    'Credit': 'sum',
    'Debit': 'sum',
    'Net_Flow': 'sum',
    'Transaction ID': 'count'
}).reset_index()
account_analysis['Account_Type'] = account_analysis['Account'].str.extract(r'(\w+)')
account_analysis = account_analysis.sort_values('Net_Flow', ascending=False)

print("\nAccount-wise Performance:")
print(account_analysis.head(10))

plt.figure(figsize=(15, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Monthly Growth Rates
ax1.plot(monthly_patterns['Month'], monthly_patterns['MoM_Debit_Growth'], marker='o', label='Debit Growth %', linewidth=2)
ax1.plot(monthly_patterns['Month'], monthly_patterns['MoM_Credit_Growth'], marker='s', label='Credit Growth %', linewidth=2)
ax1.set_title('Month-over-Month Growth Rates', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Growth Rate (%)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 2: Account-wise Net Flow
top_accounts = account_analysis.head(10)
sns.barplot(data=top_accounts, x='Net_Flow', y='Account', ax=ax2)
ax2.set_title('Top 10 Accounts by Net Flow', fontsize=14, fontweight='bold')
ax2.set_xlabel('Net Flow ($)', fontsize=12)
ax2.set_ylabel('Account', fontsize=12)

plt.tight_layout()
plt.show()

# 7. Approvals and policy compliance
print("\n7. APPROVALS AND POLICY COMPLIANCE")
print("-" * 60)

# Approver consistency analysis
approver_consistency = df.groupby(['Approver', 'Department']).agg({
    'Debit': ['sum', 'count', 'mean'],
    'Credit': ['sum', 'count', 'mean']
}).reset_index()
approver_consistency.columns = ['Approver', 'Department', 'Total_Debit', 'Debit_Count', 'Avg_Debit', 'Total_Credit', 'Credit_Count', 'Avg_Credit']

# High-amount transaction approvers
high_amount_threshold = df['Transaction_Amount'].quantile(0.95)
high_amount_approvals = df[df['Transaction_Amount'] >= high_amount_threshold].groupby(['Approver', 'Department']).agg({
    'Transaction ID': 'count',
    'Transaction_Amount': 'sum'
}).reset_index()
high_amount_approvals = high_amount_approvals.sort_values('Transaction_Amount', ascending=False)

print("High-Amount Transaction Approvals (Top 95%):")
print(high_amount_approvals.head(10))

# Transaction type distribution
transaction_type_dist = df.groupby(['Category', 'Transaction_Type']).agg({
    'Transaction ID': 'count',
    'Transaction_Amount': 'sum'
}).reset_index()
transaction_type_dist = transaction_type_dist.sort_values('Transaction_Amount', ascending=False)

print("\nTransaction Type Distribution by Category:")
print(transaction_type_dist.head(10))

# 8. Key Insights Summary
print("\n" + "=" * 60)
print("KEY FINANCIAL INSIGHTS SUMMARY")
print("=" * 60)

print(f"\n📊 FINANCIAL OVERVIEW:")
print(f"Total Transactions: {len(df):,}")
print(f"Total Credit: ${df['Credit'].sum():,.2f}")
print(f"Total Debit: ${df['Debit'].sum():,.2f}")
print(f"Net Cash Flow: ${df['Net_Flow'].sum():,.2f}")
print(f"Average Transaction Amount: ${df['Transaction_Amount'].mean():,.2f}")

print(f"\n🏆 TOP PERFORMERS:")
best_dept = dept_summary.loc[dept_summary['Net_Flow'].idxmax()]
print(f"Best Department: {best_dept['Department']} (Net Flow: ${best_dept['Net_Flow']:,.2f})")
best_category = category_flow.iloc[0]
print(f"Best Category: {best_category['Category']} (Net Flow: ${best_category['Net_Flow']:,.2f})")
best_approver = approver_analysis.iloc[0]
print(f"Most Active Approver: {best_approver['Approver']} ({best_approver['Transaction_Count']} transactions)")

print(f"\n⚠️ AREAS OF CONCERN:")
worst_dept = dept_summary.loc[dept_summary['Net_Flow'].idxmin()]
print(f"Department with Lowest Net Flow: {worst_dept['Department']} (${worst_dept['Net_Flow']:,.2f})")
worst_category = category_flow.iloc[-1]
print(f"Category with Lowest Net Flow: {worst_category['Category']} (${worst_category['Net_Flow']:,.2f})")
print(f"Anomalous Transactions Detected: {len(anomalies)}")

print(f"\n📈 FINANCIAL RECOMMENDATIONS:")
print("1. Monitor departments with negative cash flows for budget adjustments")
print("2. Review high-amount transaction approvals for policy compliance")
print("3. Analyze anomalous transactions for potential fraud prevention")
print("4. Optimize spending in non-operational categories")
print("5. Implement monthly budget tracking and variance analysis")
print("6. Develop cash flow forecasting models for better liquidity management")
print("7. Enhance approval workflows for high-value transactions")

print("\n" + "=" * 60)
print("FINANCIAL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
