import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading Employee Resignation dataset...")
df = pd.read_excel('Employee_Resignation_ProjectDataset.xlsx')

# Data preprocessing
print("Preprocessing data...")
# Convert date columns to datetime
date_columns = ['Date of Joining', 'Date of Resignation']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate tenure in days and years
if 'Date of Joining' in df.columns and 'Date of Resignation' in df.columns:
    df['Tenure_Days'] = (df['Date of Resignation'] - df['Date of Joining']).dt.days
    df['Tenure_Years'] = (df['Tenure_Days'] / 365).round(2)

# Extract year and month from dates
if 'Date of Resignation' in df.columns:
    df['Resignation_Year'] = df['Date of Resignation'].dt.year
    df['Resignation_Month'] = df['Date of Resignation'].dt.month
    df['Resignation_Quarter'] = df['Date of Resignation'].dt.quarter

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("EMPLOYEE RESIGNATION ANALYSIS")
print("=" * 60)

# 1. Dataset Overview
print("\n1. DATASET OVERVIEW")
print("-" * 50)
print(f"Total Records: {len(df):,}")
print(f"Dataset Shape: {df.shape}")
print(f"\nColumn Information:")
print(df.info())

print(f"\nFirst 5 records:")
print(df.head())

print(f"\nBasic Statistics:")
print(df.describe(include='all'))

# 2. Resignation Trends by Time
print("\n2. RESIGNATION TRENDS BY TIME")
print("-" * 50)

if 'Resignation_Year' in df.columns:
    yearly_resignations = df.groupby('Resignation_Year').size().reset_index(name='Resignation_Count')
    print("Yearly Resignation Trends:")
    print(yearly_resignations)
    
    plt.figure(figsize=(15, 10))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Yearly Resignation Trends
    sns.lineplot(data=yearly_resignations, x='Resignation_Year', y='Resignation_Count', marker='o', linewidth=3, ax=ax1)
    ax1.set_title('Yearly Resignation Trends', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Resignations', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Monthly Resignation Patterns
    monthly_resignations = df.groupby('Resignation_Month').size().reset_index(name='Resignation_Count')
    sns.barplot(data=monthly_resignations, x='Resignation_Month', y='Resignation_Count', ax=ax2)
    ax2.set_title('Monthly Resignation Patterns', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Number of Resignations', fontsize=12)
    
    # Plot 3: Quarterly Trends
    quarterly_resignations = df.groupby('Resignation_Quarter').size().reset_index(name='Resignation_Count')
    sns.barplot(data=quarterly_resignations, x='Resignation_Quarter', y='Resignation_Count', ax=ax3)
    ax3.set_title('Quarterly Resignation Trends', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Quarter', fontsize=12)
    ax3.set_ylabel('Number of Resignations', fontsize=12)
    
    # Plot 4: Tenure Distribution
    if 'Tenure_Years' in df.columns:
        tenure_bins = pd.cut(df['Tenure_Years'], bins=[0, 1, 2, 3, 5, 10, float('inf')], 
                           labels=['<1 Year', '1-2 Years', '2-3 Years', '3-5 Years', '5-10 Years', '10+ Years'])
        tenure_distribution = tenure_bins.value_counts().reset_index()
        tenure_distribution.columns = ['Tenure_Range', 'Count']
        
        sns.barplot(data=tenure_distribution, x='Count', y='Tenure_Range', ax=ax4)
        ax4.set_title('Tenure Distribution at Resignation', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Resignations', fontsize=12)
        ax4.set_ylabel('Tenure Range', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# 3. Department-wise Analysis
print("\n3. DEPARTMENT-WISE RESIGNATION ANALYSIS")
print("-" * 50)

if 'Department' in df.columns:
    dept_analysis = df.groupby('Department').agg({
        'Employee ID': 'count',
        'Tenure_Years': ['mean', 'median', 'std'] if 'Tenure_Years' in df.columns else None
    }).reset_index()
    
    if 'Tenure_Years' in df.columns:
        dept_analysis.columns = ['Department', 'Resignation_Count', 'Avg_Tenure', 'Median_Tenure', 'Std_Tenure']
    else:
        dept_analysis.columns = ['Department', 'Resignation_Count']
    
    dept_analysis = dept_analysis.sort_values('Resignation_Count', ascending=False)
    print("Department-wise Resignation Analysis:")
    print(dept_analysis)
    
    plt.figure(figsize=(15, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Resignation Count by Department
    sns.barplot(data=dept_analysis, x='Resignation_Count', y='Department', ax=ax1)
    ax1.set_title('Resignation Count by Department', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Resignations', fontsize=12)
    ax1.set_ylabel('Department', fontsize=12)
    
    # Plot 2: Average Tenure by Department
    if 'Avg_Tenure' in dept_analysis.columns:
        sns.barplot(data=dept_analysis, x='Avg_Tenure', y='Department', ax=ax2)
        ax2.set_title('Average Tenure by Department', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Average Tenure (Years)', fontsize=12)
        ax2.set_ylabel('Department', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# 4. Position/Role Analysis
print("\n4. POSITION/ROLE ANALYSIS")
print("-" * 50)

# Check for different possible column names for position/role
position_columns = ['Position', 'Role', 'Job Title', 'Designation', 'Title']
position_col = None
for col in position_columns:
    if col in df.columns:
        position_col = col
        break

if position_col:
    position_analysis = df.groupby(position_col).agg({
        'Employee ID': 'count',
        'Tenure_Years': ['mean', 'median'] if 'Tenure_Years' in df.columns else None
    }).reset_index()
    
    if 'Tenure_Years' in df.columns:
        position_analysis.columns = [position_col, 'Resignation_Count', 'Avg_Tenure', 'Median_Tenure']
    else:
        position_analysis.columns = [position_col, 'Resignation_Count']
    
    position_analysis = position_analysis.sort_values('Resignation_Count', ascending=False)
    print(f"Position-wise Resignation Analysis:")
    print(position_analysis.head(10))
    
    plt.figure(figsize=(15, 8))
    top_positions = position_analysis.head(10)
    sns.barplot(data=top_positions, x='Resignation_Count', y=position_col)
    plt.title(f'Top 10 Positions by Resignation Count', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Resignations', fontsize=12)
    plt.ylabel(position_col, fontsize=12)
    plt.tight_layout()
    plt.show()

# 5. Age Analysis
print("\n5. AGE ANALYSIS")
print("-" * 50)

if 'Age' in df.columns:
    age_analysis = df.groupby('Age').size().reset_index(name='Resignation_Count')
    age_analysis = age_analysis.sort_values('Age')
    
    # Age groups analysis
    age_bins = pd.cut(df['Age'], bins=[0, 25, 30, 35, 40, 45, 50, 60, 100], 
                     labels=['<25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-60', '60+'])
    age_group_analysis = age_bins.value_counts().reset_index()
    age_group_analysis.columns = ['Age_Group', 'Resignation_Count']
    
    print("Age Group Analysis:")
    print(age_group_analysis)
    
    plt.figure(figsize=(15, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Age Distribution
    sns.histplot(df['Age'], bins=20, ax=ax1)
    ax1.set_title('Age Distribution of Resigned Employees', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel('Number of Resignations', fontsize=12)
    
    # Plot 2: Age Group Analysis
    sns.barplot(data=age_group_analysis, x='Age_Group', y='Resignation_Count', ax=ax2)
    ax2.set_title('Resignations by Age Group', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Age Group', fontsize=12)
    ax2.set_ylabel('Number of Resignations', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# 6. Gender Analysis
print("\n6. GENDER ANALYSIS")
print("-" * 50)

if 'Gender' in df.columns:
    gender_analysis = df.groupby('Gender').agg({
        'Employee ID': 'count',
        'Age': ['mean', 'std'] if 'Age' in df.columns else None,
        'Tenure_Years': ['mean', 'median'] if 'Tenure_Years' in df.columns else None
    }).reset_index()
    
    if 'Age' in df.columns and 'Tenure_Years' in df.columns:
        gender_analysis.columns = ['Gender', 'Resignation_Count', 'Avg_Age', 'Std_Age', 'Avg_Tenure', 'Median_Tenure']
    elif 'Age' in df.columns:
        gender_analysis.columns = ['Gender', 'Resignation_Count', 'Avg_Age', 'Std_Age']
    else:
        gender_analysis.columns = ['Gender', 'Resignation_Count']
    
    gender_analysis['Resignation_Percentage'] = (gender_analysis['Resignation_Count'] / gender_analysis['Resignation_Count'].sum() * 100).round(2)
    print("Gender-wise Resignation Analysis:")
    print(gender_analysis)
    
    plt.figure(figsize=(15, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Resignation Count by Gender
    sns.barplot(data=gender_analysis, x='Gender', y='Resignation_Count', ax=ax1)
    ax1.set_title('Resignation Count by Gender', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Gender', fontsize=12)
    ax1.set_ylabel('Number of Resignations', fontsize=12)
    
    # Plot 2: Resignation Percentage by Gender
    sns.barplot(data=gender_analysis, x='Gender', y='Resignation_Percentage', ax=ax2)
    ax2.set_title('Resignation Percentage by Gender', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Gender', fontsize=12)
    ax2.set_ylabel('Percentage of Resignations (%)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# 7. Salary Analysis
print("\n7. SALARY ANALYSIS")
print("-" * 50)

salary_columns = ['Salary', 'Current Salary', 'Last Salary', 'Annual Salary', 'CTC']
salary_col = None
for col in salary_columns:
    if col in df.columns:
        salary_col = col
        break

if salary_col:
    salary_analysis = df.groupby(pd.cut(df[salary_col], bins=5)).size().reset_index(name='Resignation_Count')
    salary_analysis['Salary_Range'] = salary_analysis[salary_col].astype(str)
    
    print("Salary Range Analysis:")
    print(salary_analysis[['Salary_Range', 'Resignation_Count']])
    
    # Correlation analysis
    if 'Tenure_Years' in df.columns:
        salary_tenure_corr = df[salary_col].corr(df['Tenure_Years'])
        print(f"\nCorrelation between Salary and Tenure: {salary_tenure_corr:.4f}")
    
    plt.figure(figsize=(15, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Salary Distribution
    sns.histplot(df[salary_col], bins=20, ax=ax1)
    ax1.set_title(f'Salary Distribution of Resigned Employees', fontsize=14, fontweight='bold')
    ax1.set_xlabel(salary_col, fontsize=12)
    ax1.set_ylabel('Number of Resignations', fontsize=12)
    
    # Plot 2: Salary vs Tenure Scatter
    if 'Tenure_Years' in df.columns:
        sns.scatterplot(data=df, x=salary_col, y='Tenure_Years', ax=ax2)
        ax2.set_title('Salary vs Tenure at Resignation', fontsize=14, fontweight='bold')
        ax2.set_xlabel(salary_col, fontsize=12)
        ax2.set_ylabel('Tenure (Years)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# 8. Comprehensive Cross-Analysis
print("\n8. COMPREHENSIVE CROSS-ANALYSIS")
print("-" * 50)

# Department vs Position analysis
if 'Department' in df.columns and position_col:
    dept_position_analysis = df.groupby(['Department', position_col]).size().reset_index(name='Resignation_Count')
    dept_position_pivot = dept_position_analysis.pivot_table(
        values='Resignation_Count', 
        index='Department', 
        columns=position_col, 
        fill_value=0
    )
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(dept_position_pivot, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Resignation Heatmap: Department vs Position', fontsize=16, fontweight='bold')
    plt.xlabel(position_col, fontsize=12)
    plt.ylabel('Department', fontsize=12)
    plt.tight_layout()
    plt.show()

# 9. Key Insights Summary
print("\n" + "=" * 60)
print("KEY INSIGHTS SUMMARY")
print("=" * 60)

print(f"\n📊 DATASET OVERVIEW:")
print(f"Total Resignations Analyzed: {len(df):,}")
if 'Tenure_Years' in df.columns:
    print(f"Average Tenure at Resignation: {df['Tenure_Years'].mean():.2f} years")
    print(f"Median Tenure at Resignation: {df['Tenure_Years'].median():.2f} years")

if 'Department' in df.columns:
    top_dept = dept_analysis.iloc[0]
    print(f"Department with Highest Resignations: {top_dept['Department']} ({top_dept['Resignation_Count']} resignations)")

if 'Gender' in df.columns:
    gender_dist = df['Gender'].value_counts()
    print(f"Gender Distribution: {gender_dist.to_dict()}")

if 'Age' in df.columns:
    print(f"Average Age at Resignation: {df['Age'].mean():.1f} years")
    print(f"Age Range: {df['Age'].min()} - {df['Age'].max()} years")

print(f"\n🏆 KEY FINDINGS:")
if 'Resignation_Year' in df.columns:
    peak_year = yearly_resignations.loc[yearly_resignations['Resignation_Count'].idxmax()]
    print(f"Peak Resignation Year: {peak_year['Resignation_Year']} ({peak_year['Resignation_Count']} resignations)")

if 'Resignation_Month' in df.columns:
    peak_month = monthly_resignations.loc[monthly_resignations['Resignation_Count'].idxmax()]
    print(f"Peak Resignation Month: {peak_month['Resignation_Month']} ({peak_month['Resignation_Count']} resignations)")

print(f"\n⚠️ AREAS OF CONCERN:")
if 'Tenure_Years' in df.columns:
    short_tenure = df[df['Tenure_Years'] < 1].shape[0]
    print(f"Employees leaving within 1 year: {short_tenure} ({short_tenure/len(df)*100:.1f}%)")

if 'Department' in df.columns:
    high_turnover_depts = dept_analysis[dept_analysis['Resignation_Count'] > dept_analysis['Resignation_Count'].mean()]
    print(f"Departments with above-average turnover: {len(high_turnover_depts)}")

print(f"\n📈 HR RECOMMENDATIONS:")
print("1. Focus retention efforts on departments with highest resignation rates")
print("2. Investigate reasons for short-tenure resignations (< 1 year)")
print("3. Analyze peak resignation periods for proactive retention planning")
print("4. Review compensation and career development opportunities")
print("5. Implement exit interview analysis for deeper insights")
print("6. Develop targeted retention strategies for high-risk employee groups")
print("7. Monitor resignation trends for early warning indicators")

print("\n" + "=" * 60)
print("EMPLOYEE RESIGNATION ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
