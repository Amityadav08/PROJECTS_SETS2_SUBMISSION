# Project 2: Healthcare Data Analysis

## Overview

I conducted comprehensive healthcare data analysis using SQL on a patient dataset. This project demonstrates advanced SQL techniques including window functions, stored procedures, and complex analytical queries for healthcare insights.

## What I Did

- Analyzed patient demographics, medical conditions, and treatment patterns
- Performed financial analysis of billing amounts and insurance providers
- Conducted blood type analysis and donation matching
- Created patient risk categorization system
- Developed hospital efficiency metrics
- Built stored procedures for blood donor-receiver matching
- Generated comprehensive healthcare insights and trends

## Database Structure

The healthcare dataset contains the following key fields:

- **Patient Info**: Name, Age, Gender, Blood_Type
- **Medical**: Medical_Condition, Medication, Test_Results
- **Hospital**: Hospital, Doctor, Room_Number, Admission_Type
- **Dates**: Date_of_Admission, Discharge_Date
- **Financial**: Billing_Amount, Insurance_Provider

## Key Analysis Areas

### 1. Data Overview & Basic Statistics

- Total patient records analysis
- Age distribution and demographics
- Patient count by medical conditions

### 2. Medical Conditions & Medications

- Most common medical conditions
- Medication prescription patterns
- Treatment effectiveness analysis

### 3. Insurance Providers & Hospitals

- Patient preferences for insurance providers
- Hospital popularity and capacity analysis
- Financial performance by provider

### 4. Financial Analysis & Duration

- Average billing amounts by medical condition
- Hospital stay duration analysis
- Cost per day calculations

### 5. Blood Type Analysis & Donation

- Blood type distribution analysis
- Universal donor and receiver identification
- Blood matching procedure development

### 6. Yearly Admissions & Insurance

- Hospital admission trends (2024-2025)
- Insurance provider financial analysis
- Seasonal admission patterns

### 7. Patient Risk Categorization

- Risk assessment based on test results
- Discharge readiness evaluation
- Follow-up care recommendations

### 8. Advanced Analytics

- Age group analysis
- Gender-based healthcare patterns
- Hospital efficiency metrics
- Cost analysis by condition

## Key SQL Techniques Used

- **Window Functions**: RANK(), DENSE_RANK() for ranking analysis
- **Date Functions**: DATEDIFF() for stay duration calculations
- **Case Statements**: Risk categorization logic
- **Aggregate Functions**: COUNT(), AVG(), MIN(), MAX(), SUM()
- **Joins**: Complex relationships between patient data
- **Stored Procedures**: Blood matching automation
- **Grouping**: Advanced GROUP BY with HAVING clauses

## Sample Insights Generated

- Most common medical conditions and their costs
- Hospital efficiency rankings based on stay duration
- Blood type distribution and donation potential
- Insurance provider financial performance
- Patient risk stratification for care planning
- Age-based healthcare utilization patterns

## How to Run

1. Import the healthcare dataset into MySQL
2. Run the `healthcare_data_analysis.sql` file
3. Execute individual queries to see specific insights
4. Use the Blood_Matcher stored procedure for donor matching

## Requirements

- MySQL Server
- Healthcare dataset (CSV import)
- MySQL Workbench (recommended)

## Expected Outputs

- Comprehensive patient demographics analysis
- Financial performance metrics
- Medical condition prevalence statistics
- Hospital efficiency rankings
- Blood donation matching capabilities
- Risk assessment categorizations

## Technical Skills Demonstrated

- Advanced SQL querying and optimization
- Healthcare data analysis and insights
- Stored procedure development
- Complex data relationships and joins
- Statistical analysis using SQL functions
- Business intelligence and reporting
- Data-driven healthcare decision making
