-- Project 2: Healthcare Data Analysis
-- Student: Amit Yadav
-- Course: Data Analysis and Data Science
-- Batch: 122

-- Creating Database named Healthcare
CREATE DATABASE IF NOT EXISTS Healthcare;
USE Healthcare;

-- Note: The healthcare dataset should be imported as a table named 'healthcare'
-- This analysis assumes the dataset is already loaded with the following structure:
-- Name, Age, Gender, Blood_Type, Medical_Condition, Date_of_Admission, Doctor, 
-- Hospital, Insurance_Provider, Billing_Amount, Room_Number, Admission_Type, 
-- Discharge_Date, Medication, Test_Results

-- 1. Data Overview & Basic Statistics
-- Counting Total Records in Database
SELECT COUNT(*) as Total_Records FROM healthcare;

-- Finding maximum age of patient admitted
SELECT MAX(age) as Maximum_Age FROM healthcare;

-- Finding Average age of hospitalized patients
SELECT ROUND(AVG(age), 0) as Average_Age FROM healthcare;

-- 2. Medical Conditions & Medications Analysis
-- Calculating Patients Hospitalized Age-wise from Maximum to Minimum
SELECT AGE, COUNT(AGE) AS Total_Patients
FROM Healthcare
GROUP BY age
ORDER BY AGE DESC;

-- Finding Count of Medical Condition of patients and listing by maximum number of patients
SELECT Medical_Condition, COUNT(Medical_Condition) as Total_Patients 
FROM healthcare
GROUP BY Medical_Condition
ORDER BY Total_patients DESC;

-- Finding Rank & Maximum number of medicines recommended to patients based on Medical Condition
SELECT Medical_Condition, Medication, COUNT(medication) as Total_Medications_to_Patients, 
       RANK() OVER(PARTITION BY Medical_Condition ORDER BY COUNT(medication) DESC) as Rank_Medicine
FROM Healthcare
GROUP BY 1,2
ORDER BY 1;

-- 3. Insurance Providers & Hospitals Analysis
-- Most preferred Insurance Provider by Patients Hospitalized
SELECT Insurance_Provider, COUNT(Insurance_Provider) AS Total_Patients 
FROM Healthcare
GROUP BY Insurance_Provider
ORDER BY Total_Patients DESC;

-- Finding out most preferred Hospital
SELECT Hospital, COUNT(hospital) AS Total_Patients 
FROM Healthcare
GROUP BY Hospital
ORDER BY Total_Patients DESC;

-- 4. Financial Analysis & Duration of Hospitalization
-- Identifying Average Billing Amount by Medical Condition
SELECT Medical_Condition, ROUND(AVG(Billing_Amount), 2) AS Avg_Billing_Amount
FROM Healthcare
GROUP BY Medical_Condition
ORDER BY Avg_Billing_Amount DESC;

-- Finding Billing Amount of patients admitted and number of days spent in respective hospital
SELECT Medical_Condition, Name, Hospital, 
       DATEDIFF(Discharge_date, Date_of_Admission) as Number_of_Days, 
       Billing_Amount
FROM Healthcare
ORDER BY Medical_Condition;

-- Finding Total number of days spent by patient in hospital for given medical condition
SELECT Name, Medical_Condition, ROUND(Billing_Amount, 2) as Billing_Amount, 
       Hospital, DATEDIFF(Discharge_Date, Date_of_Admission) as Total_Hospitalized_days
FROM Healthcare
ORDER BY Total_Hospitalized_days DESC;

-- 5. Blood Type Analysis & Donation Matching
-- Calculate number of blood types of patients which lies between age 20 to 45
SELECT Age, Blood_type, COUNT(Blood_Type) as Count_Blood_Type
FROM Healthcare
WHERE AGE BETWEEN 20 AND 45
GROUP BY 1,2
ORDER BY Blood_Type DESC, Age DESC;

-- Find how many patients are Universal Blood Donor and Universal Blood Receiver
SELECT DISTINCT 
    (SELECT COUNT(Blood_Type) FROM healthcare WHERE Blood_Type = 'O-') AS Universal_Blood_Donors, 
    (SELECT COUNT(Blood_Type) FROM healthcare WHERE Blood_Type = 'AB+') as Universal_Blood_Receivers
FROM healthcare;

-- Find the total patients of each blood group
SELECT Blood_Type, COUNT(Blood_Type) as Total_Patients 
FROM healthcare
GROUP BY Blood_Type
ORDER BY Total_Patients DESC;

-- 6. Yearly Admissions & Insurance Analysis
-- Provide a list of hospitals along with the count of patients admitted in the year 2024 AND 2025
SELECT DISTINCT Hospital, COUNT(*) as Total_Admitted
FROM healthcare
WHERE YEAR(Date_of_Admission) IN (2024, 2025)
GROUP BY 1
ORDER BY Total_Admitted DESC;

-- Find the average, minimum and maximum billing amount for each insurance provider
SELECT Insurance_Provider, 
       ROUND(AVG(Billing_Amount), 0) as Average_Amount, 
       ROUND(MIN(Billing_Amount), 0) as Minimum_Amount, 
       ROUND(MAX(Billing_Amount), 0) as Maximum_Amount
FROM healthcare
GROUP BY 1
ORDER BY Average_Amount DESC;

-- Total amount by the insurance provider
SELECT Insurance_Provider, ROUND(SUM(Billing_Amount), 2) as Total_Amount
FROM healthcare
GROUP BY Insurance_Provider
ORDER BY Total_Amount DESC;

-- 7. Patient Risk Categorization
-- Create a new column that categorizes patients as high, medium, or low risk based on their medical condition
SELECT Name, Medical_Condition, Test_Results,
CASE 
    WHEN Test_Results = 'Inconclusive' THEN 'High Risk - Need More Checks / CANNOT be Discharged'
    WHEN Test_Results = 'Normal' THEN 'Low Risk - Can take discharge, But need to follow Prescribed medications timely' 
    WHEN Test_Results = 'Abnormal' THEN 'Medium Risk - Needs more attention and more tests'
    ELSE 'Unknown Risk'
END AS 'Risk_Status', Hospital, Doctor
FROM Healthcare
ORDER BY 
    CASE 
        WHEN Test_Results = 'Inconclusive' THEN 1
        WHEN Test_Results = 'Abnormal' THEN 2
        WHEN Test_Results = 'Normal' THEN 3
        ELSE 4
    END;

-- Finding Hospitals which were successful in discharging patients after having test results as 'Normal'
SELECT Medical_Condition, Hospital, 
       DATEDIFF(Discharge_Date, Date_of_Admission) as Total_Hospitalized_days,
       Test_results
FROM Healthcare
WHERE Test_results = 'Normal'
ORDER BY Medical_Condition, Hospital;

-- 8. Advanced Analysis Queries
-- Age distribution analysis
SELECT 
    CASE 
        WHEN Age < 18 THEN 'Under 18'
        WHEN Age BETWEEN 18 AND 30 THEN '18-30'
        WHEN Age BETWEEN 31 AND 45 THEN '31-45'
        WHEN Age BETWEEN 46 AND 60 THEN '46-60'
        WHEN Age > 60 THEN 'Over 60'
    END as Age_Group,
    COUNT(*) as Patient_Count,
    ROUND(AVG(Billing_Amount), 2) as Avg_Billing
FROM healthcare
GROUP BY 
    CASE 
        WHEN Age < 18 THEN 'Under 18'
        WHEN Age BETWEEN 18 AND 30 THEN '18-30'
        WHEN Age BETWEEN 31 AND 45 THEN '31-45'
        WHEN Age BETWEEN 46 AND 60 THEN '46-60'
        WHEN Age > 60 THEN 'Over 60'
    END
ORDER BY MIN(Age);

-- Gender-based analysis
SELECT Gender, 
       COUNT(*) as Total_Patients,
       ROUND(AVG(Billing_Amount), 2) as Avg_Billing,
       ROUND(AVG(DATEDIFF(Discharge_Date, Date_of_Admission)), 1) as Avg_Hospital_Stay
FROM healthcare
GROUP BY Gender;

-- Top 10 most expensive medical conditions
SELECT Medical_Condition, 
       COUNT(*) as Patient_Count,
       ROUND(AVG(Billing_Amount), 2) as Avg_Billing,
       ROUND(MAX(Billing_Amount), 2) as Max_Billing,
       ROUND(MIN(Billing_Amount), 2) as Min_Billing
FROM healthcare
GROUP BY Medical_Condition
ORDER BY Avg_Billing DESC
LIMIT 10;

-- Hospital efficiency analysis (average stay duration vs billing)
SELECT Hospital,
       COUNT(*) as Total_Patients,
       ROUND(AVG(DATEDIFF(Discharge_Date, Date_of_Admission)), 1) as Avg_Stay_Days,
       ROUND(AVG(Billing_Amount), 2) as Avg_Billing,
       ROUND(AVG(Billing_Amount) / AVG(DATEDIFF(Discharge_Date, Date_of_Admission)), 2) as Billing_Per_Day
FROM healthcare
GROUP BY Hospital
ORDER BY Avg_Stay_Days ASC;

-- 9. Stored Procedure for Blood Matching
DELIMITER $$

CREATE PROCEDURE Blood_Matcher(IN Name_of_patient VARCHAR(200))
BEGIN 
    SELECT D.Name as Donor_name, D.Age as Donor_Age, D.Blood_Type as Donors_Blood_type, D.Hospital as Donors_Hospital, 
           R.Name as Receiver_name, R.Age as Receiver_Age, R.Blood_Type as Receivers_Blood_type, R.Hospital as Receivers_hospital
    FROM Healthcare D 
    INNER JOIN Healthcare R ON (D.Blood_type = 'O-' AND R.Blood_type = 'AB+') AND ((D.Hospital = R.Hospital) OR (D.Hospital != R.Hospital))
    WHERE (R.Name LIKE CONCAT('%', Name_of_patient, '%')) AND (D.AGE BETWEEN 20 AND 40)
    ORDER BY D.Hospital = R.Hospital DESC, D.Age;
END $$

DELIMITER ;

-- Example usage: CALL Blood_Matcher('Matthew Cruz');
