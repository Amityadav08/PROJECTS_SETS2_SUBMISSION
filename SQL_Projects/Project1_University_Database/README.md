# Project 1: University Database

## Overview

I created a comprehensive University database system using MySQL with three related tables: Students, Departments, and Courses. This project demonstrates database design, relationships, and complex SQL queries.

## What I Did

- Designed a relational database with proper foreign key relationships
- Created three tables: Departments, Students, and Courses
- Inserted sample data for testing and demonstration
- Implemented 14 complex SQL queries for data analysis
- Added additional analytical queries for comprehensive insights

## Database Structure

- **Departments Table**: DepartmentID (PK), DepartmentName
- **Students Table**: StudentID (PK), Name, Age, DepartmentID (FK)
- **Courses Table**: CourseID (PK), CourseName, StudentID (FK)

## Relationships

- A student belongs to one department (Many-to-One)
- A student can enroll in multiple courses (One-to-Many)
- A department can have multiple students (One-to-Many)

## Sample Data

- **5 Departments**: Computer Science, Mathematics, Physics, Chemistry, Biology
- **8 Students**: Various ages and department assignments
- **16 Course Enrollments**: Multiple courses per student

## Key Queries Implemented

1. Student details with department names
2. Students enrolled in specific courses
3. Department-wise student counts
4. Courses by specific students
5. Students with multiple course enrollments
6. Average age by department
7. Department with most students
8. Students not enrolled in any course
9. Student course count summary
10. Department and course filtering
11. Department-wise course distribution
12. Most popular course analysis
13. Comprehensive student-course mapping
14. Department age statistics

## How to Run

1. Open MySQL Workbench or command line
2. Run the `university_database.sql` file
3. Execute individual queries to see results

## Requirements

- MySQL Server
- MySQL Workbench (recommended)

## Expected Outputs

- Database creation and table setup
- Sample data insertion
- 14 analytical query results
- Comprehensive data insights

## Technical Skills Demonstrated

- Database design and normalization
- Foreign key relationships
- Complex JOIN operations
- Aggregate functions (COUNT, AVG, MIN, MAX)
- GROUP BY and HAVING clauses
- Subqueries and filtering
- Data analysis and reporting
