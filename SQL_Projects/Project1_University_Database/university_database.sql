-- Project 1: University Database
-- Student: Amit Yadav
-- Course: Data Analysis and Data Science
-- Batch: 122

-- Create database
CREATE DATABASE IF NOT EXISTS UniversityDB;
USE UniversityDB;

-- Create Departments table
CREATE TABLE Departments (
    DepartmentID INT PRIMARY KEY AUTO_INCREMENT,
    DepartmentName VARCHAR(100) NOT NULL
);

-- Create Students table
CREATE TABLE Students (
    StudentID INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100) NOT NULL,
    Age INT NOT NULL,
    DepartmentID INT,
    FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
);

-- Create Courses table
CREATE TABLE Courses (
    CourseID INT PRIMARY KEY AUTO_INCREMENT,
    CourseName VARCHAR(100) NOT NULL,
    StudentID INT,
    FOREIGN KEY (StudentID) REFERENCES Students(StudentID)
);

-- Insert sample data into Departments
INSERT INTO Departments (DepartmentName) VALUES
('Computer Science'),
('Mathematics'),
('Physics'),
('Chemistry'),
('Biology');

-- Insert sample data into Students
INSERT INTO Students (Name, Age, DepartmentID) VALUES
('Alice Johnson', 20, 1),
('Bob Smith', 22, 1),
('Carol Davis', 19, 2),
('David Wilson', 21, 3),
('Emma Brown', 23, 4),
('Frank Miller', 20, 5),
('Grace Lee', 22, 1),
('Henry Taylor', 19, 2);

-- Insert sample data into Courses
INSERT INTO Courses (CourseName, StudentID) VALUES
('Data Structures', 1),
('Artificial Intelligence', 1),
('Machine Learning', 1),
('Database Systems', 2),
('Software Engineering', 2),
('Calculus I', 3),
('Linear Algebra', 3),
('Quantum Mechanics', 4),
('Thermodynamics', 4),
('Organic Chemistry', 5),
('Analytical Chemistry', 5),
('Cell Biology', 6),
('Genetics', 6),
('Advanced Programming', 7),
('Computer Networks', 7),
('Differential Equations', 8);

-- Query 1: Retrieve all student details along with their department names
SELECT 
    s.StudentID,
    s.Name,
    s.Age,
    d.DepartmentName
FROM Students s
LEFT JOIN Departments d ON s.DepartmentID = d.DepartmentID;

-- Query 2: Find the names of all students who are enrolled in 'Artificial Intelligence'
SELECT DISTINCT s.Name
FROM Students s
JOIN Courses c ON s.StudentID = c.StudentID
WHERE c.CourseName = 'Artificial Intelligence';

-- Query 3: Count how many students are in each department
SELECT 
    d.DepartmentName,
    COUNT(s.StudentID) as StudentCount
FROM Departments d
LEFT JOIN Students s ON d.DepartmentID = s.DepartmentID
GROUP BY d.DepartmentID, d.DepartmentName
ORDER BY StudentCount DESC;

-- Query 4: List the courses taken by 'Alice Johnson'
SELECT c.CourseName
FROM Students s
JOIN Courses c ON s.StudentID = c.StudentID
WHERE s.Name = 'Alice Johnson';

-- Query 5: Find students who are enrolled in more than one course
SELECT s.Name, COUNT(c.CourseID) as CourseCount
FROM Students s
JOIN Courses c ON s.StudentID = c.StudentID
GROUP BY s.StudentID, s.Name
HAVING COUNT(c.CourseID) > 1
ORDER BY CourseCount DESC;

-- Query 6: Get the average age of students in each department
SELECT 
    d.DepartmentName,
    ROUND(AVG(s.Age), 2) as AverageAge
FROM Departments d
LEFT JOIN Students s ON d.DepartmentID = s.DepartmentID
GROUP BY d.DepartmentID, d.DepartmentName
ORDER BY AverageAge DESC;

-- Query 7: Find the department with the most students
SELECT 
    d.DepartmentName,
    COUNT(s.StudentID) as StudentCount
FROM Departments d
LEFT JOIN Students s ON d.DepartmentID = s.DepartmentID
GROUP BY d.DepartmentID, d.DepartmentName
ORDER BY StudentCount DESC
LIMIT 1;

-- Query 8: List all students who are NOT enrolled in any course
SELECT s.Name, s.Age, d.DepartmentName
FROM Students s
LEFT JOIN Courses c ON s.StudentID = c.StudentID
LEFT JOIN Departments d ON s.DepartmentID = d.DepartmentID
WHERE c.StudentID IS NULL;

-- Query 9: Retrieve students along with the total number of courses they are enrolled in
SELECT 
    s.Name,
    d.DepartmentName,
    COUNT(c.CourseID) as TotalCourses
FROM Students s
LEFT JOIN Departments d ON s.DepartmentID = d.DepartmentID
LEFT JOIN Courses c ON s.StudentID = c.StudentID
GROUP BY s.StudentID, s.Name, d.DepartmentName
ORDER BY TotalCourses DESC, s.Name;

-- Query 10: Find students who belong to 'Computer Science' and are taking a course with 'Data' in its name
SELECT DISTINCT s.Name, c.CourseName
FROM Students s
JOIN Departments d ON s.DepartmentID = d.DepartmentID
JOIN Courses c ON s.StudentID = c.StudentID
WHERE d.DepartmentName = 'Computer Science'
AND c.CourseName LIKE '%Data%';

-- Additional Analysis Queries

-- Query 11: Show department-wise course distribution
SELECT 
    d.DepartmentName,
    COUNT(DISTINCT c.CourseName) as UniqueCourses,
    COUNT(c.CourseID) as TotalEnrollments
FROM Departments d
LEFT JOIN Students s ON d.DepartmentID = s.DepartmentID
LEFT JOIN Courses c ON s.StudentID = c.StudentID
GROUP BY d.DepartmentID, d.DepartmentName
ORDER BY TotalEnrollments DESC;

-- Query 12: Find the most popular course
SELECT 
    c.CourseName,
    COUNT(c.StudentID) as EnrollmentCount
FROM Courses c
GROUP BY c.CourseName
ORDER BY EnrollmentCount DESC
LIMIT 1;

-- Query 13: Show students with their course details
SELECT 
    s.Name,
    s.Age,
    d.DepartmentName,
    GROUP_CONCAT(c.CourseName ORDER BY c.CourseName SEPARATOR ', ') as Courses
FROM Students s
LEFT JOIN Departments d ON s.DepartmentID = d.DepartmentID
LEFT JOIN Courses c ON s.StudentID = c.StudentID
GROUP BY s.StudentID, s.Name, s.Age, d.DepartmentName
ORDER BY s.Name;

-- Query 14: Department age statistics
SELECT 
    d.DepartmentName,
    COUNT(s.StudentID) as StudentCount,
    MIN(s.Age) as MinAge,
    MAX(s.Age) as MaxAge,
    ROUND(AVG(s.Age), 2) as AvgAge
FROM Departments d
LEFT JOIN Students s ON d.DepartmentID = s.DepartmentID
GROUP BY d.DepartmentID, d.DepartmentName
ORDER BY AvgAge DESC;
