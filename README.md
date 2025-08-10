Project Title: Predicting Student Dropout Higher Education
---
Author: Tran Thi Ngoc Quyen


Date: 7-2024


Tools Used: Python


***


📑 Table of Contents


📌 Background & Overview


📂 Dataset Description & Data Structure


🧠 Design Thinking Process


📊 Key Insights & Visualizations


🔎 Final Conclusion & Recommendations

***

📌 Background & Overview

---

Objective:

Business Question: Can we accurately predict which students are at risk of dropping out using their academic and socio-economic data?

Objective: Build a Machine Learning model that predicts the likelihood of student dropout at the Polytechnic Institute of Portalegre.

Problem Statement: The institution is facing increasing dropout rates, making it difficult to support students proactively and manage resources effectively.

Why it matters:


High dropout rates lead to financial loss and resource waste.


Early prediction allows targeted support and intervention.


Improves student retention, experience, and institutional performance.

Approach: Use real student data from enrollment to academic performance, applying supervised ML models to detect high-risk students.

👤 Who is this project for?

School administrators & academic managers: To support data-driven decisions for early intervention and design tailored policies for students at risk of dropping out.

Educational data analytics teams: To provide a predictive and visual tool that monitors dropout risks in real-time.

Academic advisors and student support staff: To help prioritize personalized mentoring and guidance for vulnerable students.

Developers & Machine Learning practitioners: As a practical case study applying ML to the education sector with meaningful social impact.

***

📂 Dataset Description & Data Structure
---

📌 Data Source

Source: Kaggle

Size: 37 columns & 4424 rows

Format: .csv

📊 Data Structure & Relationships

The dataset consists of 74 columns and one target variable, structured to represent the student profile at enrollment time and their academic performance over two semesters, aiming to predict final outcomes: Dropout, Enrolled, or Graduate

📁 Data Tables

The dataset is provided as a single flat table (CSV), with each row representing one student record.

| Column Group                  | Description                                                                                  |
|------------------------------|----------------------------------------------------------------------------------------------|
| Academic Information       | Study program, course type (day/evening), admission grade, entrance exam score, enrolled/passed/approved subjects, GPA in semesters 1 & 2           |
| Personal Information       | Gender, age at enrollment, nationality, marital status, international student status, special needs, displaced student      |
| Family Background         | Father’s education and occupation, mother’s education and occupation                       |
| Financial & Socioeconomic  | Scholarship status, tuition payment status, debt status                                     |
| Application Info           | Application mode, application order                                                         |
| Macroeconomic Context      | Unemployment rate, inflation rate, GDP at enrollment year                                   |
| Target Variable            | `Target`: 1 = Dropout, 2 = Enrolled, 3 = Graduate                                           |

🔗 Key Relationships Between Fields

Although the dataset is denormalized (flattened), several logical relationships can be derived from the fields:

Academic Success is influenced by:

Prior education background (e.g., high school grades, type of high school).

First-year academic performance – including early grades and course approvals.

Parental education & occupation, which often shape student expectations and study support at home.

Dropout Likelihood is often correlated with: Socioeconomic factors: scholarship status, delayed tuition payments, and unemployment levels.

Debtor status and special needs, which may signal financial or learning difficulties.

External conditions such as national unemployment rate.

Application Mode & Order reflect:

Motivation and competitiveness of students based on when and how they applied.

Whether the student was admitted via regular vs. special channels (e.g., quota systems, late admission).

Data Imbalance

The target variable is highly imbalanced: Most students belong to the Enrolled category

This needs to be handled during model training (e.g., through SMOTE, weighted loss, etc.)

***

🧠 Design Thinking Process
---

Empathize

This project approaches the problem from the perspective of decision-makers in the school (e.g., board of directors, academic affairs office), aiming to support the institution in early identification of students at risk of dropping out.

Rather than waiting for students to proactively raise concerns — which often happens too late — the school can take timely action based on data analysis.

Define

A high student dropout rate negatively impacts the institution’s reputation and training quality.

There is a lack of early warning systems to implement timely support.

Available student data needs to be analyzed to identify key influencing factors.

Ideate

Analyze historical student data to identify dropout-related factors.

Build a predictive model to estimate dropout likelihood.

Generate policy suggestions to support students based on different risk groups.

Prototype

Clean and preprocess student data.

Perform exploratory data analysis and data visualization.

Develop and compare predictive models: Decision Tree, Logistic Regression, and Random Forest.

Select the best-performing model based on evaluation metrics.

Design an interactive Power BI dashboard to present insights and recommendations.

Prototype
Clean and preprocess student data.

Perform exploratory data analysis and data visualization.

Develop and compare predictive models: Decision Tree, Logistic Regression, and Random Forest.

Select the best-performing model based on evaluation metrics.

Present insights and recommendations in a structured PowerPoint report

Test

Evaluate the model using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Test the model outputs and dashboard with the end users (academic affairs office).

Collect feedback and refine the dashboard or model accordingly.

***

⚒️ Main Process
---

1️⃣ Load Data

Step 1: Import Libraries

<pre>python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
</pre>

Step 1 Description
This step initializes the environment by importing all the required Python libraries for data manipulation, visualization, and analysis. It then loads the dataset directly from Google Drive into a Pandas DataFrame for further processing.

Key actions:
Import essential libraries: 

pandas for data manipulation and analysis.

numpy for numerical computations.

matplotlib.pyplot and seaborn for data visualization.

PCA from sklearn.decomposition for dimensionality reduction.

Load dataset from Google Drive:

Convert a shared Google Drive link into a direct download link and use pd.read_csv() to read the CSV file into a DataFrame.

Display settings:

Configure Pandas to display all columns and preview the first 5 rows to verify data loading.



Read CSV file from Google Drive
df = pd.read_csv(
    'https://drive.google.com/uc?export=download&id=' + link_data.split('/')[-2]
)

# Display all columns in the DataFrame
pd.set_option('display.max_columns', None)

# Preview the first 5 rows
df.head()





I.  Project Overview
This project applies machine learning techniques to predict student dropout at the Polytechnic Institute of Portugal. The objective is to identify early warning signs from student profiles and academic records that may indicate a risk of dropping out. By doing so, the institution can proactively intervene and improve student retention rates.

II. Problem Statement
Student dropout is a significant issue in higher education. It results in financial losses, lower institutional rankings, and missed opportunities for students. This project answers the key question:
“Can we predict which students are at risk of dropping out based on their demographics and academic performance?”

III. Objectives
- Build a classification model to predict whether a student will drop out.
- Identify the most important features influencing student dropout.
- Provide actionable insights for early intervention strategies.

IV. Dataset
- **Source**: Publicly available dataset from the Polytechnic Institute of Portugal.
- **Size**: 4424 records and 33 attributes.
- **Key Features**:
  - Demographics: Age, Gender, Scholarship, Marital Status
  - Academic Info: Admission Grade, Curricular Units Enrolled/Evaluated/Approved
  - Socioeconomic: Tuition Paid, Debts, Unemployment Rate, GDP
  - Target: `Target` (Dropout = 1, Graduate = 2, Enrolled = 3)

V. Methodology

1. **Exploratory Data Analysis (EDA)**  
   - Handle missing data, outliers, and perform feature transformation.
   - Visualizations to detect imbalance and patterns.

2. **Data Preprocessing**  
   - Label Encoding for categorical variables.
   - Feature Scaling (MinMax or StandardScaler).

3. **Modeling**  
   - Algorithms tried: Decision Tree, Random Forest, XGBoost, LightGBM
   - Evaluation Metrics: Accuracy, F1-score, Confusion Matrix

4. **Model Interpretation**  
   - SHAP values used to explain model decisions.
   - Feature importance visualized to guide interventions.

VI. Results

- **Best Model**: LightGBM
- **Accuracy**: ~83%
- **Top Influential Features**:
  - `Curricular units failed`
  - `Tuition paid`
  - `Scholarship holder`
  - `Age at enrollment`

VII. Insights & Recommendations

- Students who fail early in the program are more likely to drop out.
- Late tuition payment and non-scholarship holders have higher dropout risks.
- Early monitoring of academic performance and financial support are crucial.

VIII.Future Work

- Deploy model via web app using Streamlit or Flask.
- Integrate with student management systems (CRM/ERP).
- Implement real-time risk scoring for incoming students.

IX. Authors

- Team: DA62_3T2_G1
- Course: Data Analytics Final Project

X. Files

- `DA62_3T2_G1_FinalProject.ipynb`: Jupyter Notebook with full code.
- `README.md`: Project documentation (this file).

This project is for educational purposes only.
