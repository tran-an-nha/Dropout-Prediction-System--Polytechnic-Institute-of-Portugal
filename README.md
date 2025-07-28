Student Dropout Prediction – Polytechnic Institute of Portugal

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