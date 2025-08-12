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
---
Step 2: EDA

2.1 
<pre>
    
python
    
df.info()
    
</pre>

2.1 Description: 

Check the data structure, number of rows, number of columns, data types, and the count of non-null value.

Provide an initial understanding of the dataset, identify numeric columns (int64, float64) and object (text) columns, and detect whether there are any missing values.

The dataset contains 4,424 rows and 37 columns, with no missing values. It includes various information related to students’ academic records and personal characteristics, including the target column Target.

2.2
<pre>
    
python
    
df.shape

df.duplicated().sum()

df.isnull().sum()*100/len(df)
    
</pre>

2.2 Description:

Verify dataset dimensions, check for duplicate rows, and assess missing value percentages.

Ensure data quality before processing by confirming the dataset size, ensuring no duplicate records exist, and identifying the proportion of missing values in each column.

2.3 
<pre>
    
python
    
#Number of unique value in each column
l1 = []
for col in df.columns:
    l1.append((col,df[col].nunique()))

nu_df = pd.DataFrame(l1, columns=['feature','n_unique'])
nu_df.sort_values(by='n_unique',ascending=True).style.background_gradient(cmap='Greens')

#Categorical columns
cat_cols = ["Marital status", "Application mode", "Course", "Daytime/evening attendance\t",
            "Previous qualification", "Nacionality", "International", "Gender",
            "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
            "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Scholarship holder"]

#Numerical columns
num_cols = list(set(df.columns) - set(cat_cols) - set(['Target']))
df[num_cols]
df[cat_cols]

df[num_cols].describe()

#Visualize numerical columns
fig, axes = plt.subplots(3, 7, figsize=(40, 20))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}', fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i+1, 21):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#Visualize categorical colums
fig, axes = plt.subplots(5, 4, figsize=(20, 20))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(f'Countplot of {col}', fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i+1, 20):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#Pie chart of target
fig, ax = plt.subplots()
ax.pie(df['Target'].value_counts(), labels=df['Target'].value_counts().index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
ax.set_title('Target Distribution')
plt.show()

</pre>

2.3 Description:

Convert numerical encoded values in the dataset into human-readable labels to make data analysis and visualization more intuitive

In many datasets, categorical variables are encoded as numbers for storage and processing efficiency. However, this numeric encoding is less interpretable for visualization or analysis. Decode mapping will:

Help readers quickly understand the meaning of each value.

Support clearer inspection, comparison, and visualization of categorical data.

Reduce the risk of misinterpretation during feature analysis.

Columns such as Marital status, Application mode, Course, Daytime/evening attendance, etc., have been decoded from numeric codes into descriptive labels.

The dataset is now more readable and ready for the next steps, such as feature correlation analysis and visualization.

2.4  
<details>
<summary>📜 Xem đoạn code mapping_dict</summary>

```python
mapping_dict = {
    "Marital status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto union",
        6: "Legally separated"
    },
    "Application mode": {
        1: "1st phase - general contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores Island)",
        7: "Holders of other higher courses",
        10: "Ordinance No. 854-B/99",
        15: "International student (bachelor)",
        16: "1st phase - special contingent (Madeira Island)",
        17: "2nd phase - general contingent",
        18: "3rd phase - general contingent",
        26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
        27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)"
    },
    ...
}
```
</details>

2.4 Description:
This code creates a mapping_dict that stores mapping tables for specific columns in the dataset.

The purpose of mapping_dict is to convert numeric codes into descriptive categorical labels, making the data more readable and easier to analyze.

2.5 
<pre>
  Python  
# Tạo bản sao của df để thực hiện thay thế giá trị
df_decode = df.copy()

# Áp dụng mapping cho từng cột dựa trên bảng ánh xạ
for col, mapping in mapping_dict.items():
    df_decode[col] = df_decode[col].map(mapping)

# Hiển thị DataFrame sau khi thay thế
df_decode.head()
    
</pre>

2.5 Description: 

This step replaces coded values in the dataset with their corresponding descriptive labels using a predefined mapping dictionary.

2.6 
<pre>
  Python  
# features grouping
demo_cols = ["Marital status", "Nacionality", "International", "Gender",
            "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
             "Displaced", "Debtor", "Age at enrollment"]
academic_cols = ["Application mode", "Application order", "Course", "Admission grade",
                 "Previous qualification", "Previous qualification (grade)",
                 "Educational special needs", "Tuition fees up to date", "Daytime/evening attendance\t", "Scholarship holder"]
socialfactor_cols = ["GDP", "Inflation rate", "Unemployment rate"]
performance_cols = [col for col in df_decode.columns if col not in demo_cols + academic_cols + socialfactor_cols + ["Target"]]

print(len(demo_cols))
print(len(academic_cols))
print(len(socialfactor_cols))
print(len(performance_cols))
    
</pre>

2.6 Description: 

In this step, the dataset’s columns are categorized into different feature groups for further analysis or model training.

Details of each group:

demo_cols – Demographic features, including columns related to marital status, nationality, gender, parents’ education and occupation, displacement status, debtor status, and age at enrollment.

academic_cols – Academic features, including columns related to application details, course, admission grades, previous qualifications, special educational needs, tuition fee status, study schedule (daytime/evening), and scholarship status.

socialfactor_cols – Social factor features, including macroeconomic indicators such as GDP, inflation rate, and unemployment rate.

performance_cols – Performance features, containing all remaining columns except those included in the above three groups and the Target column.

2.7
<pre>
Python 
# Draw stacked column charts with hue = Target for cat_cols in demo_cols
for col in demo_cols:
      data = df_decode.groupby([col, 'Target']).size().unstack()
      if col in cat_cols:
        data = data.sort_values(by=data.columns[1], ascending=False)
      fig, ax = plt.subplots(figsize=(8, 5))
      data.plot(kind='bar', stacked=True, ax=ax, color=['#ae3130', '#3b7cb5','#53c13e'])
      ax.set_xlabel(col)
      ax.set_ylabel('Count')
      ax.set_title(f'Stacked Column Chart of {col} by Target')
      plt.show()
</pre>

2.7 Description:

To visualize the relationship between each categorical demographic feature (demo_cols) and the target variable (Target) in order to identify patterns or trends.

Process:

Group and Count:

The dataset is grouped by each column in demo_cols along with the Target column.

The size of each group is counted and transformed into a table format using .unstack().

Sorting:

If the feature is in the categorical columns (cat_cols), the values are sorted by the counts of the second target category in descending order.

Visualization:

For each demographic feature, a stacked bar chart is created, showing the distribution of the target classes for each category in the feature.

Custom colors (#ae3130, #3b7cb5, #53c13e) are used for better visual distinction between target classes.

2.8
<pre>
Python 
# Draw 100% stacked column charts with hue = Target for cat_cols in demo_cols
for col in demo_cols:
      data = df_decode.groupby([col, 'Target']).size().unstack()
      data_percent = data.div(data.sum(axis=1), axis=0) * 100
      if col in cat_cols:
        data_percent = data_percent.sort_values(by=data.columns[1], ascending=False)
      fig, ax = plt.subplots(figsize=(8, 5))
      data_percent.plot(kind='bar', stacked=True, ax=ax, color=['#ae3130', '#3b7cb5','#53c13e'])
      ax.set_xlabel(col)
      ax.set_ylabel('Count')
      ax.set_title(f'100% Stacked Column Chart of {col} by Target')
      plt.show()
</pre>

2.8 Description:

To visualize the proportional distribution of target classes within each category of demographic features, making it easier to compare relative percentages instead of raw counts.

Process:

Group and Count:

The dataset is grouped by each column in demo_cols along with the Target column.

The counts for each category-target combination are calculated and reshaped with .unstack().

Convert to Percentage:

Each value is divided by the row total (.sum(axis=1)) to obtain the percentage distribution for that category.

The result is multiplied by 100 to convert to percentage format.

Sorting:

If the feature is in the categorical columns (cat_cols), the categories are sorted by the counts of the second target category in descending order.

Visualization:

For each demographic feature, a 100% stacked bar chart is created to show the percentage breakdown of target classes for each category.

Custom colors (#ae3130, #3b7cb5, #53c13e) are used to differentiate target classes.

2.9
<pre>
Python 
data_age = df_decode.groupby(['Age at enrollment', 'Target']).size().unstack()
data_age_percent = data_age.div(data_age.sum(axis=1), axis=0) * 100
data_age_percent = data_age_percent.sort_values(by=data_age_percent.columns[1], ascending=False)

# Sort data_age_percent by age
data_age_percent = data_age_percent.sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
data_age_percent.plot(kind='bar', stacked=True, ax=ax, color=['#ae3130', '#3b7cb5','#53c13e'])

ax2 = ax.twinx()
age_counts = df_decode['Age at enrollment'].value_counts()
age_counts = age_counts.reindex(data_age_percent.index)
sns.lineplot(x=range(len(age_counts)), y=age_counts.values, color='black', ax=ax2)

ax.set_xlabel('Age at enrollment')
ax.set_ylabel('Percent')
ax2.set_ylabel('Number of data points')
ax.set_title('100% Stacked Column Chart of Age at enrollment by Target')

ax2.set_xticks(range(len(data_age_percent.index)))
ax2.set_xticklabels(data_age_percent.index)

plt.show()
</pre>

2.9 Description:

To visualize the proportional distribution of target classes by Age at enrollment while also overlaying the actual count of data points for each age group. This dual-axis chart helps

identify both the relative proportion and the absolute sample size for each age category.

Process:

Group and Count:

Group the dataset by Age at enrollment and Target, count the number of records in each group, and reshape with .unstack().

Convert to Percentage:

Divide the count in each category by the total for that age (.sum(axis=1)) and multiply by 100 to get the percentage distribution.

Sort by the second target category for a consistent visual order.

Sort by Age:

Reorder the DataFrame index so that ages are plotted in ascending order.

Create Main Plot (Primary Y-axis):

Draw a 100% stacked bar chart showing the proportional distribution of target classes for each age.

Use custom colors (#ae3130, #3b7cb5, #53c13e).

Overlay Data Point Counts (Secondary Y-axis):

Create a secondary Y-axis (twinx) to plot the total count of data points for each age using a line chart.

This line shows how many samples exist for each age, giving additional context to the percentage data.

Labeling and Styling:

Primary Y-axis shows percentages; secondary Y-axis shows number of data points.

X-axis labels are the sorted ages.

Title clearly states that the chart shows the proportional distribution by age with counts overlayed.

2.10
<pre>
Python 
# Draw stacked column charts with hue = Target for cat_cols in academic_cols
for col in academic_cols:
      data = df_decode.groupby([col, 'Target']).size().unstack()
      if col in cat_cols:
        data = data.sort_values(by=data.columns[1], ascending=False)
      fig, ax = plt.subplots(figsize=(8, 5))
      data.plot(kind='bar', stacked=True, ax=ax, color=['#ae3130', '#3b7cb5','#53c13e'])
      ax.set_xlabel(col)
      ax.set_ylabel('Count')
      ax.set_title(f'Stacked Column Chart of {col} by Target')
      plt.show()
</pre>


2.10 Description:

To visualize the distribution of Target categories across multiple academic-related columns. This step helps identify patterns or correlations between academic attributes and target outcomes.

Process:

Iterate through Academic Columns:

Loop through each column in academic_cols.

For each column, group the data by [column, 'Target'], count the occurrences, and reshape with .unstack() to separate target classes into columns.

Sort for Categorical Columns:

If the current column is also in cat_cols (categorical columns list), sort the data by the second target column in descending order.

Plot Stacked Column Chart:

Use stacked bar charts to show the distribution of target classes for each category in the academic column.

Assign custom colors:

#ae3130 (red)

#3b7cb5 (blue)

#53c13e (green)

Label and Customize:

X-axis is labeled with the column name.

Y-axis shows the count of records.

Title dynamically changes based on the column being plotted: "Stacked Column Chart of {col} by Target".

2.11
<pre>
Python 
# Draw stacked column charts with hue = Target for cat_cols in academic_cols
for col in academic_cols:
      data = df_decode.groupby([col, 'Target']).size().unstack()
      data_percent = data.div(data.sum(axis=1), axis=0) * 100
      if col in cat_cols:
        data_percent = data_percent.sort_values(by=data.columns[1], ascending=False)
      fig, ax = plt.subplots(figsize=(8, 5))
      data_percent.plot(kind='bar', stacked=True, ax=ax, color=['#ae3130', '#3b7cb5','#53c13e'])
      ax.set_xlabel(col)
      ax.set_ylabel('Count')
      ax.set_title(f'100% Stacked Column Chart of {col} by Target')
      plt.show()
</pre>

2.11 Description:

To create 100% stacked column charts showing the proportional distribution of Target categories for each academic-related column. This step highlights how target classes are distributed 

as percentages, making it easier to compare between categories regardless of sample size.

Process:

Iterate through Academic Columns:

Loop through each column in academic_cols.

Group data by [column, 'Target'], count occurrences, and reshape with .unstack() to separate Target categories.

Convert to Percentage:

Divide each count by the total count for that category (.sum(axis=1)) and multiply by 100 to get percentages.

Sort for Categorical Columns:

If the column is in cat_cols (categorical list), sort the data based on the second target category column in descending order for clearer visualization.

2.12 
<pre>
Python 
fig, ax = plt.subplots()
sns.scatterplot(data=df_decode, x='Previous qualification (grade)', y='Admission grade', hue='Target', ax=ax, palette={"Dropout":'#ae3130', "Enrolled":'#3b7cb5',"Graduate":'#53c13e'})
</pre>

2.12 Description:

To visualize the relationship between students’ previous qualification grades and admission grades, and how these relate to the Target outcome (Dropout, Enrolled, Graduate).

Process:

Create Figure & Axes:

Use plt.subplots() to create a figure and axis for plotting.

Scatter Plot with Seaborn:

x axis: 'Previous qualification (grade)'

y axis: 'Admission grade'

hue: 'Target' – to differentiate outcomes using colors.

Color mapping (palette):

Dropout → #ae3130 (red)

Enrolled → #3b7cb5 (blue)

Graduate → #53c13e (green)

Purpose of Visualization:

Identify trends or clusters indicating how prior academic performance influences admission grades.

Explore whether certain target outcomes are more common at specific grade ranges.

2.13
<pre>
Python 
#Draw stacked column charts with hue = Target for cat_cols in socialfactor_cols
for col in socialfactor_cols:
      data = df_decode.groupby([col, 'Target']).size().unstack()
      if col in cat_cols:
        data = data.sort_values(by=data.columns[1], ascending=False)
      fig, ax = plt.subplots(figsize=(8, 5))
      data.plot(kind='bar', stacked=True, ax=ax, color=['#ae3130', '#3b7cb5','#53c13e'])
      ax.set_xlabel(col)
      ax.set_ylabel('Count')
      ax.set_title(f'Stacked Column Chart of {col} by Target')
      plt.show()
</pre>

2.13 Description:

Visualize the distribution of social factor variables (socialfactor_cols) by academic outcome (Target) to identify potential relationships between social factors and the likelihood of

dropping out, staying enrolled, or graduating.

Process:

Iterate through each column in socialfactor_cols:

Group the dataset (groupby) by the current column and Target.

Count the occurrences (size()) and reshape the data with unstack() to create a stacked column format.

If the column is in cat_cols (categorical variables):

Sort the values by the second column (data.columns[1]) in descending order.

Plot the chart:

Figure size: (8, 5).

Type: bar with stacked=True for stacked column visualization.

Colors:

Dropout → #ae3130 (red)

Enrolled → #3b7cb5 (blue)

Graduate → #53c13e (green)

Add labels and title:

X-axis: Name of the current column.

Y-axis: 'Count'.

Title: "Stacked Column Chart of {col} by Target".

2.14





