
Cardiac Prediction using Machine Learning Models This project aims to predict cardiac health metrics such as BMI, obesity, and cholesterol levels using various machine learning models. The project compares the performance of different models, including Support Vector Machines (SVM), Decision Trees, Linear Regression, and Artificial Neural Networks (ANN).

### Table of Contents
1. Project Overview
2. Problem Statement
3. Dataset
4. Prerequisites
5. Installation
6. Usage
7. Models and Evaluation
8. Results
9. Visualization.
10. Conclusion


### 1. Project Overview
This project focuses on predicting the likelihood of cardiac arrest in individuals by analyzing health-related data using various machine learning models. By leveraging models such as **Logistic Regression**, **Decision Trees**, **Random Forest**, **Support Vector Machines (SVM)**, and **Artificial Neural Networks (ANN)**, the goal is to provide early warning signs of cardiac arrest for timely intervention.


### 2. Problem Statement
Cardiac arrest is a sudden cessation of heart function, often leading to death if untreated. Predicting the likelihood of cardiac arrest can enable early medical intervention, potentially saving lives. This project uses patient data to predict the onset of cardiac arrest and identify high-risk individuals.



### 3. Dataset
The dataset used in this project contains various medical and health-related parameters such as BMI, cholesterol, obesity, blood pressure (BP), physical activity, and age. 

The dataset is preprocessed to ensure model readiness:
- **Handling Missing Values**: Using techniques like mean imputation or removing incomplete records.
- **Normalization/Scaling**: Ensuring that all features are on the same scale.
- **Encoding Categorical Variables**: Converting non-numeric data into numerical form using techniques like one-hot encoding.
- **Feature Selection**: Identifying features that have a strong correlation with cardiac arrest risk.


### 4. Prerequisites
Ensure the following tools and libraries are installed:
- **Python 3.x**
- **Jupyter Notebook** or **Google Colab**

The required Python libraries include:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`


### 5. Installation
Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/cardiac-arrest-prediction.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd cardiac-arrest-prediction
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


### 6. Usage
To run the project:

1. **Load the dataset**:
   - Ensure the dataset contains columns like **BP**, **BMI**, **AGE**, **OBESITY**, **PHYSICAL_ACTIVITY**, **CHOLESTEROL**, and others as required.

2. **Run the script**:
   - Execute the script using **Jupyter Notebook** or **Google Colab**:
   ```bash
   jupyter notebook
   ```
   - Or upload the `.ipynb` file and dataset into Google Colab.


### 7. Models and Evaluation
The following machine learning models are implemented to predict the likelihood of cardiac arrest:

- **Linear Regression**: Predicts cardiac arrest risk using a linear relationship between features and the target, simple and interpretable but assumes linearity.
- **Support Vector Machine (SVM)**: Classifies patients into high/low risk categories, effective in high-dimensional spaces but computationally expensive for large datasets.
- **Artificial Neural Networks (ANN)**: Captures complex nonlinear relationships in data, flexible and powerful but requires large datasets and is prone to overfitting.
- **Decision Tree**: Splits data into decision nodes to predict cardiac arrest risk, easy to interpret but prone to overfitting with deep trees.

Each model is evaluated using appropriate metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.


### 8. Results
The results for each model are compared based on the following metrics:

-**BMI Prediction Accuracy:**
  - SVM: 99.57%
  - Decision Tree: 99.86%
  - Linear Regression: 100.0%
  - ANN: 99.80%

- **Obesity Prediction Accuracy:**
  - SVM: 98.98%
  - Decision Tree: 100.0% 
  - Linear Regression: 100.0%
  - ANN: 99.20%

- **Cholesterol Prediction Accuracy:**
   - SVM: 69.37%
   - Decision Tree: 99.87%
   - Linear Regression: 100.0%
   - ANN: 99.81%


### 9. Visualization
The project includes visualizations to compare the model performance for predicting the risk of cardiac arrest.
These visualizations help in understanding which model performs the best for each target variable.


### 10. Conclusion
This project demonstrates how machine learning can be applied to predict the risk of cardiac arrest, potentially saving lives by enabling timely medical interventions. Various models have been implemented and compared to determine which performs best in predicting cardiac arrest based on patient health data.


This README provides an overview of the **"Cardiac Arrest Prediction Using Machine Learning Models"** project, including setup instructions, model comparisons,summary results.
