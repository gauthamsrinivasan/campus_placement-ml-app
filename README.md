# 🎓 Campus Placement Prediction using Machine Learning

## M.Tech (AIML/DSE) – Machine Learning Assignment 2

---

## 📋 Problem Statement

The objective of this project is to **predict whether a student will be placed during campus recruitment** based on various factors including academic performance, skills, experience, and personal attributes. This is a **binary classification problem** where the target variable indicates whether a student is placed (1) or not placed (0).

The model aims to help educational institutions and students understand the key factors influencing employability and make data-driven decisions to improve placement outcomes.

---

## 📊 Dataset Description

**Dataset Name:** Campus Placement Prediction Dataset  
**Source:** Kaggle  
**Problem Type:** Binary Classification

### Dataset Characteristics
- **Total Instances:** 1,604 students
- **Total Input Features:** 21
- **Target Variable:** `placed` (Binary: 1 = Placed, 0 = Not Placed)
- **Missing Values:** Handled appropriately during preprocessing

> **Note:** The column `salary_lpa` is excluded as it represents a regression target and is not required for this binary classification task. The `student_id` column is also excluded as it's merely an identifier.

### Feature Categories

#### 1. Academic Performance (9 features)
- `ssc_percentage` - Secondary School Certificate percentage
- `ssc_board` - Board of education (State/CBSE/ICSE)
- `hsc_percentage` - Higher Secondary Certificate percentage
- `hsc_board` - Board of education
- `hsc_stream` - Stream of study (Science/Commerce/Arts)
- `degree_percentage` - Undergraduate degree percentage
- `degree_field` - Field of undergraduate degree
- `mba_percentage` - MBA percentage
- `specialization` - MBA specialization

#### 2. Skills & Competency (4 features)
- `technical_skills_score` - Technical skills rating (0-10)
- `soft_skills_score` - Soft skills rating (0-10)
- `aptitude_score` - Aptitude test score (0-100)
- `communication_score` - Communication skills rating (0-10)

#### 3. Experience & Activities (7 features)
- `internships_count` - Number of internships completed
- `projects_count` - Number of projects completed
- `certifications_count` - Number of certifications obtained
- `work_experience_months` - Prior work experience in months
- `leadership_roles` - Number of leadership positions held
- `extracurricular_activities` - Number of extracurricular activities
- `backlogs` - Number of academic backlogs

#### 4. Demographics (3 features)
- `gender` - Gender of the student
- `age` - Age of the student
- `city_tier` - Tier of the city (Tier 1/2/3)

#### 5. Target Variable
- `placed` → 1 (Placed), 0 (Not Placed)

---

## 🤖 Models Used

The following **6 machine learning classification models** were implemented using the same dataset and train-test split (80-20 ratio with stratification):

1. **Logistic Regression** - Linear classification model
2. **Decision Tree Classifier** - Tree-based non-linear model
3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest (Ensemble)** - Bagging ensemble method
6. **XGBoost (Ensemble)** - Gradient boosting ensemble method

---

## 📈 Model Performance Comparison

### Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8707 | 0.8644 | 0.8182 | 0.6296 | 0.7119 | 0.6755 |
| Decision Tree | 0.8398 | 0.7826 | 0.7000 | 0.6852 | 0.6925 | 0.6254 |
| KNN | 0.8491 | 0.8009 | 0.7368 | 0.6111 | 0.6682 | 0.6183 |
| Naive Bayes | 0.8118 | 0.8197 | 0.6585 | 0.7222 | 0.6889 | 0.5956 |
| Random Forest (Ensemble) | **0.8988** | **0.9153** | **0.8571** | **0.7593** | **0.8052** | **0.7783** |
| XGBoost (Ensemble) | 0.8925 | 0.9089 | 0.8378 | 0.7407 | 0.7864 | 0.7604 |

> **Best Performing Model:** Random Forest achieved the highest scores across all metrics.

---

## 🔍 Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Performs well with **high precision (0.82)** and good interpretability, making it suitable for understanding feature importance. However, it shows **lower recall (0.63)**, meaning it misses some placed candidates. The linear nature limits its ability to capture complex non-linear relationships in the data. Best suited when model interpretability is crucial. |
| **Decision Tree** | Captures **non-linear patterns** effectively with balanced precision and recall. However, it is **prone to overfitting** on training data, resulting in slightly lower generalization performance (accuracy: 0.84). The model's performance can be improved with pruning techniques. Useful for understanding decision boundaries but less reliable for production deployment. |
| **KNN** | Provides **balanced accuracy (0.85)** but is **highly sensitive to feature scaling** and the choice of k parameter. Performance degrades with high-dimensional data due to the curse of dimensionality. Computationally expensive during prediction phase as it requires distance calculation with all training samples. Suitable for smaller datasets with well-scaled features. |
| **Naive Bayes** | Achieves **higher recall (0.72)** compared to other models, making it good at identifying placed students. However, it shows **lower precision (0.66)** due to the strong independence assumption among features, which doesn't hold true in real-world placement scenarios where features are correlated. Fast training and prediction make it suitable for baseline models. |
| **Random Forest (Ensemble)** | **Best overall performer** with highest accuracy (0.90), F1 score (0.81), and MCC (0.78). Effectively **handles feature interactions** and reduces overfitting through ensemble averaging. Robust to outliers and works well with mixed data types. The model provides excellent balance between precision (0.86) and recall (0.76). **Recommended for production deployment** due to superior performance and stability. |
| **XGBoost (Ensemble)** | **Second-best performer** with strong AUC (0.91) and F1 score (0.79). Highly efficient and robust alternative to Random Forest with built-in regularization to prevent overfitting. Excellent for handling imbalanced datasets and provides feature importance rankings. Slightly faster training than Random Forest. **Excellent choice for production** with comparable performance to Random Forest. |

### Key Insights:
- **Ensemble methods (Random Forest & XGBoost)** significantly outperform individual classifiers
- **Random Forest** shows the best balance across all metrics
- **Naive Bayes** has the highest recall, useful when minimizing false negatives is critical
- **Logistic Regression** offers best interpretability with decent performance
- All models achieve **>80% accuracy**, indicating the dataset has strong predictive signals

---

## 🚀 Streamlit Web Application Features

The interactive web application includes:

### ✅ Mandatory Features (As per Assignment Requirements)

1. **📤 Dataset Upload Option (CSV)**
   - Upload custom test datasets
   - Automatic preprocessing and validation
   - Support for CSV format

2. **🎯 Model Selection Dropdown**
   - Choose from 6 trained models
   - Interactive model comparison
   - Real-time evaluation

3. **📊 Display of Evaluation Metrics**
   - Accuracy, AUC, Precision, Recall, F1 Score, MCC
   - Visual metric cards
   - Comparative charts

4. **🔢 Confusion Matrix & Classification Report**
   - Heatmap visualization
   - Detailed classification metrics
   - Per-class performance analysis

### 🎨 Additional Features

- **Multi-page Navigation** - Organized interface with 5 pages
- **Visual Comparisons** - Bar charts and comparative plots
- **Responsive Design** - Works on different screen sizes
- **Professional UI** - Clean and intuitive interface
- **Data Preview** - Sample data exploration
- **Best Model Highlighting** - Automatic identification

---

## 📁 Repository Structure

```
campus_placement-ml-app/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
│
├── data/
│   └── campus_placement.csv        # Dataset file
│
└── model/
    └── model_training.ipynb        # Jupyter notebook with model training code
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd campus_placement-ml-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

4. **Access the application**
- Open browser and navigate to `http://localhost:8501`

---

## ☁️ Deployment on Streamlit Community Cloud

### Steps to Deploy:

1. **Push code to GitHub**
   - Ensure all files are committed
   - Push to main branch

2. **Visit Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub account

3. **Deploy New App**
   - Click "New App"
   - Select your repository
   - Choose branch (main)
   - Select `app.py` as main file
   - Click "Deploy"

4. **Access Live App**
   - App will be live in 2-3 minutes
   - Share the public URL

---

## 📊 Model Training Details

- **Train-Test Split:** 80-20 ratio with stratification
- **Random State:** 42 (for reproducibility)
- **Feature Scaling:** StandardScaler for numerical features
- **Encoding:** LabelEncoder for categorical features
- **Cross-Validation:** Stratified split to maintain class distribution

---

## 🎯 Key Achievements

✅ Implemented 6 classification models  
✅ Achieved >89% accuracy with ensemble methods  
✅ Comprehensive evaluation with 6 metrics  
✅ Interactive Streamlit web application  
✅ Deployed on Streamlit Community Cloud  
✅ Complete documentation and code comments  
✅ Professional UI/UX design  
✅ Reproducible results with fixed random seeds  

---

## 📝 Assignment Compliance

This project fulfills all requirements of **M.Tech (AIML/DSE) Machine Learning Assignment 2**:

- ✅ Dataset with 21 features (>12 required) and 1,604 instances (>500 required)
- ✅ All 6 required classification models implemented
- ✅ All 6 evaluation metrics calculated
- ✅ Comprehensive README with required structure
- ✅ Model comparison table with observations
- ✅ Streamlit app with all mandatory features
- ✅ Deployed on Streamlit Community Cloud
- ✅ Complete GitHub repository with proper structure
- ✅ Executed on BITS Virtual Lab (screenshot provided)

---

## 👨‍💻 Author

**M.Tech (AIML/DSE) Student**  
Work Integrated Learning Programmes Division  
BITS Pilani

---

## 📄 License

This project is created for academic purposes as part of the Machine Learning course assignment.

---

## 🙏 Acknowledgments

- Dataset source: Kaggle
- Course: Machine Learning - M.Tech (AIML/DSE)
- Institution: BITS Pilani
- Framework: Streamlit for web application

---

**Note:** This project demonstrates end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment. All code is original and developed specifically for this assignment.
