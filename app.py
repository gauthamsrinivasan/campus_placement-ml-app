import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Campus Placement Prediction - ML Assignment 2",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Style metric containers */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #ffffff !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #2c3e50 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    [data-testid="stMetricDelta"] {
        color: #27ae60 !important;
        font-weight: 500;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid #5a67d8;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .highlight-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style='margin:0; font-size: 2.5rem;'>🎓 Campus Placement Prediction</h1>
    <p style='margin:0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>Machine Learning Classification System</p>
    <p style='margin:0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.8;'>M.Tech (AIML/DSE) – Assignment 2</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA FUNCTION
# --------------------------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv("data/campus_placement.csv")

# --------------------------------------------------
# PREPROCESS DATA FUNCTION
# --------------------------------------------------
@st.cache_data
def preprocess_data(df):
    df = df.drop(columns=["salary_lpa", "student_id"], errors="ignore")
    
    # Check if 'placed' column exists
    if 'placed' not in df.columns:
        raise ValueError("Dataset must contain 'placed' column as target variable")
    
    X = df.drop("placed", axis=1)
    y = df["placed"]
    
    # Check minimum dataset size
    if len(df) < 4:
        raise ValueError(f"Dataset too small ({len(df)} rows). Minimum 4 rows required for train-test split.")
    
    # Check class distribution
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Dataset must have both classes (0 and 1) in the 'placed' column")
    
    categorical_cols = [
        "gender", "city_tier", "ssc_board", "hsc_board",
        "hsc_stream", "degree_field", "specialization"
    ]
    
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    
    encoder = LabelEncoder()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = encoder.fit_transform(X[col])
    
    # Only use stratify if both classes have at least 2 samples
    stratify_param = y if class_counts.min() >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )
    
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, X_test, y_train, y_test, df

# Initialize with default data
df = load_default_data()
X_train, X_test, y_train, y_test, processed_df = preprocess_data(df)

# --------------------------------------------------
# TRAIN MODELS FUNCTION
# --------------------------------------------------
@st.cache_resource
def train_models(_X_train, _y_train, _X_test, _y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(_X_train, _y_train)
        trained[name] = model
        y_pred = model.predict(_X_test)
        y_prob = model.predict_proba(_X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(_y_test, y_pred), 4),
            "AUC": round(roc_auc_score(_y_test, y_prob), 4),
            "Precision": round(precision_score(_y_test, y_pred), 4),
            "Recall": round(recall_score(_y_test, y_pred), 4),
            "F1 Score": round(f1_score(_y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(_y_test, y_pred), 4),
            "Confusion Matrix": confusion_matrix(_y_test, y_pred),
            "Classification Report": classification_report(_y_test, y_pred)
        })

    return pd.DataFrame(results), trained

results_df, trained_models = train_models(X_train, y_train, X_test, y_test)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>📊 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload Dataset", "📈 Model Comparison", "🔮 Model Evaluation", "📋 Dataset Info"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
        <p style='margin: 0; font-weight: bold; color: #333;'>📚 Assignment 2</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #555;'>Binary Classification<br>6 ML Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    st.success(f"🏆 **Best Model**\n\n{best_model['Model']}\n\nAccuracy: {best_model['Accuracy']:.2%}")

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%); padding: 1.5rem; border-radius: 10px; border-left: 5px solid #00acc1;'>
            <h3 style='margin-top: 0; color: #006064;'>🎯 Problem Statement</h3>
            <p style='color: #00838f; line-height: 1.8;'>
                Predict whether a student will be <strong>placed</strong> during campus recruitment based on:
            </p>
            <ul style='color: #00838f; line-height: 1.8;'>
                <li>📚 Academic Performance (SSC, HSC, Degree, MBA)</li>
                <li>💡 Skills (Technical, Soft Skills, Aptitude, Communication)</li>
                <li>💼 Experience (Internships, Projects, Certifications, Work Experience)</li>
                <li>🌍 Background (City Tier, Leadership, Extracurricular Activities)</li>
            </ul>
            <p style='color: #006064; font-weight: bold; margin-bottom: 0;'>Type: Binary Classification (Placed = 1, Not Placed = 0)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.5rem; border-radius: 10px; border-left: 5px solid #ff9800;'>
            <h4 style='margin-top: 0; color: #e65100;'>📊 Dataset Stats</h4>
            <p style='color: #ef6c00; margin: 0.5rem 0; font-size: 1.1rem;'><strong>Records:</strong> {len(processed_df):,}</p>
            <p style='color: #ef6c00; margin: 0.5rem 0; font-size: 1.1rem;'><strong>Features:</strong> {processed_df.shape[1] - 1}</p>
            <p style='color: #ef6c00; margin: 0.5rem 0; font-size: 1.1rem;'><strong>Target:</strong> Binary</p>
            <p style='color: #ef6c00; margin: 0.5rem 0 0 0; font-size: 1.1rem;'><strong>Models:</strong> 6</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #667eea;'>📈 Performance Metrics</h3>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📊 Total Records", f"{len(processed_df):,}", delta="Complete")
    c2.metric("🔢 Input Features", X_train.shape[1], delta="Processed")
    c3.metric("🎓 Training Set", f"{len(X_train):,}", delta="80%")
    c4.metric("🧪 Test Set", f"{len(X_test):,}", delta="20%")
    c5.metric("🏆 Best Accuracy", f"{results_df['Accuracy'].max():.2%}", delta="Random Forest")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional metrics row
    best_model_row = results_df.loc[results_df['Accuracy'].idxmax()]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🎯 Best F1 Score", f"{results_df['F1 Score'].max():.4f}")
    m2.metric("📊 Best AUC", f"{results_df['AUC'].max():.4f}")
    m3.metric("🎲 Best MCC", f"{results_df['MCC'].max():.4f}", help="Matthews Correlation Coefficient")
    m4.metric("✅ Best Precision", f"{results_df['Precision'].max():.4f}")
    m5.metric("🔍 Best Recall", f"{results_df['Recall'].max():.4f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Performance Overview
    st.markdown("<h3 style='text-align: center; color: #667eea;'>🎯 Model Performance Overview</h3>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    models = results_df['Model'].tolist()
    accuracies = results_df['Accuracy'].tolist()
    colors = ['#ff6b6b' if acc < 0.85 else '#feca57' if acc < 0.88 else '#48dbfb' for acc in accuracies]
    
    bars = ax.barh(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Accuracy Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0.75, 0.92)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2, f'{acc:.2%}', 
                va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #667eea;'>🔍 Sample Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(processed_df.head(10), width='stretch', height=400)

# --------------------------------------------------
# UPLOAD DATASET PAGE
# --------------------------------------------------
elif page == "📤 Upload Dataset":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>📤 Upload Custom Dataset</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
        <p style='margin: 0; color: #1565c0; font-size: 1.1rem;'>
            📋 Upload a CSV file with the same structure as the training data for predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download sample CSV button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        sample_data = processed_df.head(10).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Test CSV",
            data=sample_data,
            file_name="sample_test_data.csv",
            mime="text/csv",
            type="primary",
            help="Download a sample CSV file to see the required format"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your dataset in CSV format")
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {uploaded_df.shape}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Rows", uploaded_df.shape[0])
            col2.metric("📋 Columns", uploaded_df.shape[1])
            col3.metric("💾 Size", f"{uploaded_file.size / 1024:.2f} KB")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(uploaded_df.head(), width='stretch', height=300)
            
            if st.button("🚀 Process Uploaded Data", type="primary"):
                with st.spinner("Processing data..."):
                    try:
                        X_train_new, X_test_new, y_train_new, y_test_new, processed_new = preprocess_data(uploaded_df)
                        st.session_state['uploaded_data'] = (X_train_new, X_test_new, y_train_new, y_test_new)
                        st.success("✅ Data processed successfully! Go to Model Evaluation to see results.")
                        st.balloons()
                    except ValueError as ve:
                        st.error(f"❌ Validation Error: {str(ve)}")
                        st.info("💡 **Tips:**\n- Ensure your dataset has at least 4 rows\n- Both classes (0 and 1) must be present in 'placed' column\n- Dataset will be processed without stratification if classes are imbalanced")
                        st.warning("⚠️ Note: Imbalanced datasets may result in biased model predictions.")
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%); padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;'>
            <h3 style='color: #f57f17; margin: 0;'>⚠️ No File Uploaded</h3>
            <p style='color: #f9a825; margin: 1rem 0 0 0;'>Please upload a CSV file to proceed with predictions</p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# MODEL COMPARISON PAGE
# --------------------------------------------------
elif page == "📈 Model Comparison":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>🏆 Model Performance Comparison</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.1rem;'>All 6 Classification Models - Evaluation Metrics</p>", unsafe_allow_html=True)
    
    display_df = results_df.drop(columns=["Confusion Matrix", "Classification Report"]).copy()
    
    # Highlight best values
    def highlight_max(s):
        if s.name != 'Model':
            is_max = s == s.max()
            return ['background-color: #90EE90; font-weight: bold' if v else '' for v in is_max]
        return ['' for _ in s]
    
    styled_df = display_df.style.apply(highlight_max, axis=0)
    st.dataframe(styled_df, width='stretch', height=300)
    
    st.markdown("---")
    st.subheader("📊 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
        x = np.arange(len(results_df))
        width = 0.12
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, results_df[metric], width, label=metric)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        results_df.plot(x='Model', y=['Accuracy', 'F1 Score', 'MCC'], kind='bar', ax=ax)
        ax.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(title='Metrics')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); padding: 1.5rem; border-radius: 10px; text-align: center; border: 3px solid #4caf50;'>
        <h3 style='margin: 0; color: #1b5e20;'>🏆 Best Performing Model</h3>
        <h2 style='margin: 0.5rem 0; color: #2e7d32;'>{best_model['Model']}</h2>
        <p style='margin: 0; font-size: 1.5rem; color: #388e3c; font-weight: bold;'>Accuracy: {best_model['Accuracy']:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------
elif page == "🔮 Model Evaluation":
    st.markdown("<h2 style='text-align: center; color: #667eea;'>🔮 Detailed Model Evaluation</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_model = st.selectbox(
            "📌 Select a Model to Evaluate:",
            results_df["Model"].tolist(),
            index=4
        )
    
    row = results_df[results_df["Model"] == selected_model].iloc[0]
    
    st.markdown(f"<h3 style='text-align: center; color: #764ba2; margin: 2rem 0 1rem 0;'>📊 Performance Metrics for {selected_model}</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{row['Accuracy']:.4f}")
    col2.metric("AUC", f"{row['AUC']:.4f}")
    col3.metric("Precision", f"{row['Precision']:.4f}")
    col4.metric("Recall", f"{row['Recall']:.4f}")
    col5.metric("F1 Score", f"{row['F1 Score']:.4f}")
    col6.metric("MCC", f"{row['MCC']:.4f}")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🔢 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            row["Confusion Matrix"], 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'],
            ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_b:
        st.subheader("📋 Classification Report")
        st.text(row["Classification Report"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #0288d1;'>
        <p style='margin: 0; color: #01579b;'><strong>💡 Tip:</strong> Use the dropdown above to compare different models and analyze their performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# DATASET INFO PAGE
# --------------------------------------------------
else:
    st.markdown("<h2 style='text-align: center; color: #667eea;'>📋 Dataset Information</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
    ### 📚 Campus Placement Prediction Dataset
    
    **Source**: Kaggle  
    **Problem Type**: Binary Classification  
    **Target Variable**: `placed` (1 = Placed, 0 = Not Placed)
    
    ---
    
    ### 📊 Dataset Characteristics
    - **Total Instances**: {len(processed_df):,}
    - **Total Features**: {processed_df.shape[1] - 1} (Input Features)
    - **Target Classes**: 2 (Binary)
    - **Missing Values**: Handled appropriately
    
    ---
    
    ### 🔍 Feature Categories
    
    **1. Academic Performance**
    - SSC Percentage, SSC Board
    - HSC Percentage, HSC Board, HSC Stream
    - Degree Percentage, Degree Field
    - MBA Percentage, Specialization
    
    **2. Skills & Competency**
    - Technical Skills Score
    - Soft Skills Score
    - Aptitude Score
    - Communication Score
    
    **3. Experience & Activities**
    - Internships Count
    - Projects Count
    - Certifications Count
    - Work Experience (months)
    - Leadership Roles
    - Extracurricular Activities
    - Backlogs
    
    **4. Demographics**
    - Gender
    - Age
    - City Tier
    
    ---
    
    ### 🎯 Target Variable
    - **placed**: Binary (1 = Placed, 0 = Not Placed)
    
    ---
    
    ### 📈 Data Distribution
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        placement_counts = processed_df['placed'].value_counts().sort_index()
        ax.bar(['Not Placed', 'Placed'], placement_counts.values, color=['#ff7f0e', '#1f77b4'])
        ax.set_title('Target Variable Distribution', fontsize=13, fontweight='bold')
        ax.set_xlabel('Placement Status', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("""\n\n\n
        **Class Distribution:**
        """)
        placement_counts = processed_df['placed'].value_counts()
        st.write(f"- Placed (1): {placement_counts.get(1, 0)} students")
        st.write(f"- Not Placed (0): {placement_counts.get(0, 0)} students")
        st.write(f"- Total: {len(processed_df)} students")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🎓 M.Tech (AIML/DSE) - Machine Learning Assignment 2 | Built with Streamlit</p>",
    unsafe_allow_html=True
)
