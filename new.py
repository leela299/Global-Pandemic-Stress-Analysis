import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="Wellness Analytics",
    page_icon="🌸",
    layout="wide"
)


st.markdown("""
    <style>
    /* Main container styling with elegant gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        min-height: 100vh;
        background-attachment: fixed;
    }
    
    /* Elegant container styling */
    .css-1d391kg {
        background: linear-gradient(to right, #fff5f5, #fff8f8);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom radio button styling */
    .stRadio > label {
        font-weight: 500;
        color: #5d576b;
        font-family: 'Quicksand', sans-serif;
    }
    
    .stRadio > div {
        display: flex;
        gap: 12px;
        margin-top: 8px;
    }
    
    .stRadio > div > label {
        background: linear-gradient(135deg, #faf9f9, #fff);
        padding: 10px 20px;
        border: 2px solid #e9e4f0;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 80px;
        text-align: center;
        font-size: 0.95em;
        color: #5d576b;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stRadio > div > label:hover {
        border-color: #b8a9c6;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Predict button styling */
    .stButton > button {
        background: linear-gradient(135deg, #c2a5d9, #a691ce);
        color: white;
        padding: 15px 30px;
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 15px rgba(166, 145, 206, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1.1em;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 20px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #a691ce, #c2a5d9);
        box-shadow: 0 6px 20px rgba(166, 145, 206, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #fff5f5, #faf0ff);
        padding: 25px;
        border-radius: 20px;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Quicksand', sans-serif;
        background: linear-gradient(135deg, #5d576b, #a691ce);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
        font-size: 2.3em;
        letter-spacing: 1px;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        background: #fff;
        border: 2px solid #e9e4f0;
        border-radius: 12px;
        color: #5d576b;
        padding: 10px 15px;
        transition: all 0.3s ease;
        font-family: 'Quicksand', sans-serif;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #a691ce;
        box-shadow: 0 0 15px rgba(166, 145, 206, 0.2);
    }
    
    /* DataFrame styling */
    div[data-testid="stDataFrame"] {
        background: #fff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e9e4f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f5f5f5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #c2a5d9, #a691ce);
        border-radius: 4px;
    }
    
    /* Success message styling */
    .element-container div[data-testid="stMarkdownContainer"] div.success {
        background-color: #f0f7f4;
        border-left: 5px solid #7fb9a2;
        color: #2c584a;
        padding: 1em;
        border-radius: 0 10px 10px 0;
    }
    
    /* Warning message styling */
    .element-container div[data-testid="stMarkdownContainer"] div.warning {
        background-color: #fff8f0;
        border-left: 5px solid #f0b775;
        color: #8b5e2b;
        padding: 1em;
        border-radius: 0 10px 10px 0;
    }
    
    /* Error message styling */
    .element-container div[data-testid="stMarkdownContainer"] div.error {
        background-color: #fdf0f0;
        border-left: 5px solid #e6a0a0;
        color: #8b2b2b;
        padding: 1em;
        border-radius: 0 10px 10px 0;
    }
    
    /* Card container styling */
    .card {
        background: #fff;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9e4f0;
    }
    
    /* Animated background pattern */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23a691ce' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
        z-index: -1;
    }
    </style>
    
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
# [Rest of the code remains exactly the same as in the previous version, starting from here]
# Load the trained model

@st.cache_resource
def load_models():
    """
    Load all models and preprocessors at once using Streamlit's caching.
    Returns a dictionary containing all loaded objects.
    """
    model_paths = {
        'rfc': './objects/rfc.pkl',
        'poly': './objects/poly.pkl',
        'scaler': './objects/scaler.pkl',
        'scaler1': './objects/scaler1.pkl',
        'enc': './objects/enc.pkl'
    }
    
    models = {}
    for name, path in model_paths.items():
        with open(path, 'rb') as file:
            models[name] = pickle.load(file)
    
    return models

# Load all models at once using the cached function
models = load_models()

# Replace individual model references with dictionary access
model = models['rfc']
poly = models['poly']
scaler = models['scaler']
scaler1 = models['scaler1']
enc = models['enc']

# Define feature names and binary features
feature_names = [
    'Increased_Work_Hours', 'Work_From_Home', 'Hours_Worked_Per_Day',
    'Meetings_Per_Day', 'Productivity_Change', 'Health_Issue', 'Job_Security',
    'Childcare_Responsibilities', 'Commuting_Changes', 'Technology_Adaptation',
    'Salary_Changes', 'Team_Collaboration_Challenges'
]

binary_features = [
    'Increased_Work_Hours', 'Work_From_Home', 'Productivity_Change', 
    'Health_Issue', 'Job_Security', 'Childcare_Responsibilities', 
    'Commuting_Changes', 'Technology_Adaptation', 'Salary_Changes', 
    'Team_Collaboration_Challenges'
]

# Title with emoji and description
st.markdown("""
    <h1>🧠 Global Pandemic Stress Level Analysis</h1>
    <div style='text-align: center; padding: 20px; margin-bottom: 30px; background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
        <p style='color: #666; font-size: 1.1em;'>
            Analyze workplace stress levels during the pandemic based on various factors.
            Enter your information in the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for input
st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px; margin-bottom: 20px; background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); border-radius: 8px;'>
        <h2 style='color: white; font-size: 1.5em; margin: 0;'>Input Features</h2>
    </div>
    """, unsafe_allow_html=True)

def user_input_features():
    data = {}
    for feature in feature_names:
        label = feature.replace('_', ' ').title()
        if feature in ['Hours_Worked_Per_Day', 'Meetings_Per_Day']:
            data[feature] = st.sidebar.number_input(
                label,
                min_value=0.0,
                max_value=24.0 if feature == 'Hours_Worked_Per_Day' else 20.0,
                value=0.0,
                help=f"Enter your {label.lower()}"
            )
        elif feature in binary_features:
            # Convert Yes/No to 1/0
            response = st.sidebar.radio(
                label,
                options=["No", "Yes"],
                help=f"Select Yes or No for {label.lower()}",
                horizontal=True  # Make radio buttons horizontal
            )
            data[feature] = 1 if response == "Yes" else 0
        else:
            data[feature] = st.sidebar.number_input(
                label,
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                help=f"Enter value between 0 and 1 for {label.lower()}"
            )
    return pd.DataFrame([data])

input_df = user_input_features()
sector = st.sidebar.selectbox(
    'Sector', 
    ['Education', 'Healthcare', 'IT', 'Retail'],
    help="Select your sector"
)
input_df['Sector'] = sector

# Show input with styled container
st.markdown("""
    <div style='margin-bottom: 30px;'>
        <h3 style='color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;'>
            📊 User Input Features
        </h3>
    </div>
    """, unsafe_allow_html=True)
st.write(input_df)

# Prediction section
if st.button("Predict Stress Level"):
    with st.spinner('Analyzing stress levels...'):
        # Apply Scaling
        df = enc.transform(input_df[['Sector']])
        
        # Convert the transformed array into a DataFrame with appropriate column names
        df = pd.DataFrame(df, columns=enc.get_feature_names_out(['Sector']))
        input_df = pd.concat([input_df,df], axis=1)
        input_df = input_df.drop('Sector', axis=1)
        
        input_df['Work_Hours_Interaction'] = input_df['Increased_Work_Hours'] * input_df['Work_From_Home']
        
        # Transform and predict
        transformed_input = poly.transform(input_df)
        prediction = model.predict(transformed_input)
        prediction_proba = model.predict_proba(transformed_input)

        st.markdown("""
            <h3 style='color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; margin-top: 20px;'>
                📊 Prediction Probabilities
            </h3>
            """, unsafe_allow_html=True)
            
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=[f"Class {c}" for c in model.classes_]
        )
        st.write(proba_df)



# st.title("Clustering, Regression, and PCA Visualization")
st.markdown("""
            <center><div style='background-color: #eea990; padding: 15px; border-radius: 8px; margin-top: 20px;'>
                <p style='color: #667; margin: 0; font-size: 20px; font-weight: bold;'>
                    REGRESSION RESULTS
                </p>
            </div></center>
            """, unsafe_allow_html=True)

# Linear Regression Metrics Visualization
st.subheader("Linear Regression Metrics")
mse = 0.9749904538695782
r2 = -0.017024725827095022

fig_metrics, ax_metrics = plt.subplots(figsize=(8, 4))
metrics = ["Mean Squared Error", "R-squared"]
values = [mse, r2]
bars = ax_metrics.bar(metrics, values, color=["blue", "orange"])
ax_metrics.set_title("Regression Metrics", fontsize=16)
ax_metrics.set_ylabel("Values", fontsize=12)
ax_metrics.set_ylim(-1, 1.5)
for bar in bars:
    ax_metrics.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
st.pyplot(fig_metrics)

# PCA Results
st.subheader("Linear Regression with PCA Analysis Results")
pca_results = {
    "Components": [2, 3, 5, 15],
    "Mean Squared Error": [0.9592, 0.9593, 0.9592, 0.9615],
    "R-squared": [-0.0006, -0.0007, -0.0006, -0.0030],
}
df_pca = pd.DataFrame(pca_results)

# Display PCA Results
st.write("PCA Results Table:")
st.dataframe(df_pca)

# Plot PCA Metrics
fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
ax_pca.plot(df_pca["Components"], df_pca["Mean Squared Error"], marker="o", label="Mean Squared Error", color="blue")
ax_pca.set_title("PCA Analysis: Metrics vs. Components", fontsize=16)
ax_pca.set_xlabel("Number of Components", fontsize=12)
ax_pca.set_ylabel("Metrics", fontsize=12)
ax_pca.legend(loc="upper left")

# Overlay R-squared
ax_r2 = ax_pca.twinx()
ax_r2.plot(df_pca["Components"], df_pca["R-squared"], marker="x", label="R-squared", color="orange")
ax_r2.set_ylabel("R-squared", fontsize=12)
ax_r2.legend(loc="upper right")

st.pyplot(fig_pca)


st.markdown("""
            <center><div style='background-color: #eea990; padding: 15px; border-radius: 8px; margin-top: 20px;'>
                <p style='color: #667; margin: 0; font-size: 20px; font-weight: bold;'>
                    DIFFERENT CLASSIFICATION MODELS
                </p>
            </div></center>
            """, unsafe_allow_html=True)


# Precision, Recall, and F1-Score (Original)
st.title("1. Random Forest Classifier Results")
classification_metrics = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-Score"],
    "Class 1": [0.72, 0.72, 0.72],
    "Class 2": [0.53, 0.59, 0.56],
    "Class 3": [0.59, 0.53, 0.56]
})
st.write("Metrics Table:")
st.dataframe(classification_metrics)

# Plot Original Metrics
fig_metrics, ax_metrics = plt.subplots(figsize=(8, 4))
x = np.arange(len(classification_metrics["Metric"]))
width = 0.25
ax_metrics.bar(x - width, classification_metrics["Class 1"], width, label="Class 1", color="blue")
ax_metrics.bar(x, classification_metrics["Class 2"], width, label="Class 2", color="orange")
ax_metrics.bar(x + width, classification_metrics["Class 3"], width, label="Class 3", color="green")

ax_metrics.set_title("Randomforest Precision, Recall, and F1-Score by Class", fontsize=16)
ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(classification_metrics["Metric"])
ax_metrics.set_ylabel("Score", fontsize=12)
ax_metrics.set_ylim(0, 1)
ax_metrics.legend(loc="upper right")

st.pyplot(fig_metrics)

# Precision, Recall, and F1-Score (After Hyperparameter Tuning)
st.title("2. Randomforest Hyperparameter Tuned Results")
tuned_metrics = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-Score"],
    "Class 1": [0.75, 0.71, 0.73],
    "Class 2": [0.57, 0.64, 0.61],
    "Class 3": [0.70, 0.65, 0.67]
})
st.write("Hyperparameter-Tuned Classification Metrics Table:")
st.dataframe(tuned_metrics)

# Plot Tuned Metrics
fig_tuned, ax_tuned = plt.subplots(figsize=(8, 4))
x = np.arange(len(tuned_metrics["Metric"]))
ax_tuned.bar(x - width, tuned_metrics["Class 1"], width, label="Class 1", color="blue")
ax_tuned.bar(x, tuned_metrics["Class 2"], width, label="Class 2", color="orange")
ax_tuned.bar(x + width, tuned_metrics["Class 3"], width, label="Class 3", color="green")

ax_tuned.set_title("Hyperparameter-Tuned Precision, Recall, and F1-Score by Class", fontsize=16)
ax_tuned.set_xticks(x)
ax_tuned.set_xticklabels(tuned_metrics["Metric"])
ax_tuned.set_ylabel("Score", fontsize=12)
ax_tuned.set_ylim(0, 1)
ax_tuned.legend(loc="upper right")

st.pyplot(fig_tuned)

# Naive Bayes Classifier Results
st.title("3. Naive Bayes Classifier Results")

# Naive Bayes Precision, Recall, F1-Score
nb_metrics = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-Score"],
    "Class 1": [0.42, 0.38, 0.40],
    "Class 2": [0.34, 0.34, 0.34],
    "Class 3": [0.38, 0.41, 0.40]
})
st.write("Naive Bayes Classification Metrics Table:")
st.dataframe(nb_metrics)

# Plot Naive Bayes Metrics
fig_nb, ax_nb = plt.subplots(figsize=(8, 4))
x = np.arange(len(nb_metrics["Metric"]))
ax_nb.bar(x - width, nb_metrics["Class 1"], width, label="Class 1", color="blue")
ax_nb.bar(x, nb_metrics["Class 2"], width, label="Class 2", color="orange")
ax_nb.bar(x + width, nb_metrics["Class 3"], width, label="Class 3", color="green")

ax_nb.set_title("Naive Bayes Precision, Recall, and F1-Score by Class", fontsize=16)
ax_nb.set_xticks(x)
ax_nb.set_xticklabels(nb_metrics["Metric"])
ax_nb.set_ylabel("Score", fontsize=12)
ax_nb.set_ylim(0, 1)
ax_nb.legend(loc="upper right")

st.pyplot(fig_nb)



classification_report_str = {
    "0": {"precision": 0.26, "recall": 0.20, "f1-score": 0.23},
    "1": {"precision": 0.51, "recall": 0.84, "f1-score": 0.63},
    "2": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00}
}

# Extract metrics for plotting
metrics = {
    "Metric": ["Precision", "Recall", "F1-Score"],
    "Class 0": [classification_report_str["0"]["precision"], classification_report_str["0"]["recall"], classification_report_str["0"]["f1-score"]],
    "Class 1": [classification_report_str["1"]["precision"], classification_report_str["1"]["recall"], classification_report_str["1"]["f1-score"]],
    "Class 2": [classification_report_str["2"]["precision"], classification_report_str["2"]["recall"], classification_report_str["2"]["f1-score"]]
}

# Create a DataFrame from the metrics
df_metrics = pd.DataFrame(metrics)

# Streamlit display
st.title("4. Voting Classifier Performance Metrics")

# Display the metrics table
st.write("### Classification Report Table:")
st.dataframe(df_metrics)

# Plot the metrics
fig_metrics, ax_metrics = plt.subplots(figsize=(8, 4))
x = range(len(df_metrics["Metric"]))
width = 0.25
ax_metrics.bar([p - width for p in x], df_metrics["Class 0"], width, label="Class 0", color="blue")
ax_metrics.bar(x, df_metrics["Class 1"], width, label="Class 1", color="orange")
ax_metrics.bar([p + width for p in x], df_metrics["Class 2"], width, label="Class 2", color="green")

ax_metrics.set_title("Voting Classifier Precision, Recall, and F1-Score by Class", fontsize=16)
ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(df_metrics["Metric"])
ax_metrics.set_ylabel("Score", fontsize=12)
ax_metrics.set_ylim(0, 1)
ax_metrics.legend(loc="upper right")

# Display the plot in Streamlit
st.pyplot(fig_metrics)

classification_report_catboost = {
    "1": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00},
    "2": {"precision": 0.51, "recall": 1.00, "f1-score": 0.67},
    "3": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00}
}

# Extract metrics for plotting
metrics_catboost = {
    "Metric": ["Precision", "Recall", "F1-Score"],
    "Class 1": [classification_report_catboost["1"]["precision"], classification_report_catboost["1"]["recall"], classification_report_catboost["1"]["f1-score"]],
    "Class 2": [classification_report_catboost["2"]["precision"], classification_report_catboost["2"]["recall"], classification_report_catboost["2"]["f1-score"]],
    "Class 3": [classification_report_catboost["3"]["precision"], classification_report_catboost["3"]["recall"], classification_report_catboost["3"]["f1-score"]]
}

# Create a DataFrame from the metrics
df_metrics_catboost = pd.DataFrame(metrics_catboost)

# Streamlit display
st.title("5. CatBoost Classifier Performance Metrics")

# Display the metrics table
st.write("### CatBoost Classification Report Table:")
st.dataframe(df_metrics_catboost)

# Plot the metrics
fig_metrics_catboost, ax_metrics_catboost = plt.subplots(figsize=(8, 4))
x = range(len(df_metrics_catboost["Metric"]))
width = 0.25
ax_metrics_catboost.bar([p - width for p in x], df_metrics_catboost["Class 1"], width, label="Class 1", color="blue")
ax_metrics_catboost.bar(x, df_metrics_catboost["Class 2"], width, label="Class 2", color="orange")
ax_metrics_catboost.bar([p + width for p in x], df_metrics_catboost["Class 3"], width, label="Class 3", color="green")

ax_metrics_catboost.set_title("CatBoost Classifier Precision, Recall, and F1-Score by Class", fontsize=16)
ax_metrics_catboost.set_xticks(x)
ax_metrics_catboost.set_xticklabels(df_metrics_catboost["Metric"])
ax_metrics_catboost.set_ylabel("Score", fontsize=12)
ax_metrics_catboost.set_ylim(0, 1)
ax_metrics_catboost.legend(loc="upper right")

# Display the plot in Streamlit
st.pyplot(fig_metrics_catboost)

classification_report_lgbm = {
    "0": {"precision": 0.70, "recall": 0.64, "f1-score": 0.67},
    "1": {"precision": 0.54, "recall": 0.62, "f1-score": 0.58},
    "2": {"precision": 0.53, "recall": 0.48, "f1-score": 0.50}
}

# Extract metrics for plotting
metrics_lgbm = {
    "Metric": ["Precision", "Recall", "F1-Score"],
    "Class 0": [classification_report_lgbm["0"]["precision"], classification_report_lgbm["0"]["recall"], classification_report_lgbm["0"]["f1-score"]],
    "Class 1": [classification_report_lgbm["1"]["precision"], classification_report_lgbm["1"]["recall"], classification_report_lgbm["1"]["f1-score"]],
    "Class 2": [classification_report_lgbm["2"]["precision"], classification_report_lgbm["2"]["recall"], classification_report_lgbm["2"]["f1-score"]]
}

# Create a DataFrame from the metrics
df_metrics_lgbm = pd.DataFrame(metrics_lgbm)

# Streamlit display
st.title("6. LGBM Classifier Performance Metrics")

# Display the metrics table
st.write("### LGBM Classification Report Table:")
st.dataframe(df_metrics_lgbm)

# Plot the metrics
fig_metrics_lgbm, ax_metrics_lgbm = plt.subplots(figsize=(8, 4))
x = range(len(df_metrics_lgbm["Metric"]))
width = 0.25
ax_metrics_lgbm.bar([p - width for p in x], df_metrics_lgbm["Class 0"], width, label="Class 0", color="blue")
ax_metrics_lgbm.bar(x, df_metrics_lgbm["Class 1"], width, label="Class 1", color="orange")
ax_metrics_lgbm.bar([p + width for p in x], df_metrics_lgbm["Class 2"], width, label="Class 2", color="green")

ax_metrics_lgbm.set_title("LGBM Classifier Precision, Recall, and F1-Score by Class", fontsize=16)
ax_metrics_lgbm.set_xticks(x)
ax_metrics_lgbm.set_xticklabels(df_metrics_lgbm["Metric"])
ax_metrics_lgbm.set_ylabel("Score", fontsize=12)
ax_metrics_lgbm.set_ylim(0, 1)
ax_metrics_lgbm.legend(loc="upper right")

# Display the plot in Streamlit
st.pyplot(fig_metrics_lgbm)

st.markdown("""
            <center><div style='background-color: #eea990; padding: 15px; border-radius: 8px; margin-top: 20px;'>
                <p style='color: #667; margin: 0; font-size: 20px; font-weight: bold;'>
                    ACCURACY COMPARISION OF DIFFERENT MODELS
                </p>
            </div></center>
            """, unsafe_allow_html=True)

# Model accuracy data (hardcoded for comparison)
models = [
    "Random Forest",
    "Random Forest (Tuned)",
    "GaussianNB",
    "Voting Classifier",
    "CatBoost"
]

accuracies = [
    0.6130,  # Random Forest
    0.6700,  # Random Forest with Hyperparameter Tuning
    0.3779,  # GaussianNB
    0.4670,  # Voting Classifier
    0.4670   # CatBoost
]

# Streamlit display
st.title("Model Accuracy Comparison")

# Bar Plot
fig_comparison, ax_comparison = plt.subplots(figsize=(10, 6))
ax_comparison.barh(models, accuracies, color=['blue', 'green', 'red', 'orange', 'purple'])

# Add title and labels
ax_comparison.set_title("Model Accuracy Comparison", fontsize=16)
ax_comparison.set_xlabel("Accuracy", fontsize=12)
ax_comparison.set_xlim(0, 1)

# Display the accuracy values on the bars
for index, value in enumerate(accuracies):
    ax_comparison.text(value + 0.01, index, f"{value:.4f}", va='center', fontsize=10)

# Display the plot in Streamlit
st.pyplot(fig_comparison)

st.markdown("""
            <center><div style='background-color: #eea990; padding: 15px; border-radius: 8px; margin-top: 20px;'>
                <p style='color: #667; margin: 0; font-size: 20px; font-weight: bold;'>
                    HOURS WORKED VS MEETING ANALYSIS
                </p>
            </div></center>
            """, unsafe_allow_html=True)


# st.title("Hours worked vs Meeting Worked Analysis")
df = pd.read_csv('data.csv')
numeric_features = ['Hours_Worked_Per_Day', 'Meetings_Per_Day']

# Standardize numeric features
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Use only numerical columns for clustering
X = df[numeric_features]

# KMeans Clustering
st.subheader("KMeans Clustering")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
df['KMeans_Cluster'] = kmeans_labels

# Plot KMeans
fig_kmeans, ax_kmeans = plt.subplots(figsize=(10, 6))
scatter_kmeans = ax_kmeans.scatter(
    df['Hours_Worked_Per_Day'],
    df['Meetings_Per_Day'],
    c=kmeans_labels,
    cmap='viridis',
    s=100
)
ax_kmeans.set_title('KMeans Clustering (Numerical Features Only)', fontsize=16)
ax_kmeans.set_xlabel('Hours Worked Per Day', fontsize=12)
ax_kmeans.set_ylabel('Meetings Per Day', fontsize=12)
fig_kmeans.colorbar(scatter_kmeans, ax=ax_kmeans, label='Cluster')
st.pyplot(fig_kmeans)

# DBSCAN Clustering
st.subheader("DBSCAN Clustering")
eps = st.slider("Set DBSCAN eps value:", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
min_samples = st.slider("Set DBSCAN min_samples value:", min_value=1, max_value=10, value=5, step=1)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X)
df['DBSCAN_Cluster'] = dbscan_labels

# Plot DBSCAN
fig_dbscan, ax_dbscan = plt.subplots(figsize=(10, 6))
scatter_dbscan = ax_dbscan.scatter(
    df['Hours_Worked_Per_Day'],
    df['Meetings_Per_Day'],
    c=dbscan_labels,
    cmap='viridis',
    s=100
)
ax_dbscan.set_title('DBSCAN Clustering (Numerical Features Only)', fontsize=16)
ax_dbscan.set_xlabel('Hours Worked Per Day', fontsize=12)
ax_dbscan.set_ylabel('Meetings Per Day', fontsize=12)
fig_dbscan.colorbar(scatter_dbscan, ax=ax_dbscan, label='Cluster')
st.pyplot(fig_dbscan)


# Hide Streamlit's footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
