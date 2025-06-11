import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌸 Iris Species Classifier")
st.markdown("### Machine Learning Web App using Random Forest")
st.markdown("This app predicts the species of Iris flowers based on their measurements.")

# Load and prepare data
@st.cache_data
def load_data():
    # For demo purposes, we'll use the built-in iris dataset from sklearn
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'})
    return df

# Train model
@st.cache_data
def train_model(df):
    X = df.iloc[:, :-2]  # Features (excluding species and species_name)
    y = df['species']    # Target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Train model
    classifier = RandomForestClassifier(random_state=0)
    classifier.fit(X_train, y_train)
    
    # Predictions
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return classifier, accuracy, X_test, y_test, y_pred

# Load data and train model
df = load_data()
model, accuracy, X_test, y_test, y_pred = train_model(df)

# Sidebar for user input
st.sidebar.header("🔧 Input Features")
st.sidebar.markdown("Adjust the sliders to input flower measurements:")

# Input sliders
sepal_length = st.sidebar.slider("Sepal Length (cm)", 
                                float(df['sepal length (cm)'].min()), 
                                float(df['sepal length (cm)'].max()), 
                                float(df['sepal length (cm)'].mean()))

sepal_width = st.sidebar.slider("Sepal Width (cm)", 
                               float(df['sepal width (cm)'].min()), 
                               float(df['sepal width (cm)'].max()), 
                               float(df['sepal width (cm)'].mean()))

petal_length = st.sidebar.slider("Petal Length (cm)", 
                                float(df['petal length (cm)'].min()), 
                                float(df['petal length (cm)'].max()), 
                                float(df['petal length (cm)'].mean()))

petal_width = st.sidebar.slider("Petal Width (cm)", 
                               float(df['petal width (cm)'].min()), 
                               float(df['petal width (cm)'].max()), 
                               float(df['petal width (cm)'].mean()))

# Make prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
predicted_species = species_names[prediction]

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Prediction Results")
    st.markdown(f"**Predicted Species:** `{predicted_species}`")
    st.markdown(f"**Confidence:** `{max(prediction_proba):.2%}`")
    
    # Probability bar chart
    prob_df = pd.DataFrame({
        'Species': species_names,
        'Probability': prediction_proba
    })
    
    fig_prob = px.bar(prob_df, x='Species', y='Probability', 
                      title="Prediction Probabilities",
                      color='Probability',
                      color_continuous_scale='viridis')
    fig_prob.update_layout(height=400)
    st.plotly_chart(fig_prob, use_container_width=True)

with col2:
    st.subheader("📊 Input Summary")
    input_df = pd.DataFrame({
        'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
    })
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    st.subheader("🎯 Model Performance")
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': df.columns[:-2],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_imp = px.bar(feature_importance, x='Importance', y='Feature', 
                     orientation='h', title="Feature Importance")
    fig_imp.update_layout(height=300)
    st.plotly_chart(fig_imp, use_container_width=True)

# Dataset visualization
st.subheader("📈 Dataset Visualization")

col3, col4 = st.columns([1, 1])

with col3:
    # Scatter plot
    fig_scatter = px.scatter(df, x='sepal length (cm)', y='sepal width (cm)', 
                            color='species_name', title="Sepal Length vs Width",
                            hover_data=['petal length (cm)', 'petal width (cm)'])
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    # Petal scatter plot
    fig_petal = px.scatter(df, x='petal length (cm)', y='petal width (cm)', 
                          color='species_name', title="Petal Length vs Width",
                          hover_data=['sepal length (cm)', 'sepal width (cm)'])
    st.plotly_chart(fig_petal, use_container_width=True)

# Confusion Matrix
st.subheader("🔍 Model Evaluation")
col5, col6 = st.columns([1, 1])

with col5:
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=species_names, yticklabels=species_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    st.pyplot(fig_cm)

with col6:
    # Classification report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=species_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(2), use_container_width=True)

# Dataset info
st.subheader("📊 Dataset Information")
col7, col8 = st.columns([1, 1])

with col7:
    st.markdown("**Dataset Overview:**")
    st.write(f"- Total samples: {len(df)}")
    st.write(f"- Features: {len(df.columns)-2}")
    st.write(f"- Classes: {df['species_name'].nunique()}")
    
    # Species distribution
    species_count = df['species_name'].value_counts()
    fig_dist = px.pie(values=species_count.values, names=species_count.index,
                      title="Species Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

with col8:
    st.markdown("**Sample Data:**")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("**Built with Streamlit** • **Model: Random Forest** • **Dataset: Iris**")
