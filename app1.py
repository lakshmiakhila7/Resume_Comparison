import streamlit as st
import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from docx import Document  # Ensure python-docx is installed
import fitz  # PyMuPDF
import json
from streamlit_lottie import st_lottie
from imblearn.over_sampling import SMOTE

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Preprocess text function
def preprocess_text(text):
    doc = nlp(text)
    clean = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(clean)

# Function to extract text from a PDF file using PyMuPDF
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

# Function to load Lottie animation
def load_lottie_animation(path):
    with open(path, "r") as file:
        return json.load(file)

# Function to compare two resumes
def compare_resumes(resume_text1, resume_text2, df):
    # Preprocess text
    text1 = preprocess_text(resume_text1)
    text2 = preprocess_text(resume_text2)
    
    # Load and preprocess the dataset
    df['Resume_text'] = df['Resume'].apply(preprocess_text)
    label_encoder = LabelEncoder()
    df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])
    print("Category Distribution in Dataset:")
    print(df['Category'].value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['Resume_text'], df['Category_Encoded'], test_size=0.20, random_state=2, stratify=df['Category'])
    
    # Model pipelines
    models = {
        "K-Nearest Neighbors": Pipeline([('vectorizer', TfidfVectorizer()), ('model', KNeighborsClassifier())]),
        "Logistic Regression": Pipeline([('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)), ('model', LogisticRegression())]),
        "Random Forest": Pipeline([('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)), ('model', RandomForestClassifier())])
    }

    results = {}
    confusion_matrices = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy

        # Debugging model performance
        print(f"{model_name} Accuracy: {accuracy:.2f}")

        # Print Logistic Regression coefficients after training
        if model_name == "Logistic Regression":
            logistic_model = model.named_steps['model']
            print("Logistic Regression Coefficients:")
            print(logistic_model.coef_)

        # Print Random Forest feature importances after training
        if model_name == "Random Forest":
            random_forest_model = model.named_steps['model']
            print("Random Forest Feature Importances:")
            print(random_forest_model.feature_importances_)

        # Calculate and store confusion matrix
        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

    # Predictions for the uploaded resumes
    predictions = {}
    for model_name, model in models.items():
        prediction1 = model.predict([text1])[0]
        prediction2 = model.predict([text2])[0]
        predictions[model_name] = {
            "resume1": label_encoder.inverse_transform([prediction1])[0],
            "resume2": label_encoder.inverse_transform([prediction2])[0]
        }
        print(f"Model: {model_name}, Resume 1 Prediction: {predictions[model_name]['resume1']}, Resume 2 Prediction: {predictions[model_name]['resume2']}")

    # Determine which resume is better based on predictions
    better_resume = {}
    for model_name in predictions.keys():
        better_resume[model_name] = (
            "Resume 1" if predictions[model_name]["resume1"] != predictions[model_name]["resume2"] and predictions[model_name]["resume1"] > predictions[model_name]["resume2"] else "Resume 2"
        )
        print(f"Based on {model_name}: {better_resume[model_name]}")
    
    return predictions, results, better_resume, confusion_matrices

# Streamlit app layout
st.title(" Recruitment using Business Intelligence")
st.write("Upload two resumes and compare their performance based on classification models.")

# Sidebar for file uploaders
st.sidebar.header("Upload Resumes")
uploaded_file1 = st.sidebar.file_uploader("Upload the first resume", type=["txt", "pdf", "docx"], key="file_uploader1")
uploaded_file2 = st.sidebar.file_uploader("Upload the second resume", type=["txt", "pdf", "docx"], key="file_uploader2")

# Load and display Lottie animation
lottie_path = "C:\\Users\\user\\Desktop\\resume1\\Animation - 1723202433281.json"
lottie_animation = load_lottie_animation(lottie_path)
st_lottie(lottie_animation, speed=1, width=500, height=400, key="lottie")

# Extract text from uploaded files
resume_text1 = ""
resume_text2 = ""

if uploaded_file1 is not None:
    if uploaded_file1.type == "application/pdf":
        resume_text1 = extract_text_from_pdf(uploaded_file1)
    elif uploaded_file1.type == "text/plain":
        resume_text1 = uploaded_file1.getvalue().decode("utf-8")
    elif uploaded_file1.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text1 = extract_text_from_docx(uploaded_file1)

if uploaded_file2 is not None:
    if uploaded_file2.type == "application/pdf":
        resume_text2 = extract_text_from_pdf(uploaded_file2)
    elif uploaded_file2.type == "text/plain":
        resume_text2 = uploaded_file2.getvalue().decode("utf-8")
    elif uploaded_file2.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text2 = extract_text_from_docx(uploaded_file2)

if resume_text1 and resume_text2:
    st.write("**Resume 1 Text:**")
    st.text_area("Resume 1 Text", resume_text1, height=300)
    
    st.write("**Resume 2 Text:**")
    st.text_area("Resume 2 Text", resume_text2, height=300)

    # Analyze resumes
    if st.button("Compare Resumes"):
        # Load the dataset
        df = pd.read_csv("C:\\Users\\user\\Desktop\\resume1\\resume_dataset.csv")
        
        predictions, results, better_resume, confusion_matrices = compare_resumes(resume_text1, resume_text2, df)

        # Main content area
        st.header("Visualization and Results")   

        st.subheader("Model Predictions")
        for model_name, prediction in predictions.items():
            st.write(f"**{model_name} Predictions:**")
            st.write(f"Resume 1: {prediction['resume1']}")
            st.write(f"Resume 2: {prediction['resume2']}")

        st.subheader("Model Accuracy")
        accuracy_list = []
        for model_name, accuracy in results.items():
            st.write(f"**{model_name} Accuracy:** {accuracy:.2f}")
            accuracy_list.append((model_name, accuracy))

        # Plotting Model Accuracies
        st.subheader("Model Accuracy Comparison")
        df_accuracy = pd.DataFrame(accuracy_list, columns=["Model", "Accuracy"])
        sns.barplot(x="Model", y="Accuracy", data=df_accuracy)
        plt.title("Model Accuracy Comparison")
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting

        st.subheader("Which Resume is Better According to Each Model?")
        better_resume_list = []
        for model_name, better in better_resume.items():
            st.write(f"**Based on {model_name}:** {better}")
            better_resume_list.append((model_name, better))

        df_better_resume = pd.DataFrame(better_resume_list, columns=["Model", "Better Resume"])
        sns.countplot(x="Better Resume", hue="Model", data=df_better_resume)
        plt.title("Better Resume According to Each Model")
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting

        # Adding a Pie Chart
        st.subheader("Category Distribution in the Dataset")
        category_counts = df['Category'].value_counts()
        plt.figure(figsize=(12, 12))
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Category Distribution")
        st.pyplot(plt)
        plt.clf()  # Clear the figure after plotting

        # Displaying Confusion Matrices
        st.subheader("Confusion Matrices")
        for model_name, matrix in confusion_matrices.items():
            st.write(f"**{model_name} Confusion Matrix:**")
            plt.figure(figsize=(10, 7))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{model_name} Confusion Matrix")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(plt)
            plt.clf()  # Clear the figure after plotting