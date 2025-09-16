#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle


# In[4]:


# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')


# In[5]:


# Read the CSV file into a pandas DataFrame
df = pd.read_csv("C:\\Users\\gulig\\Desktop\\Projects2k25\\resume1\\resume_dataset.csv")


# In[6]:


# Print the first five rows of the DataFrame
df.head()


# In[7]:


# Define a function to preprocess text
def preprocess_text(text):
    # Create a spaCy document object
    doc = nlp(text)

    # Clean the text by removing stop words, punctuation, and non-alphabetic characters
    clean = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            clean.append(token.lemma_.lower())
    # Return the cleaned text as a string
    return " ".join(clean)


# In[8]:


# Apply the preprocess_text function to the Resume column and store the result in a new column
df['Resume_text'] = df['Resume'].apply(preprocess_text)


# In[9]:


# Print the DataFrame with the new column
df


# In[10]:


# Create a LabelEncoder object
label_encoder = LabelEncoder()


# In[11]:


# Encode the Category column and store the result in a new column
df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])


# In[12]:


# Print the first five rows of the DataFrame
df.head()


# In[13]:


# Print the value counts of the Category and Category_Encoded columns
df[['Category', 'Category_Encoded']].value_counts()


# In[14]:


# Print the value counts of the Category_Encoded column
df.Category_Encoded.value_counts()


# In[15]:


# Print the first element of the Resume column
df.Resume[0]


# In[16]:


# Print the first element of the Resume_text column
df.Resume_text[0]


# In[17]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Resume_text'], df['Category_Encoded'], test_size=0.20, random_state=2, stratify=df['Category'])


# In[18]:


# Create a pipeline consisting of a TfidfVectorizer and a KNeighborsClassifier
model =Pipeline([('vectorizer', TfidfVectorizer()), ('model', KNeighborsClassifier())])


# In[19]:


# Fit the model on the training data
model.fit(X_train, y_train)


# In[20]:


# Predict the labels of the test data
y_pred = model.predict(X_test)


# In[21]:


# Print the classification report
print(classification_report(y_test, y_pred))


# In[22]:


# Compute and display the confusion matrix
cm_matrix = confusion_matrix(y_test, y_pred)

ax = plt.subplot()
sns.heatmap(cm_matrix, annot=True, ax=ax)

ax.set_xlabel('Predicted labels', fontsize=12)
ax.set_ylabel('True labels', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14)

plt.show()


# In[23]:


# Print the accuracy score
print(accuracy_score(y_test,y_pred))


# In[ ]:





# In[114]:


# # Save the model to a file
# with open('model_res.pkl', 'wb') as f:
#     pickle.dump(model, f)


# In[ ]:




