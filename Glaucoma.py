# %%

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv("C:/Users/91949/Desktop/ml files/glaucoma.csv")

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.columns

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.describe()

# %%
df.nunique()

# %%
df.columns

# %%
df = df.drop(['Patient ID'], axis = 1)

# %%
def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []

    for column in df.columns:
        if df[column].dtype in ['object','bool']:
            if df[column].nunique() < 30:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']:
            if df[column].nunique() < 30:
                discrete_features.append(column)
            else:
                continuous_features.append(column)

    return categorical_features, non_categorical_features, discrete_features, continuous_features

# %%
categorical, non_categorical, discrete, continuous = classify_features(df)

# %%
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)

# %%
df=df.drop(['Medication Usage', 'Visual Field Test Results', 'Optical Coherence Tomography (OCT) Results', 'Visual Symptoms'],axis=1)

# %%
df

# %%
df=df.drop_duplicates()

# %%
df.isnull().sum()

# %%
df=df.drop(['Medical History'],axis=1)

# %%
df

# %%
categorical, non_categorical, discrete, continuous = classify_features(df)

# %%
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)

# %%
df=pd.get_dummies(df,columns=['Gender', 'Visual Acuity Measurements', 'Family History', 'Cataract Status', 'Angle Closure Status', 'Diagnosis'],drop_first=False)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 'Pachymetry']]=scaler.fit_transform(df[['Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 'Pachymetry']])

# %%
df['Glaucoma Type'].value_counts()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

X = df.drop('Glaucoma Type', axis=1)
y = df['Glaucoma Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Before Oversampling:", y_train.value_counts())

# %%
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

print("After Oversampling:", y_train_res.value_counts())

# %%
log_reg_model = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg_model.fit(X_train_res, y_train_res)

# %%
y_pred = log_reg_model.predict(X_test)

# %%
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
from sklearn import svm

# %%
svc = svm.SVC(kernel = 'linear')

# %%
svc.fit(X_train, y_train)

# %%
y_pred = svc.predict(X_test)

# %%
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
import pickle
with open('glaucoma_model.pkl', 'wb') as file:
    pickle.dump(glau_reg, file)
print("Model saved as glaucoma_model.pkl")




