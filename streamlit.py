from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import plotly as px
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from pipelinehelper import PipelineHelper
from sklearn.model_selection import train_test_split
import streamlit as st
st.set_page_config(layout="wide", page_title="Heart Attack Analysis & Prediction",
                   page_icon="❤️", initial_sidebar_state="collapsed")


st.write("""
        # Heart Attack Analysis & Prediction
        # Made by Muhammad Ali Butt & Team""")

st.sidebar.header("Input Parameters")

st.write(""" ## Heart attack Analysis dataset's keys definition

**1. Age** : Age of the patient

**2. Sex** : Sex of the patient

**3. cp** : Chest Pain type

    Value 0: Typical angina

    Value 1: Atypical angina

    Value 2: Non-anginal pain

    Value 3: Asymptomatic

**4. trtbps** : Blood pressure after receiving treatment (in mm Hg)

**5. chol**: Cholesterol in mg/dl fetched via BMI sensor

**6. fbs**: (Fasting blood sugar > 120 mg/dl)

    1 = true

    0 = false

**7. rest_ecg**: Resting electrocardiographic results
    Value 0: normal

    Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

    Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

**8. thalach**: Maximum heart rate achieved

**9.exang**: Exercise induced angina(discomfort du)

    1 = yes

    0 = no

**10. old peak**: ST depression induced by exercise relative to rest

**11. slp**: The slope of the peak exercise ST segment

    0 = Unsloping

    1 = flat

    2 = downsloping

**12. caa**: Number of major vessels (0-3)

**13. thall** : Thalassemia

    0 = null

    1 = fixed defect

    2 = normal

    3 = reversable defect

**14. output**: diagnosis of heart disease (angiographic disease status)

    0: < 50% diameter narrowing. less chance of heart disease

    1: > 50% diameter narrowing. more chance of heart disease""")


def user_input_features():
    min_value = 25
    max_value = 80
    min_value_1 = 90
    max_value_1 = 200
    min_value_2 = 125
    max_value_2 = 565
    min_value_3 = 70
    max_value_3 = 202
    min_value_4 = 0.0
    max_value_4 = 7.0
    min_value_5 = 0
    max_value_5 = 4
    min_value_6 = 0
    max_value_6 = 3
    age = st.sidebar.slider("age", min_value, max_value,
                            value=None, step=None, format=None)
    sex = st.sidebar.slider("sex", 0, 1)
    cp = st.sidebar.slider("Chest pain", 1, 2, 3)
    trtbps = st.sidebar.slider(
        "trtbps", min_value_1, max_value_1)  # , value=None, step=None, format=None)
    chol = st.sidebar.slider(
        "chol", min_value_2, max_value_2)
    fbs = st.sidebar.slider("fbs", 1, 2, 3)
    restecg = st.sidebar.slider("restecg", 0, 1)
    thalach = st.sidebar.slider(
        "thalach", min_value_3, max_value_3)
    exng = st.sidebar.slider("exng", 0, 1)
    oldpeak = st.sidebar.slider(
        "oldpeak", min_value_4, max_value_4)
    slp = st.sidebar.slider("slp", 0, 1, 2)
    caa = st.sidebar.slider("caa", min_value_5, max_value_5,)
    thall = st.sidebar.slider("thall", min_value_6,
                              max_value_6)

    data = {"age": age,
            "sex": sex,
            "cp": cp,
            "trtbps": trtbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exng": exng,
            "oldpeak": oldpeak,
            "slp": slp,
            "caa": caa,
            "thall": thall}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader("Heart Attack parameters")
st.write(df)

df1 = pd.read_csv("heart.csv")
# profile = ProfileReport(
#     df1, title="Heart Attack Analysis & Prediction", explorative=True)
# st_profile_report(profile)
st.subheader("Heart Attack dataset")
st.write(df1)


st.subheader("List of Columns")
st.write(df1.columns)

st.subheader("Heart Attack dataset's description")
st.write(df1.describe())

# unique vals retrievd and converted to list
gender_option = df1["sex"].unique().tolist()
sex = st.selectbox("which sex should we plot seaprately?",
                   gender_option, index=0, )  # index=0 is the default value
st.write('You selected the following sex', sex)
df = df[df["sex"] == sex]

if st.button('plot 0') and sex == 0:
    sns.set_style('white')
    plt.figure(figsize=(4, 3), facecolor='yellow')
    ax = plt.axes()
    ax.set_facecolor('white')
    sns.barplot(x='sex', y='age', data=df1[df1.sex == 0], color='red', saturation=.9,
                edgecolor='.3', linewidth=1
                )
    plt.title('Sex:0', fontsize=10, fontweight='bold', color='crimson')
    plt.xlabel('Sex', fontsize=6, fontweight='bold', color='red')
    plt.ylabel('Age', fontsize=6, fontweight='bold', color='red')
    st.pyplot(plt.gcf())
elif st.button('plot 1') and sex == 1:
    sns.barplot(x='sex', y='age', data=df1[df1.sex == 1], color='crimson',
                edgecolor='.3', linewidth=1
                )
    plt.title('Sex:1', fontsize=10, fontweight='bold', color='crimson')
    st.pyplot(plt.gcf())
else:
    st.write('Select sex and press plot to see the plot')


warnings.filterwarnings('ignore')

st.write("""
# Model Selection App
Select the best model by adjusting the hyperparameters!
""")

st.write("### Load Data")
df2 = pd.read_csv('heart.csv')
# df2 = df.drop_duplicates()
X = df2.drop(['output'], axis=1)
y = df2['output']


# st.write("### Train Model")
# selected_model = pipe.named_steps['classifier'].get(model_type)

# st.write("### Train Model")
# try:
#     selected_model = [model for model in pipe.named_steps['classifier']
#                       if model.__class__.__name__ == model_type][0]
#     selected_model.set_params(**params)
# except AttributeError as e:
#     st.write("An error occurred:", e)


# # ! best estimator for our model ye best_params_ krnay se bhi pata chal hi jata ha
# st.write(grid.best_estimator_)
# grid = GridSearchCV(pipe, params, scoring='accuracy')
# grid.fit(X, y)
# y_pred = grid.predict(X_test)
# st.write("### Model Score")
# st.write(grid.best_score_)
# st.write("### Model Parameters")
# st.write(grid.best_params_)
# st.write("### Model Estimator")
# st.write(grid.best_estimator_)
#! ALT WAY
# st.write("### Select Model Type")
# model_type = st.selectbox("Select a model type:",
#                           ['svm', 'rf', 'knn', 'dt', 'lr'])
# st.write("You selected ", model_type)

# st.write("### Adjust Hyperparameters")

# classifiers = {
#     'svm': LinearSVC(),
#     'rf': RandomForestClassifier(),
#     'knn': KNeighborsClassifier(),
#     'dt': DecisionTreeClassifier(),
#     'lr': LogisticRegression()
# }


# params = {}
# if model_type == 'svm':
#     params['svm__C'] = [st.slider('C', 0.1, 1.0, 0.5)]
#     params['loss'] = [st.selectbox(
#         'loss', ['hinge', 'squared_hinge'])]
#     params['penalty'] = [st.selectbox('penalty', ['l1', 'l2'])]
# elif model_type == 'rf':
#     params['n_estimators'] = [st.slider('n_estimators', 100, 20, 100)]
#     params['criterion'] = [st.selectbox(
#         'criterion', ['gini', 'entropy'])]
#     params['max_depth'] = [st.slider('max_depth', 3, 100, 3)]
# elif model_type == 'knn':
#     params['n_neighbors'] = [st.slider('n_neighbors', 3, 100, 3)]
#     params['weights'] = [st.selectbox(
#         'weights', ['uniform', 'distance'])]
#     params['algorithm'] = [st.selectbox(
#         'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])]
# elif model_type == 'dt':
#     params['max_depth'] = [st.slider('max_depth', 3, 100, 3)]
#     params['criterion'] = [st.selectbox(
#         'criterion', ['gini', 'entropy'])]
#     params['splitter'] = [st.selectbox('splitter', ['best', 'random'])]
# elif model_type == 'lr':
#     params['fit_intercept'] = [st.selectbox(
#         'fit_intercept', [True, False])]
#     params['solver'] = [st.selectbox(
#         'solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])]
#     params['max_iter'] = [st.slider('max_iter', 100, 5000, 100)]

# pipe = Pipeline([
#     ('classifier', classifiers[model_type])
# ])

# grid = GridSearchCV(pipe, params, scoring='accuracy')
# grid.fit(X, y)
# y_pred = grid.predict(X_test)
# st.write(grid.best_params_)
# st.write(grid.best_score_)
# st.write(grid.best_estimator_)
