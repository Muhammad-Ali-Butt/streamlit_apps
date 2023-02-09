# import libraries

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import plotly as px
from sklearn.decomposition import PCA
from pipelinehelper import PipelineHelper
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
from PIL import Image
import base64
import streamlit as st
from streamlit_option_menu import option_menu
# # Insert Pictures
# st.image("th.jpg")
# st.image("th2.jpg")
st.set_page_config(
    layout="wide", page_title="Kyu k ye Dil ka Maamla ha", initial_sidebar_state="collapsed", page_icon="Favicon Preview 32x32")

# Insert A Navbar
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Data Download', 'About'],
        icons=['file-arrow-down-fill', 'book'],
        menu_icon='hourglass-split',
        # default_index=0,
    )
if selected == 'Data Download':
    st.markdown("""
            1. Click the link below and right click raw button and tap **Save link as**.
            2. Select location and change the extention of file to .csv if the file is being downloaded as .txt.
            3. Upload the file in the sidebar to view pandas profiling report.
            [click here to download the heart attack dataset](https://github.com/ayazparhyar/streamlit_pkc_app/blob/main/heart.csv)"""
                )
elif selected == 'About':
    st.write("""
            <h3 style=" color:red;"> Made with ❤️ by Qadir Shahbaz, Muhammad Ayaz, Muhammad Ali Farrukh & Muhammad Ali Butt&#128640 &#128075 &#127825</h3>
            <strong>CopyRight © 2023 PKC Team1 All Rights Reserved.</strong> """, unsafe_allow_html=True)

col1, col2 = st.columns(2)
# Insert sidebar to upload dataset to the streamlit app CSV format
with col1:
    st.write('''# Heart Attack Analysis & Prediction :ambulance:''')
with col2:
    # """### from local file"""
    file_ = open("heartattack.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" width="300" height="200" alt="heart attack.gif">',
        unsafe_allow_html=True,
    )

st.markdown("""# **01. Exploratory Data Analysis**""")
with st.sidebar.header("Upload your dataset(.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv"])
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header("**Input DF**")
    st.write(df)
    st.write("---")
    st.header("**Profiling report with pandas**")
    st_profile_report(pr)
else:
    st.info("Awaiting for CSV file")
    if st.button("press to use example data"):
        def load_data():
            a = pd.DataFrame(np.random.rand(100, 5),
                             columns=["age", "banana", "Codenics", "duck", "Ear"])
            return a
        df1 = load_data()
        pr = ProfileReport(df, explorative=True)
        st.Header("**Input DF**")
        st.write(df)
        st.write("---")
        st.header("**Profiling report with pandas**")
        st_profile_report(pr)

# Insert User Input Parameters
st.sidebar.header("Patient Data")


def user_report():
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

    sex_values = ("0", "1")
    cp_values = ("0", "1", "2", "3")
    age = st.sidebar.slider("age", min_value, max_value,
                            value=None, step=None, format=None)
    sex = st.sidebar.selectbox("sex", sex_values)
    cp = st.sidebar.selectbox("Chest pain", cp_values)
    trtbps = st.sidebar.slider(
        "trtbps", min_value_1, max_value_1, value=None, step=None, format=None)
    chol = st.sidebar.slider(
        "chol", min_value_2, max_value_2, value=None, step=None, format=None)
    fbs = st.sidebar.selectbox("fbs", ("1", "2", "3"))
    restecg = st.sidebar.selectbox("restecg", ("0", "1"))
    thalachh = st.sidebar.slider(
        "thalachh", min_value_3, max_value_3, value=None, step=None, format=None)
    exng = st.sidebar.selectbox("exng", ("0", "1"))
    oldpeak = st.sidebar.slider(
        "oldpeak", min_value_4, max_value_4, value=None, step=None, format=None)
    slp = st.sidebar.selectbox("slp", ("0", "1", "2"))
    caa = st.sidebar.selectbox("caa", ("0", "1", "2", "3", "4"))
    thall = st.sidebar.selectbox("thall", ("0", "1", "2", "3"))

    user_report_data = {"age": age,
                        "sex": sex,
                        "cp": cp,
                        "trtbps": trtbps,
                        "chol": chol,
                        "fbs": fbs,
                        "restecg": restecg,
                        "thalachh": thalachh,
                        "exng": exng,
                        "oldpeak": oldpeak,
                        "slp": slp,
                        "caa": caa,
                        "thall": thall}
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


# Description of the Columns
st.write(""" ## Heart attack Analysis dataset's keys definition
         
**1. Age** : Age of the patient

**2. Sex** : Sex of the patient

    Value 0: Female
    
    Value 1: Male

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

# To upload the file
df1 = pd.read_csv("heart.csv")

# Inserting another subheading
st.subheader("Heart Attack dataset")
st.write(df1)

# Performing EDA
st.subheader("List of Columns")
st.write(df1.columns)
st.subheader("Heart Attack dataset's description")
st.write(df1.describe().T)

# Removing duplicate value from dataset
df1.drop_duplicates(inplace=True)

# Patient data
st.title("Heart Attack prediction")
user_data = user_report()
st.subheader("Patient Data")

warnings.filterwarnings('ignore')

st.write("""
# Model Selection App
Select the best model by adjusting the hyperparameters!
""")

selected_model = st.selectbox(
    'Select a model', ['rf', 'knn', 'dt', 'lr'])


def get_params(selected_model):
    params = dict()
    if selected_model == 'rf':

        n_estimators = st.slider('n_estimators', 100, 20, step=1)
        criterion = st.selectbox('criterion', ['gini', 'entropy'])
        max_depth = st.slider('max_depth', 3, 100, step=1)
        params['n_estimators'] = n_estimators
        params['criterion'] = criterion
        params['max_depth'] = max_depth

    elif selected_model == 'knn':
        n_neighbors = st.slider('n_neighbors', 3, 100, step=1)
        weights = st.selectbox('weights', ['uniform', 'distance'])
        algorithm = st.selectbox(
            'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        params['n_neighbors'] = n_neighbors
        params['weights'] = weights
        params['algorithm'] = algorithm

    elif selected_model == 'dt':
        criterion = st.selectbox('criterion', ['gini', 'entropy'])
        max_depth = st.slider('max_depth', 3, 100, step=1)
        params['criterion'] = criterion
        params['max_depth'] = max_depth

    elif selected_model == 'lr':
        C = st.slider('C', 0.01, 5.0, step=0.01)
        max_iter = st.slider('max_iter', 100, 5000, step=1)
        fit_intercept = st.selectbox('fit_intercept', [True, False])
        solver = st.selectbox(
            'solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        params['C'] = C
        params['max_iter'] = max_iter
        params['fit_intercept'] = fit_intercept
        params['solver'] = solver
    return params


params = get_params(selected_model)


def get_classifier(selected_model, params):
    classifier = None
    if selected_model == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=params['n_estimators'], criterion=params['criterion'], max_depth=params['max_depth'], random_state=1234)
    elif selected_model == 'knn':
        classifier = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'], weights=params['weights'], algorithm=params['algorithm'])
    elif selected_model == 'dt':
        classifier = DecisionTreeClassifier(
            criterion=params['criterion'], max_depth=params['max_depth'], random_state=1234)
    elif selected_model == 'lr':
        classifier = LogisticRegression(max_iter=params['max_iter'], fit_intercept=params['fit_intercept'],
                                        C=params['C'], solver=params['solver'], random_state=1234)
    return classifier


model = get_classifier(selected_model, params)

# data splitting into X and y and Train test split
X = df1.drop(["output"], axis=1)
y = df1["output"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)
model.fit(X_train, y_train)
user_result = model.predict(user_data)
st.write('Prediction', user_result)

acc = accuracy_score(y_test, model.predict(X_test))
st.write(f'Model = {selected_model}')
st.write(f'Parameters =', params)
st.write(f'Accuracy =', acc)

# output
st.header("Your Report:")
output = ''
if user_result[0] == 0:
    output = 'You are safe'
    st.balloons()
elif user_result[0] == 1:
    output = "Look after your health"
    # st.warning("Attack detected", "Please take necessary precautions.")
st.title(output)


# Visualizing the dataset

pca = PCA(2)
X_projected = pca.fit_transform(X)

# first principal component of the data 0 means first column
x1 = X_projected[:, 0]
# second principal component of the data 1 means second column
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(df1['sex'], df1['age'], c=df1['output'], alpha=0.8,
            cmap='autumn', edgecolors='crimson', s=60)
plt.xlabel('Sex', fontsize=15, color='red', fontweight='bold')
plt.ylabel('Age', fontsize=15, color='red', fontweight='bold')
plt.colorbar()
st.pyplot(fig)
# Age vs Chol
st.header("Age vs Cholesterol")
fig_preg = plt.figure()
dimension1 = sns.scatterplot(
    x="age", y="chol", data=df1, hue="output", palette="autumn_r", s=60)
dimension2 = sns.scatterplot(
    x=user_data["age"], y=user_data["chol"], s=60)
plt.xticks(ha='right', rotation=45, fontsize=8)
plt.yticks(ha='right', rotation=45, fontsize=8)
plt.title("0 - Healty & 1 Possiblity of Heart attach",
          fontsize=15, color='crimson', fontweight='bold')
st.pyplot(fig_preg)
