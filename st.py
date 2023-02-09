import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pipelinehelper import PipelineHelper

pipe = Pipeline([
    ('classifier', PipelineHelper([
        ('svm', LinearSVC()),
        ('rf', RandomForestClassifier()),
        ('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier()),
        ('lr', LogisticRegression())
    ])),
])

selected_model = st.selectbox(
    'Select a model', ['svm', 'rf', 'knn', 'dt', 'lr'])

params = {}
if selected_model == 'svm':
    params['classifier__selected_model'] = pipe.named_steps['classifier'].generate({
        'svm__C': [0.1, 1.0],
        'svm__loss': ['hinge', 'squared_hinge'],
        'svm__penalty': ['l1', 'l2'],
    })
    C = st.slider('C', 0.1, 1.0, step=0.1)
    loss = st.selectbox('loss', ['hinge', 'squared_hinge'])
    penalty = st.selectbox('penalty', ['l1', 'l2'])
    params['classifier__selected_model__svm__C'] = C
    params['classifier__selected_model__svm__loss'] = loss
    params['classifier__selected_model__svm__penalty'] = penalty

elif selected_model == 'rf':
    params['classifier__selected_model'] = pipe.named_steps['classifier'].generate({
        'rf__n_estimators': [100, 20],
        'rf__criterion': ['gini', 'entropy'],
        'rf__max_depth': [3, 5, 10, 20, 50, 100],
    })
    n_estimators = st.slider('n_estimators', 100, 20, step=1)
    criterion = st.selectbox('criterion', ['gini', 'entropy'])
    max_depth = st.slider('max_depth', 3, 100, step=1)
    params['classifier__selected_model__rf__n_estimators'] = n_estimators
    params['classifier__selected_model__rf__criterion'] = criterion
    params['classifier__selected_model__rf__max_depth'] = max_depth

elif selected_model == 'knn':
    params['classifier__selected_model'] = pipe.named_steps['classifier'].generate({
        'knn__n_neighbors': [3, 5, 10, 20, 50, 100],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    })
    n_neighbors = st.slider('n_neighbors', 3, 100, step=1)
    weights = st.selectbox('weights', ['uniform', 'distance'])
    algorithm = st.selectbox(
        'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    params['classifier__selected_model__knn__n_neighbors'] = n_neighbors
    params['classifier__selected_model__knn__weights'] = weights
    params['classifier__selected_model__knn__algorithm'] = algorithm

elif selected_model == 'dt':
    params['classifier__selected_model'] = pipe.named_steps['classifier'].generate({
        'dt__criterion': ['gini', 'entropy'],
        'dt__max_depth': [3, 5, 10, 20, 50, 100],
    })
    criterion = st.selectbox('criterion', ['gini', 'entropy'])
    max_depth = st.slider('max_depth', 3, 100, step=1)
    params['classifier__selected_model__dt__criterion'] = criterion
    params['classifier__selected_model__dt__max_depth'] = max_depth

elif selected_model == 'lr':
    params['classifier__selected_model'] = pipe.named_steps['classifier'].generate({
        'lr__C': [0.1, 1.0],
        'lr__penalty': ['l1', 'l2'],
        'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    })
    C = st.slider('C', 0.1, 1.0, step=0.1)
    penalty = st.selectbox('penalty', ['l1', 'l2'])
    params['classifier__selected_model__lr__C'] = C
    params['classifier__selected_model__lr__penalty'] = penalty


params = {
    'classifier__selected_model': pipe.named_steps['classifier'].generate({
        'svm__C': [0.1, 1.0],
        'svm__loss': ['hinge', 'squared_hinge'],
        'svm__penalty': ['l1', 'l2'],
        'rf__n_estimators': [100, 20],
        'rf__criterion': ['gini', 'entropy'],
        'rf__max_depth': [3, 5, 10, 20, 50, 100],
        'knn__n_neighbors': [3, 5, 10, 20, 50, 100],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'dt__max_depth': [3, 5, 10, 20, 50, 100],
        'dt__criterion': ['gini', 'entropy'],
        'dt__splitter': ['best', 'random'],
        'lr__fit_intercept': [True, False],
        'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'lr__max_iter': [100, 1000, 2500, 5000],
    })
}
