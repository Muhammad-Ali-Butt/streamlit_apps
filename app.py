import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# ? Make containers
header = st.container()
data_sets = st.container()
featuress = st.container()
model_training = st.container()

with header:
    st.title('Kashti ki app')
    st.text('In this project we will work on kashti data')

with data_sets:
    st.header('Kashti doob gai')
    st.text('We will work with titanic data set')
    # ? Import data
    df = sns.load_dataset('titanic')
    df = df.dropna()  # filling missing values
    st.write(df.head(10))

    st.subheader('Barchart of sexes')
    st.bar_chart(df['sex'].value_counts())

    # ? Other plot
    st.subheader('Barchart of classes')
    st.bar_chart(df['class'].value_counts())

    # ? barplot
    st.bar_chart(df['age'].sample(10))

with featuress:
    st.header('These are our app Features:')
    st.text('We add features here')

    st.markdown('1. **Feature 1:** This will tell us some ting')
    st.markdown('2. **Feature 2:** This will tell us some ting')

with model_training:
    st.header('Kashti waalo ka kia bana')
    st.text('In this project we can train our model')
    # ? making columns
    input, display = st.columns(2)
    # ? In 1st column we add selection points
    max_depth = input.slider(
        'How many people do you know', min_value=10, max_value=100, value=20, step=5)

# ? n_estimators
n_estimators = input.selectbox('How many Trees should be in RF', options=[
    50, 100, 200, 300, 'NO LIMIT'])

# Adding list of features
input.write(df.columns)

# Input features from user
infe = input.text_input('Which feature we should use')

# Machine learning model
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
#! If user select NO LIMIT then
if n_estimators == 'NO LIMIT':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(
        max_depth=max_depth, n_estimators=n_estimators)

# define X and y
X = df[[infe]]
y = df[['fare']]

# Fit model
model.fit(X, y)
pred = model.predict(y)

# Display metrics
display.subheader('Mean absolute error of the model is:')
display.write(mean_absolute_error(y, pred))
display.subheader('Mean squared error of the model is:')
display.write(mean_squared_error(y, pred))
display.subheader('R2 squared of the model is:')
display.write(r2_score(y, pred))


#! input hamesha sliders mai lena
