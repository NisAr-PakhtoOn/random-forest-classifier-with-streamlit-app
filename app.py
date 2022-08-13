import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import load_iris

#------------------------
# here we are going to set the page layout
st.set_page_config(page_title='Machine Learning App with Random Forest',layout='wide')

#-----------
# we will build the model here 
def build_model(df):
    df.dropna()
    X = df.iloc[:,:-1] # using all the colum except for the last column that is going to be predicted (Y)
    Y = df.iloc[:,-1] # using the last column

    # we will now split the data into test and train split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=split_size)

# Basic information printing

    st.markdown('**1.2 Data Splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Testing set')
    st.info(X_test.shape)

    st.markdown('**Variables details**')
    st.write('The X Variable')
    st.info(list[X.columns])
    st.write('The Y Variable')
    st.info(Y.name)

    # Model training (Random Forest)

    rf = RandomForestClassifier(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        # criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
# Fitting the data into the RD model
    rf.fit(X_train, Y_train)
    st.subheader('2. Model Performace')
    
    st.markdown('**2.1. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Classification Report:')
    st.info( metrics.classification_report(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

    #---------------------------------#
st.write("""
# The Machine Learning App

In this implementation, the *RandomForestClassifier()* function is used in this app for build a regression model using the **Random Forest** algorithm.

Try adjusting the hyperparameters!

By **NisAr PakhtoOn**

""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    # parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # irish dataset
        irish = load_iris()
        X = pd.DataFrame(irish.data, columns=irish.feature_names)
        Y = pd.Series(irish.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Diabetes dataset is used as the example.')
        st.write(df.head(5))


        build_model(df)





