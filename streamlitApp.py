import streamlit as st
from joblib import load
import pandas as pd
import sklearn
import os
import time


# Data Loading: 
cwd = os.getcwd()
ip_filepath = cwd+'/WineQT.csv'
wq_df = pd.read_csv(ip_filepath)

# Data Transformation:
wq = wq_df.copy()
wq.drop('Id',axis=1, inplace=True)

def replace_spaces(df):
    return df.columns.str.replace(" ","_")
def title_name(df):
    return df.columns.str.title()

wq.columns = replace_spaces(wq)
wq.columns = title_name(wq)

wq.Quality = wq.Quality.apply(lambda x: 1 if x>= 6 else 0)

# Loading the model
ip_modelpath = cwd+"/RandomForestClassifier/wineclassification.joblib"
rfc = load(ip_modelpath)




nav = st.sidebar.radio("Goto",["Home",'Classify'])

if nav == "Home":
    
    st.title("Know the Data")
    
    if st.button("Display Dataset"):
        st.dataframe(wq, width=1000)

if nav == "Classify":
    
    st.title("Classification of the wine:")
    
    # Fixed_Acidity = 7.8000
    Fixed_Acidity = st.slider("what is the fixed acidity:",4.00,17.00, 6.70,0.05)
    
    #Volatile_Acidity = 0.8800
    Volatile_Acidity = st.slider("what is the volatile acidity:",
                                 0.00,2.00, 0.10,0.17)
    
    #Citric_Acid = 0.0000
    Citric_Acid = st.slider("what is the citric acid:",0.00,1.50, 0.00,0.05)
    
    #Residual_Sugar = 2.6000
    Residual_Sugar = st.slider("what is the residual sugar:",
                               0.50,16.00, 0.90,0.50)
    
    #Chlorides  = 0.0980
    Chlorides = st.slider("what is the chloride content:",0.00,0.80, 0.01,0.05)
    
    #Free_Sulfur_Dioxide = 25.0000
    Free_Sulfur_Dioxide = st.slider("Percentage of sulfur-dioxide free:",
                                    1.00,70.00, 2.00,1.00)
    
    #Total_Sulfur_Dioxide = 67.0000
    Total_Sulfur_Dioxide = st.slider("Total sulfur-dioxide content:",
                                    5.00,290.00, 6.00,2.00)
    
    #Density = 0.9968
    Density = st.slider("what is the density:",0.00,1.50, 0.50,0.10)
    
    #Ph = 3.2000
    Ph = st.slider("what is the pH level:",2.00,4.50, 2.50,0.10)
    
    #Sulphates = 0.6800
    Sulphates = st.slider("what is the sulphate content:",0.00,2.50, 0.30,0.15)
    
    #Alcohol = 9.8000
    Alcohol = st.slider("what is the alcohol content:",7.00,16.00, 8.00,0.50)
    
    test_data = [[Fixed_Acidity, Volatile_Acidity, Citric_Acid, Residual_Sugar,
           Chlorides, Free_Sulfur_Dioxide, Total_Sulfur_Dioxide, Density,
           Ph, Sulphates, Alcohol]]
    
    goodbad = rfc.predict(test_data)
    
    ref_dict = {
                    1 : "Good",
                    0 : "Bad"
               }
    
    #print(ref_dict[goodbad[0]])
    
    if st.button("Classify"):
        my_bar = st.progress(0)
        
        for percent_complete in range(100):
             time.sleep(0.01)
             my_bar.progress(percent_complete + 1)
             
        if goodbad[0] == 0:
            st.error(f"The wine quality is {ref_dict[goodbad[0]]}")
        else:
            st.success(f"The wine quality is {ref_dict[goodbad[0]]}")
        
