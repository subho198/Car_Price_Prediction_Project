import streamlit as st
import sklearn
import pandas as pd
import numpy as np
import pickle

model=pickle.load(open('random_model.pkl','rb'))
ndf=pickle.load(open('df.pkl','rb'))

st.title('Car Price Predictor App')

col1,col2=st.columns(2)

with col1:
    name=st.selectbox('Select the model name of the car : ',sorted(ndf['name'].unique()))

with col2:
    company=st.selectbox('Select the model name of the car : ',sorted(ndf['company'].unique()))

col3,col4=st.columns(2)

with col3:
    year=st.selectbox('Select the Year car purchased: ',sorted(ndf['year'].unique()))

with col4:
    fuel_type=st.selectbox('Select the model name of the car : ',sorted(ndf['fuel_type'].unique()))

col5,col6=st.columns(2)

with col5:
    kms_driven=st.number_input('Enter the distance travelled by the Car(Km) : ')

if st.button("Predict the Price of the Car"):
    input_df=pd.DataFrame({'name':[name],'company':[company],'year':[year],'kms_driven':[kms_driven],'fuel_type':[fuel_type]})
    st.table(input_df)
    st.title(model.predict(input_df))


