# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:18:49 2024

@author: ADMIN
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav','rb'))

def diabetes_prediction(input_data):
    input_data_as_array = np.asarray(input_data)
    reshaped_input_data = input_data_as_array.reshape(1,-1)
    result = loaded_model.predict(reshaped_input_data)

    if(result == 1):
      return 'The patient is Diabetic'
    else:
      return 'The patient is Not Diabetic'
  
    
  
def main():
    st.title('Diabetes Prediction')
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Level of Glucose')
    BloodPressure = st.text_input('Bp level')
    SkinThickness = st.text_input('Thickness of skin')
    Insulin = st.text_input('Insulin Level')
    BMI =  st.text_input('Body mass Idex value')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function')
    Age = st.text_input('Age of person')

    
    diagonosis = ''
    
    if st.button('Diabetes Test Result'):
        diagonosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagonosis)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
        
