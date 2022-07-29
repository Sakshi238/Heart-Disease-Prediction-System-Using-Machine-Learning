import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('heart_disease_model.sav','rb'))

#creating a function for prediction
def heart_prediction(input_data):
    input_data_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for an instance
    input_data_reshaped = input_data_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person does not have a Heart Disease'
    else:
        return 'The person have a Heart Disease'


def main():
    #Giving title
    st.title('Heart Disease Prediction')

    #getting input data from the user

    age = st.text_input('Enter your age:')
    gender = st.text_input('Enter your Gender: 1 for Female & 0 for Male')
    chest_pain = st.text_input('Enter your chest pain type:  , 0- Atypical angina, 1- Non-Anginal Pain, 2- Typical Angina,  3- Asymptomatic')
    rest_bp = st.text_input('Enter your Resting Blood Pressure in mm/Hg:')
    chol = st.text_input('Enter your Cholestrol level in mg/dl:')
    fbs = st.text_input('Enter your Fasting Blood Pressure(>120mg/dl): 1 - True, 0 - False :')
    restecg = st.text_input('Enter your resting Electrocardiographic results: 0 - Normal, 1- having ST-T, 2-having LVH :')
    MaxHR = st.text_input('Enter maximum heart rate achieved:')
    exang = st.text_input('Enter exercise induced angina: 0- yes,1 - no:')
    Oldpeak = st.text_input('Enter the old peak value:')
    ST_slope = st.text_input('Enter the ST_Slope: 0-UP , 1-Flat, 2-Down')


    #code for prediction
    diagnosis = ''

    if st.button('Heart Disease Test Result'):
        diagnosis = heart_prediction([age, gender, chest_pain, rest_bp, chol, fbs, restecg, MaxHR , exang, Oldpeak, ST_slope])

    st.success(diagnosis)


if __name__ == '_main_':
    main()