import streamlit as st
import numpy as np
import pickle

#Load saved model and scaler
model_knn = pickle.load(open("KNN_servival.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Titanic dataset prediction by Manas")
st.markdown("Provide the following details")


#Input fields
PassengerClass = st.selectbox("Passenger Class(1=1st, 2=2nd, 3=3rd)",[1,2,3])
Sex = st.selectbox("Gender",["Male","Female"])
Age = st.slider("Age", 2, 80, 18)
SibSp = st.number_input("Number of sibling", 0,8,2)
Parch = st.number_input("Number of parent", 0,6,2)
Fare = st.number_input("Fare", 0.0,600.0,50.0)
Embarked = st.selectbox("Embarked (C=0 ,Q=1, S=2)",[0,1,2])
FamilySize = st.number_input("Family Size", 1,11,5) 

Sex = 1 if Sex == "Male" else 0

#input_data = np.array([[PassengerClass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize]])

#Predict Button
if st.button("Predict"):
    input_data = np.array([[PassengerClass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize]])
    input_scaled = scaler.transform(input_data)
    prediction = model_knn.predict(input_scaled)

    #st.write("Input data (scaled):", input_scaled)
    #st.write("Prediction output:", prediction)

    if prediction[0] == 1:
        st.success("Survived")
    else:
        st.error("Did not Survived")  

