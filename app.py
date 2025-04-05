import streamlit as st
import pandas as pd
import numpy as np
import pickle

############################## functions ###################################

# load mapping that map between bins and labels for each column
with open('mapping.pkl', 'rb') as file:
    mapping = pickle.load(file)

# funciton that make equal-width-binning on the scaling columns
def equal_width_binning(data:pd.DataFrame, col_name:str, bins:int, labels:list, object:int):
    data = data.copy()
        
    # convert the column from categorical to object datatype
    if object == 1:
        data[col_name], edges = pd.cut(data[col_name], bins=bins, labels=labels, retbins=True)
        data[col_name] = data[col_name].astype('object')
        return data[col_name], edges
    
    # convert the column from categorical to integer datatype
    elif object == 0:
        data[col_name], edges = pd.cut(data[col_name], bins=bins, labels=labels, retbins=True)
        data[col_name] = data[col_name].astype('int32')
        return data[col_name], edges

def preprocess_data(data:pd.DataFrame, train=0):
    global mapping
    
    data_preprocess = data.copy()

    # Numeric Columns ['Age', 'Height', 'Weight']
    data_preprocess['Log_Age'] = np.log1p(data_preprocess['Age'])

    # Age Column -> Round to the nearest integer and then convert the datatype to int32 
    data_preprocess['Age'] = data_preprocess['Age'].round().astype('int32')

    # Height Column -> Round to the nearest 2 floating points 
    data_preprocess['Height'] = data_preprocess['Height'].round(2)

    # Weight Column -> Round to the nearest 2 floating points
    data_preprocess['Weight'] = data_preprocess['Weight'].round(2)

    # will calculate the suitable binning and apply it on the train data
    if train == 1:
        mapping = {}

        # Scaling Columns [vegetable_consumption, meals_number, water_consumption, physical_activity, screen_time]
        # make equal width binning
        # Apply Function on [vegetable_consumption, screen_time, water_consumption]
        target_columns = ['vegetable_consumption', 'screen_time', 'water_consumption']
        for col in target_columns:
            labels = ['low', 'medium', 'high']
            data_preprocess[col], edges = equal_width_binning(data_preprocess, col, 3, labels, object=1)
            
            # update mapping list
            mapping[col] = [] 
            mapping[col].append(labels)
            mapping[col].append(edges)

        # Apply Function on physical_activity
        labels = ['Sedentary', 'Light Activity', 'Moderate Activity', 'High Activity']
        data_preprocess['physical_activity'], edges = equal_width_binning(data_preprocess, 'physical_activity', 4, labels, object=1)
        
        # update mapping list
        mapping['physical_activity'] = [] 
        mapping['physical_activity'].append(labels)
        mapping['physical_activity'].append(edges)

        # Apply Function on meals_number
        labels = [1, 2, 3, 4]
        data_preprocess['meals_number'], edges = equal_width_binning(data_preprocess, 'meals_number', 4, labels, object=0)
        
        # update mapping list
        mapping['meals_number'] = [] 
        mapping['meals_number'].append(labels)
        mapping['meals_number'].append(edges)

        return data_preprocess

    # we just want to apply the calculated values on the test data
    elif train == 0:
        columns = ['vegetable_consumption', 'screen_time', 'water_consumption', 'physical_activity', 'meals_number']
        for col in columns:
            labels = mapping[col][0]
            edges = mapping[col][1]
            data_preprocess[col] = pd.cut(data_preprocess[col], bins=edges, labels=labels)
            if col == 'meals_number':
                data_preprocess[col] = data_preprocess[col].astype('int32')
            else:
                data_preprocess[col] = data_preprocess[col].astype('object')
        
        return data_preprocess

# load preprocessor
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# apply pipeline
def pipeline(input_data):
    processed_data = preprocessor.transform(input_data)
    return processed_data

# load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def prediction(input_data):
    processed_data = pipeline(input_data)
    predicted_class = int(model.predict(processed_data)[0])
    max_prob_index = model.predict_proba(processed_data)[0].argmax()
    predicted_class_proba = model.predict_proba(processed_data)[0][max_prob_index]
    return predicted_class, predicted_class_proba

st.set_page_config(page_title="Obesity Level Predictor", layout="centered")
st.title("🧠 Obesity Level Prediction App")
st.markdown("Fill in the following details to predict your obesity level.")

# Input fields
gender = st.radio("Gender", ['Male', 'Female'])

age = st.number_input("Age", min_value=1, max_value=120, step=1)

height = st.number_input("Height (in meters)", min_value=0.5, max_value=2.5, step=0.01)

weight = st.number_input("Weight (in kilograms)", min_value=10.0, max_value=300.0, step=0.1)

family_history = st.radio("Family History with Overweight", ['yes', 'no'])

high_cal = st.radio("High Calorie Consumption", ['yes', 'no'])

vegetable = st.radio("Vegetable Consumption", [0, 1, 2])

meals = st.radio("Number of Meals per Day", [1, 2, 3, 4])

eating_between = st.radio("Eating Between Meals", ['no', 'sometimes', 'frequently', 'always'])

smoke = st.radio("Do You Smoke?", ['yes', 'no'])

water = st.radio("Daily Water Consumption", [0, 1, 2])

calorie_monitoring = st.radio("Do You Monitor Your Calorie Intake?", ['yes', 'no'])

physical_activity = st.radio("Physical Activity Level", [0, 1, 2, 3])

screen_time = st.radio("Daily Screen Time", [0, 1, 2])

alcohol = st.radio("Alcohol Consumption", ['no', 'sometimes', 'frequently'])

transport = st.radio("Primary Transportation Mode", ['Walking', 'Bike', 'Motorbike', 'Automobile', 'Public_Transportation'])

# Predict Button
if st.button("Predict Obesity Level"):
   
    # Validation check
    if any(field is None for field in [gender, family_history, high_cal, vegetable, meals, eating_between,
                                   smoke, water, calorie_monitoring, physical_activity, screen_time, alcohol, transport]):
        st.error("⚠️ Please fill out all fields before submitting.")
    else:
        # Build input DataFrame
        input_data = pd.DataFrame([[
            gender, age, height, weight, family_history, high_cal, vegetable,
            meals, eating_between, smoke, water, calorie_monitoring,
            physical_activity, screen_time, alcohol, transport
        ]], columns=[
            'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'high_cal_consumption', 'vegetable_consumption', 'meals_number',
            'eating_between_meals', 'SMOKE', 'water_consumption',
            'calorie_monitoring', 'physical_activity', 'screen_time',
            'alcohole_consumption', 'transportation_mode'
        ])

        st.markdown("### 🧾 Input Data")
        st.dataframe(input_data)

        # Prediction
        predicted_class, predicted_class_proba = prediction(input_data)
        
        st.write(predicted_class)
        st.write(predicted_class_proba)
        
        



