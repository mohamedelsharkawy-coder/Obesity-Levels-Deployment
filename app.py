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
st.title("🏋️‍♂️ Obesity Level Prediction App")
st.markdown("Fill in the following details to predict your obesity level.")

# Input fields
st.markdown("### **Age**")
age = st.number_input("", min_value=14, max_value=60, step=1)

st.markdown("### **Height (in meters)**")
height = st.number_input("", min_value=1.40, max_value=2.0, step=0.01)

st.markdown("### **Weight (in kilograms)**")
weight = st.number_input("", min_value=39.0, max_value=170.0, step=0.1)

st.markdown("### **Gender**")
gender = st.radio("", ['Male', 'Female'], horizontal=True)

st.markdown("### **Family History with Overweight**")
family_history = st.radio(" ", ['No', 'Yes'], horizontal=True)

st.markdown("### **High Calorie Consumption**")
high_cal = st.radio("  ", ['No', 'Yes'], horizontal=True)

st.markdown("### **Vegetable Consumption**")
vegetable = st.radio("", ['Low', 'Medium', 'High'], horizontal=True) # map from 0 to 2

st.markdown("### **Number of Meals per Day**")
meals = st.radio("", [1, 2, 3, 4], horizontal=True)

st.markdown("### **Eating Between Meals**")
eating_between = st.radio("", ['No', 'Sometimes', 'Frequently', 'Always'], horizontal=True)

st.markdown("### **Do You Smoke?**")
smoke = st.radio("      ", ['No', 'Yes'], horizontal=True)

st.markdown("### **Daily Water Consumption**")
water = st.radio(" ", ['Low', 'Medium', 'High'], horizontal=True) # map from 0 to 2

st.markdown("### **Do You Monitor Your Calorie Intake?**")
calorie_monitoring = st.radio("             ", ['No', 'Yes'], horizontal=True)

st.markdown("### **Physical Activity Level**")
physical_activity = st.radio("", ['Sedentary', 'Light Activity', 'Moderate Activity', 'High Activity'], horizontal=True) # map from 0 to 3

st.markdown("### **Daily Screen Time**")
screen_time = st.radio("          ", ['Low', 'Medium', 'High'], horizontal=True) # map to 0 to 2

st.markdown("### **Alcohol Consumption**")
alcohol = st.radio("", ['No', 'Sometimes', 'Frequently'], horizontal=True)

st.markdown("### **Primary Transportation Mode**")
transport = st.radio("", ['Walking', 'Bike', 'Motorbike', 'Automobile', 'Public_Transportation'], horizontal=True)

# Predict Button
if st.button("Predict Obesity Level"):
   
    # Validation check
    if any(field is None for field in [gender, family_history, high_cal, vegetable, meals, eating_between,
                                   smoke, water, calorie_monitoring, physical_activity, screen_time, alcohol, transport]):
        st.error("⚠️ Please fill out all fields before submitting.")
    else:
        # in case of vegetable
        if vegetable == 'Low': vegetable = 0
        if vegetable == 'Medium': vegetable = 1
        if vegetable == 'High': vegetable = 2

        # in case of water
        if water == 'Low': water = 0
        if water == 'Medium': water = 1
        if water == 'High': water = 2
        
        # in case of physical_activity
        if physical_activity == 'Sedentary': physical_activity = 0
        if physical_activity == 'Light Activity': physical_activity = 1
        if physical_activity == 'Moderate Activity': physical_activity = 2
        if physical_activity == 'High Activity': physical_activity = 3

        # in case of screen_time
        if screen_time == 'Low': screen_time = 0
        if screen_time == 'Medium': screen_time = 1
        if screen_time == 'High': screen_time = 2

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

        # Prediction
        predicted_class, predicted_class_proba = prediction(input_data)
        class_map = {0:"Unsufficient_Weight", 1:"Normal_Weight", 2:"Overweight_I", 3:"Overweight_II", 
                    4:"Obesity_I", 5:"Obesity_II", 6:"Obesity_III"}
        class_name = class_map[predicted_class]
        
        st.markdown(f"""
    <style>
        body {{
            background-color: black; /* Set the background to black */
            color: white; /* Set text color to white for contrast */
        }}
        table {{
            width: 90%; /* Make the table wider */
            border-collapse: collapse;
            font-size: 24px; /* Make text larger */
            margin: auto; /* Center the table */
        }}
        th, td {{
            padding: 20px; /* Increase padding for larger space */
            text-align: left;
            border: 1px solid #00796B; /* Add border with teal color */
        }}
        th {{
            background-color: #00796B; /* Teal color for headers */
            color: white; /* White text color */
        }}
        tr:nth-child(even) {{
            background-color: #000000; /* Light teal color for even rows */
        }}
        tr:nth-child(odd) {{
            background-color: #000000; /* White color for odd rows */
        }}
        tr:hover {{
            background-color: #b2dfdb; /* Light hover effect */
        }}
    </style>
    <table>
        <tr>
            <th>Prediction</th>
            <th>Confidence</th>
        </tr>
        <tr>
            <td>{class_name}</td>
            <td>{round(predicted_class_proba * 100, 2)}%</td>
        </tr>
    </table>
""", unsafe_allow_html=True)



        
        



