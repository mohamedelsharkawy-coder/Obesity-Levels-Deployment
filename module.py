import pandas as pd
import numpy as np
import pickle

# input data for testing
columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'high_cal_consumption', 'vegetable_consumption', 
           'meals_number', 'eating_between_meals', 'SMOKE', 'water_consumption', 'calorie_monitoring', 'physical_activity', 'screen_time', 'alcohole_consumption', 'transportation_mode']

input_data = pd.DataFrame([['Female', 22.0, 1.62, 58.0, 'no', 'yes', 2.0, 3.0, 'Sometimes',     
       'no', 1.0, 'no', 2.0, 0.0, 'Sometimes', 'Public_Transportation']], columns=columns)


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


predicted_class, predicted_class_proba = prediction(input_data)
print(predicted_class, predicted_class_proba)





