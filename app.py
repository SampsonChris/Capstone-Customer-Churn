# Importing relevant libraries
import gradio as gr
import pandas as pd
import os, pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# Defining a function to load exported ML toolkit                    
def load_saved_objects(filepath='ML_items'):
    "Function to load saved objects"

    with open(filepath, 'rb') as file:
        loaded_object = pickle.load(file)
    
    return loaded_object

# Loading the toolkit
loaded_toolkit = load_saved_objects('/Users/Admin/Desktop/Churn Capstone/ML_items')

# Instantiating the elements of the Machine Learning Toolkit
print('Instantiating')
randmodel = loaded_toolkit["model"]
le = loaded_toolkit["encoder"]   
scaler = loaded_toolkit["scaler"]

# Relevant data inputs
expected_inputs = ['TENURE'	,'MONTANT','FREQUENCE_RECH','REVENUE','ARPU_SEGMENT','FREQUENCE','DATA_VOLUME','REGULARITY','FREQ_TOP_PACK'	]

cat_cols = ['TENURE']

num_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE','DATA_VOLUME','REGULARITY', 'FREQ_TOP_PACK']

# Defining the predict function
def predict(*args, encoder = le, scaler = scaler, model = randmodel):
    
    # Creating a dataframe of inputs
    Input_data = pd.DataFrame([args], columns= expected_inputs)

# Encoding    
    print("Encoding")
    #Input_data[categoricals].to_csv('categorical data.csv', index= False)
    Input_data['TENURE'] = le.fit_transform(Input_data['TENURE'])

#Scaling
    print("scaling")
    #Final_input[columns_to_scale].to_csv('columns to scale data.csv', index= False)
    Input_data[num_cols] = scaler.fit_transform(Input_data[num_cols])

# Modeling
    model_output = model.predict(Input_data)
    return float(model_output[0])    

# Working on inputs 
with gr.Blocks() as demo:

        gr.Markdown("# Classification Model that Predicts Customer Churn")

        gr.Markdown("Select Inputs fron the fields below")

        with gr.Row():
            TENURE = gr.Dropdown(['K > 24 month', 'E 6-9 month', 'H 15-18 month', 'G 12-15 month','I 18-21 month', 'J 21-24 month', 'F 9-12 month', 'D 3-6 month'], 
            label="Tenure", value = 'K > 24 month')
            MONTANT = gr.Slider(0, 10000,label = 'Top-up Amount')
            FREQUENCE_RECH = gr.Slider(0, 100,label = 'Refill Times')
            REVENUE = gr.Slider(0, 100000,label = 'Monthly Income')

        with gr.Column():    
            ARPU_SEGMENT = gr.Slider(0, 10000,label = 'Income Over 3 months')
            FREQUENCE = gr.Slider(0, 100,label = 'FREQUENCE')
            DATA_VOLUME = gr.Slider(0, 10000,label = 'Number of Connections')
            REGULARITY = gr.Slider(0, 100,label = 'Active Periods')
            FREQ_TOP_PACK = gr.Slider(0, 100,label = 'Top-Pack Activation Times')
        
           
        with gr.Row():
            btn = gr.Button("Predict").style(full_width=True)
            output = gr.Textbox(label="Classification Result") 
               
        btn.click(fn=predict,inputs=[TENURE,MONTANT,FREQUENCE_RECH,REVENUE,ARPU_SEGMENT,FREQUENCE,DATA_VOLUME,REGULARITY,FREQ_TOP_PACK	],outputs=output)

demo.launch(share= True, debug= True)      
