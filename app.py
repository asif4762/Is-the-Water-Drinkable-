import gradio as gr
import pandas as pd
import pickle
import numpy as np


with open('svc_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_potability(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    input_df = pd.DataFrame({
        'ph': [ph],
        'Hardness': [Hardness],
        'Solids': [Solids],
        'Chloramines': [Chloramines],
        'Sulfate': [Sulfate],
        'Conductivity': [Conductivity],
        'Organic_carbon': [Organic_carbon],
        'Trihalomethanes': [Trihalomethanes],
        'Turbidity': [Turbidity]
    })
    prediction = model.predict(input_df)[0]
    if prediction == 1: return 'The water is potable'
    else: return 'The water is not potable'

inputs = [
    gr.Number(label='ph'),
    gr.Number(label='Hardness'),
    gr.Number(label='Solids'),
    gr.Number(label='Chloramines'),
    gr.Number(label='Sulfate'),
    gr.Number(label='Conductivity'),
    gr.Number(label='Organic_carbon'),
    gr.Number(label='Trihalomethanes'),
    gr.Number(label='Turbidity')
]


app = gr.Interface(fn=predict_potability, inputs=inputs, outputs='text', title='Water Potability Prediction')

app.launch()
