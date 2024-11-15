#importing libraries
import pickle
import pandas as pd
import streamlit as slt
import numpy as np

# page setting
if 'page' not in slt.session_state:
    slt.session_state.page='page1'

if slt.session_state.page=='page1':
    slt.set_page_config(layout="wide")
    slt.image("E:\CarDekho-Feature.jpg",use_column_width=True)
    slt.markdown("""
        <style>
        .stButton>button {
            padding: 25px 90px;  
            font-size: 60px;     
            border-radius: 30px; 
            background-color: red; 
            color: white;        
        }

        .stButton {
            display: flex;
            justify-content: center;  
            align-items: center;      
            height: 100%;  
        }
        </style>
        """, unsafe_allow_html=True)
    if slt.button('Sell My Car'):
        slt.session_state.page='page2'

elif slt.session_state.page=='page2':

    #slt.set_page_config(layout="wide")
    slt.header(':blue[CARDHEKO-USED CAR PRICE PREDICTION 🚗]')
    # Load data
    df = pd.read_csv("final_data.csv")
    print(df.columns)
    
    # Streamlit interface

    col1, col2 = slt.columns(2)
    with col1:
        Brand = slt.selectbox("Brand", options=df['Brand'].unique())  # Select brand
        
        # Filter the dataframe based on the selected brand
        filtered_models = df[df['Brand'] == Brand]['Model'].unique()
        # Ensure we are getting only the relevant models for the selected brand
        Model = slt.selectbox("Model", options=filtered_models)
        
        # Other selectboxes remain the same
        filtered_years = df[df['Model'] == Model]['Model Year'].unique()
        Model_year = slt.selectbox("Model Year", options=filtered_years)
        
        filtered_ft = df[df['Model'] == Model]['Fuel Type'].unique()
        Ft = slt.radio("Fuel Type",options=filtered_ft)

        filtered_bt = df[df['Model'] == Model]['Body Type'].unique()
        Bt = slt.selectbox("Body Type", options=filtered_bt)
        
        filtered_tr = df[df['Model'] == Model]['Transmission'].unique()
        Tr = slt.selectbox("Transmission",options=filtered_tr)
        
    
        # Now dynamically update the "Model" selectbox options based on the selected brand
        filtered_models = df[(df['Brand'] == Brand) & (df['Body Type'] == Bt) & (df['Fuel Type'] == Ft)]['Model'].unique()

        Owner = slt.selectbox("Owner", [0, 1, 2, 3, 4, 5])
   
        Km = slt.slider("Kilometers Driven", min_value=100, max_value=100000, step=1000)
        ML = slt.number_input("Mileage", min_value=5, max_value=50, step=1)

        filtered_s = df[df['Model'] == Model]['Seats'].unique()
        seats = slt.selectbox("Seats", options=filtered_s)

        filtered_c = df[df['Model'] == Model]['Color'].unique()
        color = slt.selectbox("Color",options=filtered_c)
        
        city = slt.selectbox("City", options=df['City'].unique())
        IV = slt.selectbox("Insurance Validity", ['Third Party insurance', 'Comprehensive', 'Third Party', 'Zero Dep', '2', '1', 'Not Available'])
        
    with col2:
        Submit = slt.button("Predict")
        if Submit:
            # load the model,scaler and encoder
            with open('pipeline.pkl','rb') as files:
                pipeline=pickle.load(files)
                

            # input data
            new_df=pd.DataFrame({
                'Brand':Brand,
                'Model Year':Model_year,
                "Model":Model,
                'Fuel Type': Ft,
                'Owner No':Owner,
                'Body Type':Bt,
                'Transmission':Tr,
                'Insurance Validity':IV,
                'Kms Driven':Km,
                'Mileage':ML,
                'Seats':seats,
                'Color':color,
                'City': city},index=[0])
        
            
            # Final Model Prediction
            prediction = pipeline.predict(new_df)
            price = round(prediction[0], 2)

            # Inline styling in Markdown
            slt.markdown(f"<p style='font-size: 40px; color:#FF7F50 ;'>The price of the {new_df['Brand'].iloc[0]} car is   :    {price} lakhs</p>", 
                         unsafe_allow_html=True)

          

