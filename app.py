import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# Load Saved Files
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# MUST be the first Streamlit command
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load Saved Files
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right,#086E82,#243B55,#141E30);
    color:white;
}

h1,h2,h3,h4,label {
    color:white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 House Price Prediction")






# ==========================
# User Inputs
# ==========================

col1, col2 = st.columns(2)

with col1:
    square_ft = st.number_input("Square Feet",100,10000,1000)

    bhk = st.number_input("BHK",1,10,2)

    latitude = st.number_input(
        "Latitude",
        value=19.0760,
        format="%.6f"
    )

    address = st.selectbox("Address", ["Mumbai"])

    posted_by = st.selectbox(
        "Posted By",
        ["Owner","Dealer","Builder"]
    )
    resale = st.selectbox(
        "Resale",
        ["No","Yes"]
    )


with col2:
    longitude = st.slider(
        "Longitude",
        min_value=72.8777,
        max_value=77.8777,
        value=72.8777,
        format="%.6f"
    )

    bhk_or_rk = st.selectbox(
        "Property Type",
        ["BHK","RK"]
    )

    under_construction = st.selectbox(
        "Under Construction",
        ["No","Yes"]
    )

    ready_to_move = st.selectbox(
        "Ready To Move",
        ["Yes","No"]
    )

    rera = st.selectbox(
        "RERA Approved",
        ["Yes","No"]
    )

    
 
# ==========================
# Prediction
# ==========================

# ==========================
# Prediction
# ==========================


st.markdown("""
<style>
.stButton>button{
    background-color:#4CAF50;
    color:white;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    padding:10px 20px;
    width:100%;
}
.stButton>button:hover{
    background-color:#45a049;
}
</style>
""", unsafe_allow_html=True)




if st.button("🔍 Predict Price"):

    with st.spinner("Predicting House Price..."):

        # --------------------------
        # Log Transform
        # --------------------------
        square_ft_log = np.log1p(square_ft)
        bhk_log = np.log1p(bhk)

        # --------------------------
        # Target Encoding
        # --------------------------
        address_encoded = target_encoder.transform(
            pd.DataFrame({"ADDRESS": [address]})
        )["ADDRESS"].iloc[0]

        # --------------------------
        # One-Hot Encoding
        # --------------------------
        owner = 1 if posted_by == "Owner" else 0
        dealer = 1 if posted_by == "Dealer" else 0
        rk = 1 if bhk_or_rk == "RK" else 0

        under_construction_value = 1 if under_construction == "Yes" else 0
        ready_to_move_value = 1 if ready_to_move == "Yes" else 0
        rera_value = 1 if rera == "Yes" else 0
        resale_value = 1 if resale == "Yes" else 0

        # --------------------------
        # Create Input DataFrame
        # --------------------------
        input_df = pd.DataFrame({
            "UNDER_CONSTRUCTION": [under_construction_value],
            "RERA": [rera_value],
            "BHK_NO.": [bhk_log],
            "SQUARE_FT": [square_ft_log],
            "READY_TO_MOVE": [ready_to_move_value],
            "RESALE": [resale_value],
            "ADDRESS": [address_encoded],
            "LONGITUDE": [longitude],
            "LATITUDE": [latitude],
            "POSTED_BY_Dealer": [dealer],
            "POSTED_BY_Owner": [owner],
            "BHK_OR_RK_RK": [rk]
        })

        # --------------------------
        # Scale Numerical Features
        # --------------------------
        scale_cols = [
            "SQUARE_FT",
            "BHK_NO.",
            "LONGITUDE",
            "LATITUDE"
        ]

        input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        # --------------------------
        # Arrange Columns
        # --------------------------
        input_df = input_df[
            [
                "UNDER_CONSTRUCTION",
                "RERA",
                "BHK_NO.",
                "SQUARE_FT",
                "READY_TO_MOVE",
                "RESALE",
                "ADDRESS",
                "LONGITUDE",
                "LATITUDE",
                "POSTED_BY_Dealer",
                "POSTED_BY_Owner",
                "BHK_OR_RK_RK"
            ]
        ]

        # --------------------------
        # Predict
        # --------------------------
        prediction = model.predict(input_df)[0]

        # Reverse Log Transform
        predicted_price = np.expm1(prediction)

        
        st.subheader(f"🏡 Estimated House Price: ₹ {predicted_price:,.2f} Lakhs")
        # Show Processed Input
        with st.expander("Show Processed Input Data"):
            st.dataframe(input_df)



