import streamlit as st
import pandas as pd
import joblib
import numpy as np


# -----------------------------
# Page Config
# -----------------------------


model = joblib.load("house_price_model.pkl")

scaler = joblib.load("scaler.pkl")

target_encoder = joblib.load("target_encoder.pkl")




st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#EDE7F6,#D1C4E9,#B39DDB);
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Title */
.title{
text-align:center;
font-size:45px;
font-weight:bold;
color:#4A148C;
}

.subtitle{
text-align:center;
font-size:18px;
color:#555;
margin-bottom:20px;
}

/* Cards */
.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 10px rgba(0,0,0,.2);
margin-bottom:20px;
}

/* Section Heading */
.section{
background:#2E7D32;
padding:10px;
border-radius:8px;
color:white;
font-size:22px;
font-weight:bold;
text-align:center;
margin-top:15px;
margin-bottom:15px;
}

/* Predict Button */
.stButton>button{
width:100%;
background:#43A047;
color:white;
font-size:20px;
font-weight:bold;
border-radius:10px;
padding:12px;
}

.stButton>button:hover{
background:#2E7D32;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.image(r"E:\data science\capstone 1\background.png")
    

    st.markdown("## 🏠 House Price Predictor")

    st.info("""
Predict house prices using Machine Learning.

✔ Easy to use

✔ Fast Prediction

✔ AI Powered
""")

# -----------------------------
# Title
# -----------------------------
st.markdown(
"<div class='title'>🏠 House Price Prediction Dashboard</div>",
unsafe_allow_html=True
)

st.markdown(
"<div class='subtitle'>AI Powered Real Estate Price Estimator</div>",
unsafe_allow_html=True)
st.markdown("""
<style>

.stApp{
    background-color:#F3E8FF;
}

h1{
    color:#6A1B9A;
    text-align:center;
}

div[data-testid="stVerticalBlock"]{
    padding:10px;
}

.stButton>button{
    background-color:#6A1B9A;
    color:white;
    border-radius:8px;
    font-size:18px;
}

.stButton>button:hover{
    background-color:#4A148C;
}

</style>
""", unsafe_allow_html=True)
st.subheader("🏠 Property Details")

col1, col2 = st.columns(2)

with col1:
    square_ft = st.number_input("Square Feet",100,10000,1000)
    bhk = st.number_input("BHK",1,10,2)
    bhk_or_rk = st.selectbox("Property Type",["BHK","RK"])

with col2:
    posted_by = st.selectbox("Posted By",["Owner","Dealer","Builder"])
    resale = st.selectbox("Resale",["No","Yes"])
st.subheader("📍 Location")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude",value=19.0760)

with col2:
    longitude = st.number_input("Longitude",value=72.8777)

address = st.selectbox("Address",["Mumbai","Delhi","Bangalore","Hyderabad","Ahmedabad",
                                  "Chennai","Kolkata","Pune","Jaipur","Lucknow","Kanpur","Nagpur",
                                  "Indore","Thane","Bhopal","Visakhapatnam","Pimpri-Chinchwad",
                                  "Patna","Vadodara","Ghaziabad" ,"Ludhiana","Agra","Nashik","Faridabad",
                                  "Meerut","Rajkot","Kalyan-Dombivli","Vasai-Virar","Varanasi","Srinagar",
                                  "Aurangabad","Dhanbad","Amritsar","Navi Mumbai","Allahabad","Ranchi",
                                  "Howrah","Coimbatore","Jabalpur",  "Gwalior","Vijayawada","Jodhpur","Madurai",
                                  "Raipur","Kota","Chandigarh","Guwahati","Solapur","Hubli-Dharwad","Mysore",
                                  "Tiruchirappalli","Bareilly","Aligarh","Tiruppur","Moradabad","Jalandhar","Bhubaneswar",
                                  "Salem","Warangal","Mira-Bhayandar","Thiruvananthapuram","Bhiwandi-Nizampur",
                                  "Saharanpur", "Gorakhpur","Bikaner","Amravati","Noida","Jamshedpur","Bhilai",
                                  "Cuttack","Firozabad","Kochi","Nellore","Bhavnagar","Dehradun","Durgapur","Asansol",
                                  "Rourkela","Nanded-Waghala","Kolhapur","Ajmer","Akola","Gulbarga","Jamnagar",
                                  "Ujjain","Loni","Siliguri","Jhansi","Ulhasnagar",   "Jammu","Nellore","Mangalore",
                                  "Belgaum","Ambattur","Tirunelveli","Malegaon","Gaya","Jalgaon","Udaipur","Maheshtala",
                                  "Davanagere","Kozhikode","Kurnool","Rajahmundry","Bokaro Steel City","South Dumdum","Bellary",
                                  "Patiala","Agartala","Bhagalpur","Muzaffarpur","Bhatpara","Panihati","Latur","Dhule","Rohtak",   
                                    "Korba","Bardhaman","Bhilwara","Begusarai","New Delhi","Mysuru","Kakinada","Nizamabad","Tumkur",
                                    "Ozhukarai","Kollam","Guntur","Shimoga","Tirupati","Pali-Marwar","Panipat","Ambala","Kadapa","Anantapur",
                                    "Karimnagar","Darbhanga","Nagercoil","Etawah","Hapur","Haldia","Bharatpur", "Panvel-Navi Mumbai",  "Alwar",
                                    "Kharagpur","Bally","Naihati","Panchkula","Tiruvottiyur","Rampur","Shahjahanpur","Satara","Mathura","Farrukhabad",
                                    "Bharuch","Sambalpur","Hugli-Chinsurah","Nadiad","Purnia","Barasat","Ballygunge","Munger","Kulti-Asansol","Sikar","Chhapra","Bhiwani"])

st.subheader("🏡 Property Status")

col1, col2 = st.columns(2)

with col1:
    under_construction = st.selectbox("Under Construction",["No","Yes"])
    ready_to_move = st.selectbox("Ready To Move",["Yes","No"])

with col2:
    rera = st.selectbox("RERA Approved",["Yes","No"])
    

if st.button("🔍 Predict Price"):

    with st.spinner("Predicting House Price..."):

        square_ft_log = np.log1p(square_ft)
        bhk_log = np.log1p(bhk)

        address_encoded = target_encoder.transform(
            pd.DataFrame({"ADDRESS": [address]})
        )["ADDRESS"].iloc[0]

        owner = 1 if posted_by == "Owner" else 0
        dealer = 1 if posted_by == "Dealer" else 0
        rk = 1 if bhk_or_rk == "RK" else 0

        under_construction_value = 1 if under_construction == "Yes" else 0
        ready_to_move_value = 1 if ready_to_move == "Yes" else 0
        rera_value = 1 if rera == "Yes" else 0
        resale_value = 1 if resale == "Yes" else 0

        input_df = pd.DataFrame({
            "UNDER_CONSTRUCTION":[under_construction_value],
            "RERA":[rera_value],
            "BHK_NO.":[bhk_log],
            "SQUARE_FT":[square_ft_log],
            "READY_TO_MOVE":[ready_to_move_value],
            "RESALE":[resale_value],
            "ADDRESS":[address_encoded],
            "LONGITUDE":[longitude],
            "LATITUDE":[latitude],
            "POSTED_BY_Dealer":[dealer],
            "POSTED_BY_Owner":[owner],
            "BHK_OR_RK_RK":[rk]
        })

        scale_cols = [
            "SQUARE_FT",
            "BHK_NO.",
            "LONGITUDE",
            "LATITUDE"
        ]

        input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        prediction = model.predict(input_df)[0]

        predicted_price = np.expm1(prediction)

        st.success(f"🏡 Estimated House Price: ₹ {predicted_price:,.2f} Lakhs")

        with st.expander("Show Processed Input Data"):
            st.dataframe(input_df)










