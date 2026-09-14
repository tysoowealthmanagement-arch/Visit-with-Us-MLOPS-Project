# Streamlit app for Tourism Package Prediction
# Loads the model trained by train.py and predicts from user input.

import os
import joblib
import pandas as pd
import streamlit as st

model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

st.set_page_config(page_title="Tourism Package Prediction", page_icon="🧳", layout="wide")

st.title("🧳 Tourism Package Prediction")
st.caption("Visit with Us — will this customer buy the Wellness Tourism Package?")

with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts whether a customer is likely to purchase the "
        "Wellness Tourism Package, based on their profile and how the "
        "sales pitch went."
    )
    st.write("Model: trained and picked automatically by the MLOps pipeline.")

tab1, tab2, tab3 = st.tabs(["👤 Customer", "🧳 Trip Preferences", "📞 Sales Pitch"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Age", 18, 70, 35)
        Gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    with col2:
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Income", 0, 100000, 20000, step=500)
        CityTier = st.select_slider("City Tier", options=[1, 2, 3], value=1)
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 10, 3)
        NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 5, 0)
        NumberOfTrips = st.slider("Trips per Year (avg)", 0, 20, 2)
    with col2:
        PreferredPropertyStar = st.select_slider("Preferred Property Star", options=[3.0, 4.0, 5.0], value=3.0)
        Passport = st.checkbox("Has a Passport")
        OwnCar = st.checkbox("Owns a Car")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        DurationOfPitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
    with col2:
        NumberOfFollowups = st.slider("Number of Follow-ups", 0, 10, 3)
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# A plain button (not a form) so the result below only shows for the run
# where Predict was just clicked -- changing any input reruns the app and
# clears the old result immediately, instead of leaving it stuck on screen.
submitted = st.button("Predict", use_container_width=True, type="primary")

if submitted:
    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": 1 if Passport else 0,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": 1 if OwnCar else 0,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    result_col, chart_col = st.columns([2, 1])

    with result_col:
        if prediction == 1:
            st.success("### ✅ Likely to purchase the package")
            st.balloons()
        else:
            st.error("### ❌ Unlikely to purchase the package")
        st.write("Confidence the customer will buy:")
        st.progress(float(probability))

    with chart_col:
        st.metric("Purchase Probability", f"{probability:.0%}")

    with st.expander("See the data sent to the model"):
        st.dataframe(input_data, use_container_width=True)
