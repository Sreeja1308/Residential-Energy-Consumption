import streamlit as st 
import pandas as pd 
import joblib
import datetime

st.set_page_config(page_title="Energy consumption")
st.title("Energy consumption")

model = joblib.load("Random_forest_model (2).pkl")

model_columns = [
    'num_occupants',
    'house_size_sqft',
    'monthly_income',
    'outside_temp_celsius',
    'year',
    'month',
    'day',
    'season',
    'heating_type_Electric',
    'heating_type_Gas',
    'heating_type_None',
    'cooling_type_AC',
    'cooling_type_Fan',
    'cooling_type_None',
    'manual_override_Y',
    'manual_override_N',
    'is_weekend',
    'temp_above_avg',
    'income_per_person',
    'square_feet_per_person',
    'high_income_flag',
    'low_temp_flag',
    'season_spring',
    'season_summer',
    'season_fall',
    'season_winter',
    'day_of_week_0',
    'day_of_week_6',
    'energy_star_home'
]

with st.form("User_inputs"):
    st.header("Enter Inputs: (info)")
    col1, col2 = st.columns(2)
    with col1:
        num_occupants = st.number_input("Number of Occupants", min_value=1, max_value=10)
        house_size = st.number_input("House size sqrt:", min_value=100, max_value=200)
        income = st.number_input("Income User :", min_value=10, max_value=2000)
        temp = st.number_input("Outside Temp ", value=32.2)
    with col2:
        date = st.date_input("Date", value=datetime.date.today())
        heating = st.selectbox("heating type", ["Gas", "Electric", "None"])
        cooling = st.selectbox("cooling type", ["AC", "Fan", "None"])
        manual = st.radio("Manual", ["Yes", "No"])
        energy_star = st.checkbox("Energy Star Certified Home")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        day_of_week = date.weekday()
        season_label = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
        }.get(date.month, 'fall')

        features = {
            'num_occupants': num_occupants,
            'house_size_sqft': house_size,
            'monthly_income': income,
            'outside_temp_celsius': temp,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'season': {'Spring': 2, 'Summer': 3, 'fall': 4, 'Winter': 1}[season_label],
            'heating_type_Electric': int(heating == 'Electric'),
            'heating_type_Gas': int(heating == 'Gas'),
            'heating_type_None': int(heating == 'None'),
            'cooling_type_AC': int(cooling == 'AC'),
            'cooling_type_Fan': int(cooling == 'Fan'),
            'cooling_type_None': int(cooling == 'None'),
            'manual_override_Y': int(manual == 'Yes'),
            'manual_override_N': int(manual == 'No'),
            'is_weekend': int(day_of_week >= 5),
            'temp_above_avg': int(temp > 22),
            'income_per_person': income / num_occupants,
            'square_feet_per_person': house_size / num_occupants,
            'high_income_flag': int(income > 10000),
            'low_temp_flag': int(temp < 22),
            'season_spring': int(season_label == 'spring'),
            'season_summer': int(season_label == 'summer'),
            'season_fall': int(season_label == 'fall'),
            'season_winter': int(season_label == 'winter'),
            'day_of_week_0': int(day_of_week == 0),
            'day_of_week_6': int(day_of_week == 6),
            'energy_star_home': int(energy_star)
        }

        df = pd.DataFrame([{col: features.get(col, 0) for col in model_columns}])
        prediction = model.predict(df)[0]
        st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")

    except Exception as e:
        st.error(f"An error occurred: {e}")