
import streamlit as st
import pandas as pd
import joblib

# 1. Load the AI Brain
model = joblib.load('titanic_model.pkl')

# 2. Design the App
st.title("🚢 Titanic Survival Predictor")
st.write("Would you have survived the sinking of the Titanic? Enter your details below to find out!")

# 3. Create Inputs in the Sidebar
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3], help="1st Class is the most expensive")
sex = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 25)
fare = st.sidebar.slider("Ticket Fare (£)", 0.0, 500.0, 32.0)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 10, 0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

# 4. Predict Button
if st.button("Did I Survive? 🌊"):
    
    # --- Format the data exactly how the AI expects it ---
    
    # Feature Engineering (Just like we did in training!)
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # One-Hot Encoding Conversions
    is_male = 1 if sex == "Male" else 0
    embarked_Q = 1 if embarked == "Queenstown" else 0
    embarked_S = 1 if embarked == "Southampton" else 0
    
    # Create a DataFrame with the exact same columns as our training data
    user_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'Sex_male': [is_male],
        'Embarked_Q': [embarked_Q],
        'Embarked_S': [embarked_S]
    })
    
    # 5. Make the Prediction
    prediction = model.predict(user_data)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.success("### 🎉 You Survived!")
        st.write("Your combination of class, gender, and age got you onto a lifeboat.")
        st.balloons()
    else:
        st.error("### 🧊 You Did Not Survive...")
        st.write("Sadly, you went down with the ship.")