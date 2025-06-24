import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
@st.cache_data
def load_data():
    fert_df = pd.read_csv("Fertilizer Prediction.csv")
    crop_df = pd.read_csv("Crop_recommendation.csv")
    return fert_df, crop_df

fertilizer_df, crop_recommendation_df = load_data()

# --------------------------
# Label Encoding for Fertilizer Data
# --------------------------
soil_type_encoder = LabelEncoder()
fertilizer_df['Soil Type Enc'] = soil_type_encoder.fit_transform(fertilizer_df['Soil Type'])

crop_type_encoder = LabelEncoder()
fertilizer_df['Crop Type Enc'] = crop_type_encoder.fit_transform(fertilizer_df['Crop Type'])

fert_name_encoder = LabelEncoder()
fertilizer_df['Fertilizer Name Enc'] = fert_name_encoder.fit_transform(fertilizer_df['Fertilizer Name'])

# --------------------------
# Label Encoding for Crop Data
# --------------------------
crop_label_encoder = LabelEncoder()
crop_recommendation_df['label Enc'] = crop_label_encoder.fit_transform(crop_recommendation_df['label'])

# --------------------------
# Train Models
# --------------------------
@st.cache_resource
def train_models():
    # Crop model
    X_crop = crop_recommendation_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_crop = crop_recommendation_df['label Enc']
    crop_model = RandomForestClassifier()
    crop_model.fit(X_crop, y_crop)
    crop_preds = crop_model.predict(X_crop)
    crop_acc = accuracy_score(y_crop, crop_preds)
    crop_cm = confusion_matrix(y_crop, crop_preds)

    # Fertilizer model
    X_fert = fertilizer_df[['Temparature', 'Humidity', 'Moisture', 'Soil Type Enc', 'Crop Type Enc',
                            'Nitrogen', 'Potassium', 'Phosphorous']]
    y_fert = fertilizer_df['Fertilizer Name Enc']
    fert_model = RandomForestClassifier()
    fert_model.fit(X_fert, y_fert)
    fert_preds = fert_model.predict(X_fert)
    fert_acc = accuracy_score(y_fert, fert_preds)
    fert_cm = confusion_matrix(y_fert, fert_preds)

    return crop_model, fert_model, X_crop.columns, X_fert.columns, crop_acc, fert_acc, crop_cm, fert_cm

crop_model, fert_model, crop_features, fert_features, crop_acc, fert_acc, crop_cm, fert_cm = train_models()

# --------------------------
# UI Section
# --------------------------
st.title("🌾 Crop and Fertilizer Prediction App")

choice = st.radio("Choose Prediction Type:", ["Crop", "Fertilizer"])

st.subheader("📝 Input Parameters")

# Encode dropdowns
soil_type_str = st.selectbox("🌱 Soil Type", fertilizer_df['Soil Type'].unique())
crop_type_str = st.selectbox("🌾 Crop Type", fertilizer_df['Crop Type'].unique())
soil_type_encoded = soil_type_encoder.transform([soil_type_str])[0]
crop_type_encoded = crop_type_encoder.transform([crop_type_str])[0]

if choice == "Crop":
    nitrogen = st.number_input("🧬 Nitrogen")
    phosphorous = st.number_input("🧪 Phosphorous")
    potassium = st.number_input("🧪 Potassium")
    temperature = st.number_input("🌡️ Temperature")
    humidity = st.number_input("💧 Humidity")
    ph = st.number_input("🔬 pH")
    rainfall = st.number_input("🌧️ Rainfall")

    if st.button("🔍 Predict Crop"):
        input_df = pd.DataFrame([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]],
                                columns=crop_features)
        prediction_enc = crop_model.predict(input_df)[0]
        prediction = crop_label_encoder.inverse_transform([prediction_enc])[0]
        st.success(f"🌿 Recommended Crop: **{prediction}**")
        st.write(f"📈 Accuracy: `{crop_acc * 100:.2f}%`")

        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(crop_cm, cmap='Blues', ax=ax)
        st.pyplot(fig)

elif choice == "Fertilizer":
    temp = st.number_input("🌡️ Temperature (for fertilizer)")
    humidity = st.number_input("💧 Humidity")
    moisture = st.number_input("🧪 Moisture")
    nitrogen = st.number_input("🧬 Nitrogen")
    potassium = st.number_input("🧪 Potassium")
    phosphorous = st.number_input("🧪 Phosphorous")

    if st.button("🔍 Predict Fertilizer"):
        input_df = pd.DataFrame([[temp, humidity, moisture, soil_type_encoded, crop_type_encoded,
                                  nitrogen, potassium, phosphorous]], columns=fert_features)
        prediction_enc = fert_model.predict(input_df)[0]
        prediction = fert_name_encoder.inverse_transform([prediction_enc])[0]
        st.success(f"🧪 Recommended Fertilizer: **{prediction}**")
        st.write(f"📈 Accuracy: `{fert_acc * 100:.2f}%`")

        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(fert_cm, cmap='Greens', ax=ax)
        st.pyplot(fig)
