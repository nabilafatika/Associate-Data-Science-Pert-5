import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Fungsi untuk memuat dan menampilkan dataset awal
@st.cache_data
def load_data():
    df = pd.read_csv("cybersecurity_attacks.csv")
    return df

# Fungsi untuk preprocessing dataset
def preprocess_data(df):
    # Menghapus kolom yang tidak diinginkan
    df.drop(columns=['Unnamed: 0','Timestamp','Payload Data', 'User Information', 'Geo-location Data',
                     'Source Port', 'Destination Port', 'Source IP Address', 'Destination IP Address',
                     'Alerts/Warnings', 'Proxy Information', 'Firewall Logs'], inplace=True)

    # Encoding Fitur Kategorikal
    categorical_features = ['Protocol', 'Packet Type', 'Traffic Type', 'Attack Type',
                            'Attack Signature', 'Severity Level', 'Log Source',
                            'Browser', 'Device/OS','Malware Indicators', 'Action Taken',
                            'Network Segment', 'IDS/IPS Alerts']

    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Standarisasi Fitur Numerik
    numerical_features = ['Packet Length', 'Anomaly Scores', 'Year', 'Month', 'Day',
                          'Hour', 'Minute', 'Second', 'DayOfWeek']
    
    scaler = StandardScaler()
    df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
    
    return df_encoded

# Fungsi untuk memuat model dan melakukan prediksi
def train_model(df_encoded):
    # Membagi Fitur dan Target
    target_column = 'Severity Level_Low'
    X = df_encoded.drop(columns=[col for col in df_encoded.columns if 'Severity Level' in col])
    y = df_encoded[target_column]

    # Membagi Data Menjadi Training dan Testing Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan melatih model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, y_test, y_pred

# Fungsi untuk menampilkan heatmap korelasi
# Fungsi untuk menampilkan heatmap korelasi
def plot_heatmap(df_encoded):
    numerical_features = ['Packet Length', 'Anomaly Scores', 'Year', 'Month', 'Day',
                          'Hour', 'Minute', 'Second', 'DayOfWeek']
    correlation_matrix = df_encoded[numerical_features].corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
    st.pyplot(fig)


# Main App
st.title('Cybersecurity Attacks Data Analysis & Modelling')

# Memuat data
df = load_data()
st.subheader('Dataset Awal')
st.write(df.head())

# Menampilkan informasi dataset
st.write(f"Jumlah Baris: {df.shape[0]}")
st.write(f"Jumlah Kolom: {df.shape[1]}")
st.write(df.describe())

# Preprocessing data
df_encoded = preprocess_data(df)
st.subheader('Dataset Setelah Preprocessing')
st.write(df_encoded.head())

# Menyimpan dataset yang sudah diolah
df_encoded.to_csv("dataset_cyber_olah.csv", index=False)
st.success('Dataset yang telah diolah berhasil disimpan.')

# Melatih model dan menampilkan hasil evaluasi
st.subheader('Modeling')
model, accuracy, y_test, y_pred = train_model(df_encoded)
st.write(f"Akurasi model: {accuracy:.2f}")
st.text(classification_report(y_test, y_pred))

# Menampilkan heatmap korelasi
st.subheader('Heatmap Korelasi Fitur Numerik')
plot_heatmap(df_encoded)

