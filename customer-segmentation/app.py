import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# =============================
# Load Model & Scaler
# =============================
scaler = joblib.load('models/scaler.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')

# Label & rekomendasi per cluster (Emoji dihapus)
segment_info = {
    0: {
        'label': 'Premium Customer',
        'karakteristik': 'Income tinggi, Spending tinggi',
        'strategi': 'Promo eksklusif, loyalty program premium, early access produk baru'
    },
    1: {
        'label': 'Loyal Buyer',
        'karakteristik': 'Income rendah, Spending tinggi',
        'strategi': 'Program reward, cashback, tingkatkan engagement & retensi'
    },
    2: {
        'label': 'Potential Target',
        'karakteristik': 'Income tinggi, Spending rendah',
        'strategi': 'Kampanye edukasi produk, upselling, penawaran bundle premium'
    },
    3: {
        'label': 'Middle Segment',
        'karakteristik': 'Income & Spending menengah',
        'strategi': 'Promo reguler, bundle package, diskon seasonal'
    },
    4: {
        'label': 'Price Sensitive',
        'karakteristik': 'Income rendah, Spending rendah',
        'strategi': 'Flash sale, diskon besar, voucher, program cicilan'
    }
}

# =============================
# UI Streamlit
# =============================
# Mengosongkan page_icon agar menggunakan default browser icon yang clean
st.set_page_config(page_title='Customer Segmentation', page_icon='', layout='centered')

st.title('Customer Segmentation App')
st.markdown('Masukkan data pelanggan untuk mengetahui segmen dan rekomendasi strategi pemasarannya.')
st.divider()

# Input
col1, col2 = st.columns(2)
with col1:
    income = st.slider('Annual Income (k$)', min_value=15, max_value=137, value=60)
with col2:
    spending = st.slider('Spending Score (1-100)', min_value=1, max_value=100, value=50)

st.divider()

# Predict
if st.button('Prediksi Segmen', use_container_width=True):
    input_data = np.array([[income, spending]])
    input_scaled = scaler.transform(input_data)
    cluster_id = kmeans.predict(input_scaled)[0]
    info = segment_info[cluster_id]

    # Hasil (Emoji bawaan di teks hasil juga sudah dibersihkan)
    st.success(f"**Segmen: {info['label']}**")
    st.info(f"Karakteristik: {info['karakteristik']}")
    st.warning(f"Rekomendasi: {info['strategi']}")

    st.divider()

    # Visualisasi posisi pelanggan baru
    df = pd.read_csv('data/Mall_Customers.csv')
    colors = ['#E63946', '#2A9D8F', '#E9C46A', '#457B9D', '#A8DADC']

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(5):
        cluster_data = df[df.index.isin(
            [j for j, c in enumerate(kmeans.predict(
                scaler.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']]))) if c == i]
        )]
        ax.scatter(
            df['Annual Income (k$)'][kmeans.predict(
                scaler.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])) == i],
            df['Spending Score (1-100)'][kmeans.predict(
                scaler.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])) == i],
            c=colors[i], alpha=0.5, s=60, label=segment_info[i]['label']
        )

    # Plot pelanggan baru
    ax.scatter(income, spending, c='black', marker='*', s=300, zorder=5, label='Pelanggan Baru')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Posisi Pelanggan Baru dalam Cluster')
    ax.legend(loc='upper left', fontsize=7)
    st.pyplot(fig)