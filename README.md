# 🛍️ Customer Segmentation Using K-Means Clustering

> Unsupervised Machine Learning project to segment mall customers based on purchasing behavior — built as a Data Science portfolio project.

---

## 📌 Overview

Businesses often struggle to understand their customers at scale. This project applies **K-Means Clustering** to automatically segment mall customers into meaningful groups based on their **Annual Income** and **Spending Score**, enabling more targeted and effective marketing strategies.

| | |
|---|---|
| **Type** | Unsupervised Machine Learning (Clustering) |
| **Dataset** | Mall Customer Segmentation — Kaggle |
| **Algorithm** | K-Means Clustering + DBSCAN (comparison) |
| **Tools** | Python, Scikit-learn, Pandas, Matplotlib, Seaborn |

---

## 📊 Results

| Metric | K-Means | DBSCAN |
|--------|---------|--------|
| Silhouette Score | **0.5547** ✅ | 0.3876 |
| Davies-Bouldin Index | **0.5722** ✅ | 0.7889 |
| Clusters Found | **5** | 2 |
| Noise/Outlier | 0 | 8 |

**→ K-Means outperforms DBSCAN** on this dataset and is selected as the final model.

---

## 👥 Customer Segments

| Cluster | Segment | Characteristic | Strategy |
|---------|---------|----------------|----------|
| 0 | 💎 Premium Customer | High income, high spending | Exclusive promos, premium loyalty program |
| 1 | 🤝 Loyal Buyer | Low income, high spending | Rewards, cashback, high engagement |
| 2 | 🎯 Potential Target | High income, low spending | Upselling, product education campaigns |
| 3 | 📦 Middle Segment | Average income & spending | Regular promos, bundle packages |
| 4 | 💰 Price Sensitive | Low income, low spending | Flash sales, big discounts, vouchers |

---

## 🗂️ Project Structure

```
customer-segmentation/
│
├── data/
│   ├── Mall_Customers.csv          # Dataset
│   ├── distribusi_fitur.png        # Feature distribution plot
│   ├── scatter_plot.png            # Scatter plot EDA
│   ├── elbow_method.png            # Elbow method result
│   ├── silhouette_score.png        # Silhouette score plot
│   ├── cluster_result.png          # K-Means clustering result
│   └── dbscan_result.png           # DBSCAN clustering result
│
├── models/
│   ├── kmeans_model.pkl            # Trained K-Means model
│   └── scaler.pkl                  # Fitted StandardScaler
│
├── notebooks/
│   └── customer_segmentation.ipynb # Main notebook
│
└── README.md
```

---

## ⚙️ How to Run

**1. Clone repository**
```bash
git clone https://github.com/NamaKamu/customer-segmentation.git
cd customer-segmentation
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

**3. Open notebook**
```bash
jupyter notebook notebooks/customer_segmentation.ipynb
```

**4. Run all cells**

`Kernel → Restart & Run All`

---

## 📚 References

- Choudhary, V. (2019). Mall Customer Segmentation Data. [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.

---

## 👤 Author

**Mochmad Ilham Nadhif**  
Teknik Informatika — Universitas Dian Nuswantoro  
`A11.2024.15995`
