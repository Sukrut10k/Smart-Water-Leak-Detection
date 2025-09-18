# Smart Water Leak Detection & Monitoring Dashboard

## Overview

The **Smart Water Leak Detection System** leverages **Machine Learning** and **Interactive Dashboards** to detect and monitor water leakages in pipelines.
It uses IoT sensor data (pressure, flow rate, vibration, temperature, etc.) and provides **real-time leak detection, what-if analysis, batch predictions, and geo-visualization**.

This project consists of:

* **ML Models (Jupyter Notebook)**: Training and evaluation (`water_leak_detection.ipynb`).
* **Frontend Dashboard (Streamlit App)**: Interactive monitoring (`app.py`).

---

## Features

*  **Leak Prediction** – Detect leakage for individual pipeline inputs using trained ML models (Random Forest, Decision Tree, Logistic Regression, SVM).
*  **What-if Analysis** – Simulate different sensor values with sliders and instantly see prediction changes.
*  **Zone Map & Geo Analysis** – Interactive GIS map showing leak locations, active/dead pipes, and zone-wise analytics.
*  **Batch Prediction** – Upload a CSV dataset for bulk leak detection with downloadable results.
*  **Analytics & Insights** – Feature importance, model comparisons, and system insights.
*  **AI Assistant (Groq LLaMA Model)** – Ask AI for leak trends, maintenance insights, and generate plots dynamically.
*  **About Page** – Quick overview of the system.

---

##  Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Visualization**: Plotly, Folium (maps), Streamlit-Folium
* **Machine Learning**: Scikit-learn (Random Forest, Decision Tree, Logistic Regression, SVM)
* **AI Assistant**: Groq API (LLaMA models)
* **Data Handling**: Pandas, NumPy

---

##  Project Structure

```
Smart-Water-Leak-Detection/
│── app.py                        # Streamlit frontend
│── water_leak_detection.ipynb    # ML model training & experiments
│── random_forest_model.pkl       # Pre-trained ML model
│── decision_tree_model.pkl
│── logistic_regression_model.pkl
│── svm_model.pkl
│── location_aware_gis_leakage_dataset.csv   # dataset
│── testing.csv #this is for batch processing test 
│── requirements.txt               # Dependencies
│── README.md                      # Project documentation
```

---

##  Installation & Setup

###  Clone Repository

```bash
git clone https://github.com/your-username/Smart-Water-Leak-Detection.git
cd Smart-Water-Leak-Detection
```

###  Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Run ML Notebook (Optional - Training Models)

Open `water_leak_detection.ipynb` in Jupyter Notebook / JupyterLab:

```bash
jupyter notebook water_leak_detection.ipynb
```

This will allow you to retrain and export `.pkl` models.

### Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## How to Use

 **Navigate using Sidebar Menu**:

   * Predict Leakage → Input pipeline parameters & predict.
   * What-if Analysis → Adjust sliders for simulations.
   * Zone Map & Geo Analysis → Visualize leaks on interactive maps.
   * Batch Prediction → Upload CSV for bulk predictions.
   * Analytics & Insights → Explore feature importance & model comparison.
   * AI Assistant → Ask questions and request plots.

 **Dataset Requirements (for Batch Prediction & AI Assistant)**:

   * Columns: `Pressure, Flow_Rate, Temperature, Vibration, RPM, Operational_Hours, Latitude, Longitude, Zone, Block, Pipe, Location_Code, Leakage_Flag(optional)`

---

## Model Performance
* Random Forest: 0.98 accuracy (best performing)
* Decision Tree: 0.91 accuracy
* SVM: 0.90 accuracy
* Logistic Regression: 0.89 accuracy

##  Future Improvements

* Integrate **real IoT sensor data streams** (MQTT/Kafka).
* Deploy on **cloud platforms** (AWS/GCP/Azure).
* Add **deep learning models** for anomaly detection.
* Enhance **AI assistant with multimodal support** (voice + text).
