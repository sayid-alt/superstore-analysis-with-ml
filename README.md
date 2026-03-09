# Superstore Analysis Project

## **Objective**
The primary objective of this analysis is to discover how to provide better actions for customers, such as offering personalized discounts, by leveraging customer segmentation and order prediction models.

### **Dashboard Streamlit link:**
```
https://superstore-dashboard-with-ml.streamlit.app/
```

---

## **Project Flows**

### **1. Data Preparation & Feature Engineering**
The project starts by processing the Superstore dataset (`data/SuperStore.csv`).
- **Feature Engineering**: The [feature_eng_pipeline.py](file:///Users/heykalsayid/Desktop/myown/superstore_analysis/src/superstore_analysis/pipelines/feature_eng_pipeline.py) extracts temporal features (`Month_Order`, `Day_Order`, `Days_Shipping`) and aggregates sales data by month, product, and sub-category.
- **Encoding**: Categorical variables are transformed using OneHot and Ordinal encoding via the [processor.py](file:///Users/heykalsayid/Desktop/myown/superstore_analysis/src/superstore_analysis/processor.py) utility.

### **2. Customer Segmentation (Clustering)**
Customers are grouped into distinct segments based on their purchasing behavior.
- **Pipeline**: The [training_pipeline.py](file:///Users/heykalsayid/Desktop/myown/superstore_analysis/src/superstore_analysis/pipelines/training_pipeline.py) defines a `ClusterPipeline` that uses `MinMaxScaler` for scaling and `PCA` for dimensionality reduction.
- **Algorithms**: It explores multiple clustering models (`KMeans`, `DBSCAN`, `MeanShift`) using `GridSearchCV` to find the optimal segments.
- **Output**: The results are saved in [Clustered.csv](file:///Users/heykalsayid/Desktop/myown/superstore_analysis/data/Clustered.csv), which is used for further analysis.

### **3. Predictive Modeling**
The project includes two main predictive capabilities:
- **Cluster Prediction (Classification)**: A classification model ([classify_model.pkl](file:///Users/heykalsayid/Desktop/myown/superstore_analysis/models/classify_model.pkl)) is trained to assign new customers to one of the identified segments based on their attributes (Sales, Category, Region, etc.).
- **Order Count Prediction (Regression)**: A regression model ([regression_model.pkl](file:///Users/heykalsayid/Desktop/myown/superstore_analysis/models/regression_model.pkl)) predicts future order volumes, aiding in resource planning and inventory management.

### **4. Interactive Dashboard**
A Streamlit application ([streamlit_app.py](file:///Users/heykalsayid/Desktop/myown/superstore_analysis/streamlit_app.py)) provides a user-friendly interface to explore the analysis results:
- **Order Count Prediction**: Visualizes historical and predicted order counts for a user-specified number of days.
- **Cluster Analysis**: Provides deep dives into each customer segment, highlighting profit margins, sales distribution, and regional trends.
- **Cluster Prediction**: An interactive form where users can input customer data to predict their segment in real-time.

---

## **Project Structure**
- `data/`: Raw and processed datasets.
- `models/`: Serialized machine learning models and preprocessing objects.
- `notebooks/`: Exploratory Data Analysis (EDA) and model development experiments.
- `pages/`: Individual Streamlit pages for different dashboard features.
- `src/`: Core logic, including data processing, custom encoders, and training/inference pipelines.
- `streamlit_app.py`: Main entry point for the Streamlit application.

---

## **How to Run**
1. Install dependencies (e.g., using `pip install -r requirements.txt` or `uv sync`).
2. Launch the dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```
