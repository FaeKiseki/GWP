# GWP
Here’s the **README.md** content:

---

# Garment Worker Productivity Prediction

This project is a **Machine Learning-based system** designed to predict the productivity of garment workers based on various operational and environmental factors. The system leverages regression models and a user-friendly web interface to deliver accurate predictions.

---

## **Project Overview**

Garment manufacturing is a labor-intensive industry, and predicting worker productivity is crucial for optimizing operations, improving efficiency, and reducing costs. This project aims to:
- Analyze relationships between productivity and operational features.
- Train, evaluate, and optimize regression models.
- Implement a web-based prediction system for end-user accessibility.

---

## **Dataset**

The dataset contains various operational metrics and productivity data for garment workers. It includes features like:
- **SMV**: Standard Minute Value.
- **Overtime**: Extra working hours.
- **Incentive**: Monetary rewards for workers.
- **Team, Department, Quarter, Day, Month**: Categorical features.

You can download the dataset here:  
**[Garment Worker Productivity Dataset](https://drive.google.com/file/d/1Y06CTFXy0_R67YlrRrAT-M4ddPAdlgON/view)**

---

## **Key Features**

1. **Exploratory Data Analysis (EDA)**:
   - Visualizations such as heatmaps and box plots to uncover feature relationships.
2. **Model Training and Evaluation**:
   - Models used: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, Support Vector Regressor.
   - Hyperparameter tuning with GridSearchCV.
3. **Web-Based Prediction System**:
   - Flask-based web app for users to input data and get predictions.

---

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-repo/garment-worker-productivity.git
cd garment-worker-productivity
```

### **2. Install Dependencies**

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### **3. Run the Web Application**

Start the Flask app:

```bash
python app/predictive_system.py
```

Access the app at `http://127.0.0.1:5000`.

---

## **Model Performance**

The following models were evaluated based on Mean Squared Error (MSE) and R-squared (R²):

- **Best Model**: Random Forest Regressor (MSE: 0.402, R²: 0.543).
- **Important Features**: Targeted Productivity, Team, No. of Workers, and Incentive.

---

## **Requirements**

The project dependencies are listed in the `requirements.txt` file:

```
Flask==2.3.2
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.0
matplotlib==3.7.2
xgboost==1.7.4
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## **Future Enhancements**

- **Feature Engineering**: Create derived features like `incentive_per_worker` or `efficiency_ratios`.
- **Cloud Deployment**: Host the application on AWS, Azure, or Heroku for real-time usage.
- **Dashboards**: Develop interactive dashboards for better visualization.

---

## **Acknowledgments**

This project is part of the **HubbleMind Capstone** initiative. Special thanks to the team for providing the dataset and project structure.

--- 

Let me know if you’d like further edits! 😊
