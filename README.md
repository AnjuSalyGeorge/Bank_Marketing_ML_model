**Bank Marketing Campaign - Predicting Client Subscription**

## Objective
The goal of this project is to build and evaluate machine learning models to predict whether a client will subscribe to a term deposit based on data from a direct marketing campaign conducted by a Portuguese banking institution.

## Dataset Information
The dataset used in this project includes information collected from previous marketing campaigns and contains the following attributes:
- **Bank Client Data**: Age, job, marital status, education, etc.
- **Campaign Data**: Number of contacts, days since last contact, and previous campaign outcome.
- **Economic Indicators**: Employment variation rate, consumer price index, and other macroeconomic variables.
- **Target Variable**: Whether the client subscribed to a term deposit (`y`).

**Source**:https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing

## Installation
To run this project, ensure you have Python installed along with the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

## Usage
### 1. **Data Preprocessing**
   - Load the dataset and perform data cleaning, including handling missing values and encoding categorical variables.
   - Split the data into training and testing sets.

### 2. **Model Training**
   - Several machine learning models are trained and evaluated, including:
     - Random Forest
     - XGBoost
   - Use cross-validation and hyperparameter tuning to optimize the models.

### 3. **Model Evaluation**
   - Evaluate the models using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC curve.
   - Select the best model based on performance metrics.

### 4. **Prediction**
   - Use the trained model to predict whether a new client will subscribe to a term deposit.

### 5. **Visualization**
   - Visualize the distribution of variables, feature importance, and model performance using plots.

## Results
The final model selected for deployment is **XGBoost**, which outperformed other models with the following performance metrics:
- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1-Score**: XX%
- **AUC-ROC**: XX

The model demonstrates strong predictive power and is robust in identifying potential subscribers to the term deposit product.

## Contributors
- **Anju George** - Data Analytics for Bussiness Student, St Clair College
