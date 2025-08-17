# Titanic: Machine Learning from Disaster

## 1. Project Overview

This project presents a complete data science workflow to predict the survival of passengers aboard the RMS Titanic. Using the famous dataset from the Kaggle competition, we explore the data, engineer new features based on historical context, and build a predictive model to determine whether a passenger survived the tragedy.

The primary goal is to create a robust classification model that achieves high accuracy and provides insights into the factors that influenced survival. This project serves as a demonstration of skills in data cleaning, feature engineering, model selection, and evaluation.

---

## 2. The Dataset

The dataset is split into two files: `train.csv` and `test.csv`.

* **`train.csv`**: Contains data for a subset of passengers, including whether they survived (`Survived` column), which serves as our training data.
* **`test.csv`**: Contains data for the remaining passengers, without the survival information. Our goal is to predict survival for these passengers.

### Key Features:
* **`Pclass`**: Ticket class (a proxy for socio-economic status).
* **`Sex`**: Passenger's gender.
* **`Age`**: Passenger's age in years.
* **`SibSp`**: Number of siblings or spouses aboard.
* **`Parch`**: Number of parents or children aboard.
* **`Fare`**: The fare paid for the ticket.
* **`Cabin`**: The passenger's cabin number.
* **`Embarked`**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## 3. Methodology

The project follows a structured machine learning pipeline:

1.  **Exploratory Data Analysis (EDA)**: Initial analysis to understand the data's structure, identify missing values, and uncover relationships between features and the survival outcome.
2.  **Feature Engineering**: New features are created to enhance the model's predictive power. This includes:
    * Extracting passenger **titles** (e.g., Mr., Mrs., Master) from names.
    * Creating a **`FamilySize`** feature from `SibSp` and `Parch`.
    * Categorizing `FamilySize` into groups like `Alone`, `Small_Family`, and `Large_Family`.
    * Extracting the **`Deck`** from the `Cabin` number.
3.  **Data Preprocessing**: The data is cleaned and prepared for modeling. This involves:
    * **Imputing Missing Values**: Filling missing `Age`, `Fare`, and `Embarked` values using appropriate strategies (e.g., median imputation grouped by class and title).
    * **Encoding Categorical Features**: Converting categorical variables like `Sex`, `Embarked`, `Title`, and `Deck` into a numerical format using one-hot encoding.
4.  **Modeling**: An **XGBoost (Extreme Gradient Boosting)** classifier is chosen for its high performance and robustness. The model is trained on the entire training dataset.
5.  **Prediction**: The trained model is used to predict survival outcomes for the passengers in the test dataset.
6.  **Submission**: A `submission.csv` file is generated in the format required by the Kaggle competition.

---

## 4. How to Run

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd Titanic-Survival-Prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```
3.  **Place the data:** Download `train.csv` and `test.csv` from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data) and place them in the root directory of the project.
4.  **Run the Jupyter Notebook:** Open and run the `titanic_survival_prediction.ipynb` notebook to see the full analysis and generate the `submission.csv` file.
