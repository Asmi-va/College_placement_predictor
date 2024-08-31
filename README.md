---

# College Placement Prediction

## Overview

This project involves predicting college placement outcomes using machine learning algorithms. The dataset, `collegePlace.csv`, includes various features related to students and their placement status. The goal is to build and evaluate classification models to predict whether a student will be placed based on their features.

## Features

- **Gender**: Gender of the student.
- **Stream**: Academic stream of the student.
- **Age**: Age of the student (commented out in the code).
- **Hostel**: Hostel status of the student (commented out in the code).
- **PlacedOrNot**: Target variable indicating whether the student has been placed (1) or not (0).

## Installation

### Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/college-placement-prediction.git
    cd college-placement-prediction
    ```

2. **Install Required Packages**

    It is recommended to use a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

    Install the necessary packages:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3. **Download the Dataset**

    Ensure the dataset `collegePlace.csv` is in the project directory.

## Usage

1. **Load and Explore the Data**

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline

    df = pd.read_csv('collegePlace.csv')
    df.shape
    df.head()
    df.describe()
    df.info()
    df.isnull().sum()
    ```

2. **Preprocess the Data**

    ```python
    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Stream'] = le.fit_transform(df['Stream'])
    ```

3. **Visualize the Data**

    ```python
    sns.pairplot(df)
    tc = df.corr()
    sns.heatmap(tc)
    ```

4. **Prepare Data for Modeling**

    ```python
    x = df.drop(columns=['PlacedOrNot', 'Hostel'])
    y = df['PlacedOrNot']
    ```

5. **Split the Data**

    ```python
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)
    ```

6. **Train and Evaluate Models**

    **Decision Tree Classifier**

    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics

    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    cm = metrics.confusion_matrix(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    pre = metrics.precision_score(y_test, y_pred)
    re = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    ```

    **Random Forest Classifier**

    ```python
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(n_estimators=1000)
    classifier.fit(x_train, y_train)
    y_pred_rf = classifier.predict(x_test)

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    acc_rf = metrics.accuracy_score(y_test, y_pred_rf)
    pre_rf = metrics.precision_score(y_test, y_pred_rf)
    ```

## Results

- **Decision Tree Classifier:**
    - Accuracy: `acc`
    - Precision: `pre`
    - Recall: `re`
    - F1 Score: `f1`

- **Random Forest Classifier:**
    - Accuracy: `acc_rf`
    - Precision: `pre_rf`

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

---

