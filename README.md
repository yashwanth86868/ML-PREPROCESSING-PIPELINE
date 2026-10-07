
# Machine Learning Data Preprocessing

This project demonstrates the basic data preprocessing steps required before building a Machine Learning model.

## Topics Covered

- Loading a dataset using Pandas
- Handling missing values
- Encoding categorical variables
- Encoding the target variable
- Splitting data into training and testing sets
- Feature scaling using StandardScaler
- Understanding `fit_transform()` and `transform()`

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Preprocessing Steps

### 1. Load Dataset

The dataset is loaded using Pandas.

```python
Dataset = pd.read_csv("data/Data.csv")
```

### 2. Separate Features and Target

```python
x = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, -1].values
```

### 3. Handle Missing Values

`SimpleImputer` is used to replace missing values.

```python
imputer = SimpleImputer()
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
```

### 4. Encode Categorical Data

`LabelEncoder` converts categorical values into numerical values.

```python
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
```

The target variable is also encoded:

```python
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
```

### 5. Train-Test Split

The dataset is divided into training and testing data.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0
)
```

### 6. Feature Scaling

`StandardScaler` is used to standardize the features.

```python
sc_X = StandardScaler()

x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
```

The scaler is fitted only on the training data and then used to transform the test data. This prevents data leakage.

## Learning Outcome

Through this project, I learned the fundamental data preprocessing techniques used before training Machine Learning models.

## Author

Yashwanth Balija
