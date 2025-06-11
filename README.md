# ML-Streamlit-Deployment

# Iris Species Classification using Random Forest

A machine learning project that classifies iris flowers into three species using the Random Forest algorithm. This project demonstrates basic data preprocessing, model training, and evaluation using the famous Iris dataset.

## Dataset

The project uses the Iris dataset (`Iris.csv`), which contains:
- **150 samples** of iris flowers
- **4 features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **3 species**: Iris-setosa, Iris-versicolor, Iris-virginica

## Features

- Data preprocessing and cleaning
- Species encoding (text to numeric labels)
- Train-test split (70% training, 30% testing)
- Random Forest classification
- Model accuracy evaluation

## Requirements

```
pandas
numpy
scikit-learn
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd iris-classification
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn
```

3. Ensure you have the `Iris.csv` file in the project directory

## Usage

Run the classification script:

```python
python iris_classifier.py
```

The script will:
1. Load and preprocess the Iris dataset
2. Split the data into training and testing sets
3. Train a Random Forest classifier
4. Make predictions on the test set
5. Display the accuracy score

## Code Structure

```python
# Data loading and preprocessing
df = pd.read_csv('Iris.csv')
df.drop('Id', axis=1, inplace=True)
df['Species'] = df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# Feature and target separation
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model training
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Prediction and evaluation
y_pred = classifier.predict(X_test)
score = accuracy_score(y_test, y_pred)
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Train-Test Split**: 70-30
- **Random State**: 0 (for reproducible results)
- **Features Used**: All 4 iris measurements

## Species Encoding

| Species | Numeric Label |
|---------|---------------|
| Iris-setosa | 0 |
| Iris-versicolor | 1 |
| Iris-virginica | 2 |

## Expected Results

The Random Forest classifier typically achieves high accuracy (>95%) on the Iris dataset due to the clear separation between species based on the measured features.

## File Structure

```
├── iris_classifier.py    # Main classification script
├── Iris.csv             # Dataset file
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- The Iris dataset was originally collected by Edgar Anderson and made famous by Ronald Fisher
- Built using scikit-learn's powerful machine learning tools
