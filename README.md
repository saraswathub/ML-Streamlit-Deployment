# ML-Streamlit-Deployment

# Iris Species Classification using Random Forest

A machine learning project that classifies iris flowers into three species using the Random Forest algorithm. This project includes both a command-line classifier and an interactive Streamlit web application for real-time predictions.

## Dataset

The project uses the Iris dataset (`Iris.csv`), which contains:
- **150 samples** of iris flowers
- **4 features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **3 species**: Iris-setosa, Iris-versicolor, Iris-virginica

## Features

- **Command-line classifier** for batch processing
- **Interactive Streamlit web app** with real-time predictions
- Data preprocessing and cleaning
- Species encoding (text to numeric labels)
- Train-test split (70% training, 30% testing)
- Random Forest classification
- Model accuracy evaluation
- **Web app features:**
  - Interactive sliders for input
  - Real-time predictions with confidence scores
  - Data visualizations and charts
  - Model performance metrics
  - Confusion matrix and classification report

## Requirements

```
pandas
numpy
scikit-learn
streamlit
plotly
seaborn
matplotlib
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd iris-classification
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn streamlit plotly seaborn matplotlib
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

3. Ensure you have the `Iris.csv` file in the project directory

## Usage

### Command Line Version

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

### Streamlit Web App

Launch the interactive web application:

```bash
streamlit run app.py
```

The web app provides:
- **Interactive input sliders** for flower measurements
- **Real-time predictions** with confidence scores
- **Visualizations** of the dataset and model performance
- **Model evaluation metrics** including confusion matrix
- **Feature importance** analysis

#### Web App Features:
- 🎯 **Live Predictions**: Adjust sliders to see instant predictions
- 📊 **Interactive Charts**: Plotly-powered visualizations
- 🔍 **Model Insights**: Confusion matrix and classification report
- 📈 **Data Exploration**: Scatter plots and distribution charts

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

## Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as your main file
5. Deploy!

### Heroku Deployment

1. Create a `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

## File Structure

```
├── iris_classifier.py    # Main classification script
├── app.py               # Streamlit web application
├── Iris.csv             # Dataset file (optional - app uses sklearn dataset)
├── requirements.txt     # Python dependencies
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
