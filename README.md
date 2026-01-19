
# Water Potability Prediction using SVC Model

This project provides a machine learning model for predicting the potability of water based on various water quality parameters using a Support Vector Classifier (SVC). The model is trained using a dataset containing water quality features, and the trained model is made available through a simple **Gradio** web interface.

## Project Overview

The goal of this project is to predict whether the water is potable or non-potable based on the following features:

- **ph**: pH value of the water
- **Hardness**: Hardness of the water
- **Solids**: Total dissolved solids in the water
- **Chloramines**: Chloramine concentration in the water
- **Sulfate**: Sulfate concentration in the water
- **Conductivity**: Electrical conductivity of the water
- **Organic_carbon**: Organic carbon concentration in the water
- **Trihalomethanes**: Trihalomethane concentration in the water
- **Turbidity**: Water turbidity (cloudiness)

The model is trained using the **SVC (Support Vector Classifier)** algorithm and provides an interactive **Gradio** interface where users can input the features and get a prediction about the potability of water.

## Features

- **Machine Learning Model**: SVC classifier trained on the water quality dataset.
- **Web Interface**: Simple Gradio web interface to input features and get predictions.
- **Prediction Output**: The app returns whether the water is potable (safe to drink) or non-potable (unsafe to drink).

## Installation

To run this project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/asif4762/Is-the-Water-Drinkable-.git
cd water-potability-prediction
```

### 2. Install Dependencies

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Set Up the Environment

Ensure that you have Python 3.7+ installed on your machine.

If you don't have **pip** or **virtualenv** installed, you can install them via:

```bash
pip install virtualenv
```

Create a virtual environment (optional, but recommended):

```bash
virtualenv venv
source venv/bin/activate  # On Windows, use venv\Scriptsctivate
```

### 4. Start the Gradio App

To start the Gradio app and test the model locally, run:

```bash
python app.py
```

This will launch the Gradio interface in your browser, where you can input the water quality parameters and get predictions on whether the water is potable or non-potable.

## Usage

Once the app is running, open the web interface in your browser (usually at `http://127.0.0.1:7860`).

- Enter values for the following water quality parameters:
  - **pH**
  - **Hardness**
  - **Solids**
  - **Chloramines**
  - **Sulfate**
  - **Conductivity**
  - **Organic Carbon**
  - **Trihalomethanes**
  - **Turbidity**

- After entering the values, click **Submit** to get the water potability prediction.

The model will predict if the water is **potable** (safe for drinking) or **non-potable** (unsafe for drinking).

## Model Training

If you want to train the model yourself, follow these steps:

1. **Prepare the Dataset**: Ensure your dataset includes the features mentioned earlier and a target variable (`Potability`).
2. **Train the Model**: Use the following code to train the **SVC model**:

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset (replace with your dataset)
df = pd.read_csv("water_quality.csv")
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVC model
model = SVC(kernel="rbf", C=1, gamma="scale")
model.fit(X_train_scaled, y_train)

# Save the model to a file
with open('svc_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

This will save the trained model as **`svc_model.pkl`**, which you can use in the Gradio app.

## Contributing

We welcome contributions to this project! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Scikit-learn** for providing the SVC algorithm and other machine learning tools.
- **Gradio** for making it easy to build and deploy machine learning apps.
- The dataset used for training the model (e.g., from Kaggle or a similar source).

## Author - Asif Zaman
