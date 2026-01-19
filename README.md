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
git clone https://github.com/your-username/water-potability-prediction.git
cd water-potability-prediction
2. Install Dependencies

Install the required libraries using pip:

pip install -r requirements.txt

3. Set Up the Environment

Ensure that you have Python 3.7+ installed on your machine.

If you don't have pip or virtualenv installed, you can install them via:

pip install virtualenv


Create a virtual environment (optional, but recommended):

virtualenv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

4. Start the Gradio App

To start the Gradio app and test the model locally, run:

python app.py


This will launch the Gradio interface in your browser, where you can input the water quality parameters and get predictions on whether the water is potable or non-potable.

Usage

Once the app is running, open the web interface in your browser (usually at http://127.0.0.1:7860).

Enter values for the following water quality parameters:

pH

Hardness

Solids

Chloramines

Sulfate

Conductivity

Organic Carbon

Trihalomethanes

Turbidity

After entering the values, click Submit to get the water potability prediction.

The model will predict if the water is potable (safe for drinking) or non-potable (unsafe for drinking).

Model Training

If you want to train the model yourself, follow these steps:

Prepare the Dataset: Ensure your dataset includes the features mentioned earlier and a target variable (Potability).

Train the Model: Use the following code to train the SVC model:

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


This will save the trained model as svc_model.pkl, which you can use in the Gradio app.

Contributing

We welcome contributions to this project! To contribute:

Fork the repository.

Create a new branch (git checkout -b feature-name).

Make your changes and commit them (git commit -am 'Add new feature').

Push to the branch (git push origin feature-name).

Open a pull request.

License

This project is licensed under the MIT License - see the LICENSE
 file for details.

Acknowledgements

Scikit-learn for providing the SVC algorithm and other machine learning tools.

Gradio for making it easy to build and deploy machine learning apps.

The dataset used for training the model (e.g., from Kaggle or a similar source).


### **Key Sections in the README**:

1. **Project Overview**: Describes the goal of the project and what the model does.
2. **Features**: Highlights the main features, including the use of **SVC** and **Gradio**.
3. **Installation Instructions**: Explains how to set up the project, including dependencies and environment setup.
4. **Usage Instructions**: Provides details on how to interact with the Gradio interface.
5. **Model Training**: Instructions on how to train the model from scratch if needed.
6. **Contributing**: Encourages contributions from others.
7. **License and Acknowledgements**: Provides details on the licensing and credits.

---

This README file should help others understand the project, how to set it up, and how they can contribute. Let me know if you need any additional sections or further customization!
