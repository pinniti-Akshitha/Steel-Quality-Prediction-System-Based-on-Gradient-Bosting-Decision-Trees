import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import xgboost

# Define a dummy XGBoostLabelEncoder class if you don't have the actual definition
class XGBoostLabelEncoder:
    # Placeholder for the actual implementation
    pass

# Create a custom Unpickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'XGBoostLabelEncoder':
            return XGBoostLabelEncoder
        return super().find_class(module, name)

app = Flask(__name__)  # Initialize the flask App

# Load the model using the custom Unpickler
with open('xgboost.pkl', 'rb') as file:
    steel = CustomUnpickler(file).load()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        return render_template("preview.html", df_view=df)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_feature = [x for x in request.form.values()]
    print(int_feature)
    final_features = [np.array(int_feature, dtype=object)]
    result = steel.predict(final_features)
    if result == 0:
        results = "Light_Load"
    elif result == 1:
        results = "Medium_Load"
    else:
        results = "Maximum_Load"
    return render_template('prediction.html', prediction_text=results)

@app.route('/performance')
def performance():
    return render_template('performance.html')

if __name__ == "__main__":
    app.run()
