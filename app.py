from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
with open("model/titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from form
    pclass = int(request.form["pclass"])
    sex = int(request.form["sex"])
    age = float(request.form["age"])
    fare = float(request.form["fare"])
    
    # Prepare data for prediction
    features = np.array([[pclass, sex, age, fare]])
    prediction = model.predict(features)[0]
    
    # Return result
    result = "Survived" if prediction == 1 else "Did Not Survive"
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
