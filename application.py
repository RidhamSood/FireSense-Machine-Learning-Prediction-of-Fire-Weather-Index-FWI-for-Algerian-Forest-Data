from flask import Flask, request, jsonify, render_template
# render_template is responsible for finding out the url of any 'html' file
import pickle
import numpy as np
import pandas as pd

# import ridge_regressor and  standard_model for prediction and standardization resp.
ridge_model = pickle.load(open(r'section_29\models\ridge_model.pkl','rb'))
standard_scaler = pickle.load(open(r'section_29\models\standardScaler.pkl','rb'))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        DC = float(request.form.get("DC"))
        ISI = float(request.form.get("ISI"))
        BUI = float(request.form.get("BUI"))
        new_standard_scaler = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI]])
        
        result = ridge_model.predict(new_standard_scaler)
        return render_template('home.html', result = result)

    else:
        return render_template('home.html')
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")# you can also change the port by using port = 8000, or 8080