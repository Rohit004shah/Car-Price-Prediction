# flask, pandas, scikit-learn, pickle-mixin
from flask import Flask, render_template,request
import pandas as pd
import pickle


app = Flask(__name__)

model = pickle.load(open("linearRegressionModel.pkl",'rb'))
car = pd.read_csv("Cleaned Car.csv")


@app.route('/')
def index():
    car_models = sorted(car['car_name'].unique())
    fuel_type = car['fuel_type'].unique()
    cities = sorted(car['city'].unique())
    years = sorted(car['year_of_manufacture'].unique())
    return render_template('index.html',car_models=car_models,fuel_types=fuel_type,
                           cities=cities,years=years)


@app.route('/predict', methods=['POST'])
def predict():
    city= request.form.get('city')
    car_model = request.form.get('car_model')
    year = request.form.get('year_of_manufacture')
    fuel_type = request.form.get('fuel_type')
    kms_driven = request.form.get('kms_driven')

    prediction=model.predict(pd.DataFrame([[car_model,kms_driven,fuel_type,city,year]],
                                          columns=['car_name','kms_driven','fuel_type','city','year_of_manufacture']))
    return str(prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
