import os
from flask import (
     Flask, 
     request, 
     render_template)

from model import predict_price

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/calc",methods=['GET','POST'])
def predict():
    if request.method == "GET":
        return render_template('result.html')
    elif request.method == "POST":
        predict_day = request.form['predict_day']
        vegetable = request.form['vegetable']
        price, data = predict_price(predict_day, vegetable)

        return render_template('result.html', result=price, data = data, predict_day = predict_day)


if __name__ == "__main__":
    app.run(debug=True)
