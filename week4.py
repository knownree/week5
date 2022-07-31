import numpy as np
from flask import Flask, request, render_template,url_for
import pickle
import csv

app=Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predict():
  int_features = [int(x) for x in request.form.values()]
  features=[np.array(int_features)]   
  prediction = model.predict(features)
  output = round(prediction[0],2)

  return render_template('index.html',prediction_text='Travel Cost should be $ {}'.format(output))

with app.test_request_context():
  print(url_for('predict'))

if __name__=='__main__':
  app.run(port=5000,debug=True)

