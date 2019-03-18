from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    #Alternative usage of Saved Model
   

    if request.method == 'POST':
        pro=request.form["Job"]
        my_prediction=0
        if pro=='0':
            loaded_model=pickle.load(open('Engineering.sav','rb'))
            exp=request.form['Experiance']
            my_prediction = loaded_model.predict([[float(exp)]])
        elif pro=='1':
            #global my_prediction
            loaded_model=pickle.load(open('Medicine.sav','rb'))
            exp=request.form['Experiance']
            my_prediction = loaded_model.predict([[float(exp)]])
        elif pro=='2':
            #global my_prediction
            loaded_model=pickle.load(open('Finance.sav','rb'))
            exp=request.form['Experiance']
            my_prediction = loaded_model.predict([[float(exp)]])
        elif pro=='3':
            #global my_prediction
            loaded_model=pickle.load(open('Law.sav','rb'))
            exp=request.form['Experiance']
            my_prediction = loaded_model.predict([[float(exp)]])
        elif pro=='4':
            #global my_prediction
            loaded_model=pickle.load(open('Business.sav','rb'))
            exp=request.form['Experiance']
            my_prediction = loaded_model.predict([[float(exp)]])
        elif pro=='5':
            #global my_prediction
            loaded_model=pickle.load(open('Teaching.sav','rb'))
            exp=request.form['Experiance']
            my_prediction = loaded_model.predict([[float(exp)]])
        
    return render_template('result.html', prediction=my_prediction[0][0])

if __name__ == '__main__':
    app.run(debug=True)


