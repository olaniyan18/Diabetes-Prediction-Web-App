import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='template', static_folder='static')

model = pickle.load(open('diabetes.pkl', 'rb'))

@app.route('/homepage.html')
def home():
    return render_template('homepage.html', data = None)

@app.route('/heart_disease.html')
def home_2():
    return render_template('heart_disease.html', data = None)

@app.route('/predict', methods = ['GET','POST'])
def result():
    p = 0
    g = 0
    b = 0
    s = 0
    i = 0
    b = 0
    d = 0
    a = 0
    if request.method == 'POST':
        p = int(request.form['a'])
        g = int(request.form['b'])
        b = int(request.form['c'])
        s = int(request.form['d'])
        i = int(request.form['e'])
        bm = float(request.form['f'])
        d = float(request.form['g'])
        a = int(request.form['h'])
        print(f"p: {p}, car_type: {g}, meter: {b}, location: {s}, import type: {i}, engine: {bm},transmission: {d}, fuel_type:{a}")
            
        
    
        
        features = np.array([[p,g,b,s,i,bm,d,a]])        
        prediction = model.predict(features)
        
        print(f"Data to be sent: {prediction}")
        return render_template('homepage.html',data = prediction)
        
    return render_template('homepage.html', data = None)


if __name__ == '__main__':
    app.run(debug = True)
    