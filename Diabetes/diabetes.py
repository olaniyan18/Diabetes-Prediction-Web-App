import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='template', static_folder='static')

model = pickle.load(open('diabetes.pkl', 'rb'))
model_heart = pickle.load(open('heart_disease.pkl', 'rb'))

@app.route('/homepage.html')
def home():
    return render_template('homepage.html', data = None)

@app.route('/h_disease.html')
def home_2():
    return render_template('h_disease.html', data = None)

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
                
        
        features = np.array([[p,g,b,s,i,bm,d,a]])        
        prediction = model.predict(features)
        
        print(f"Data to be sent: {prediction}")
        return render_template('homepage.html',data = prediction)
        
    return render_template('homepage.html', data = None)

@app.route('/predict_heart', methods = ['GET', 'POST'])
def result_heart():
    aa = 0
    ss = 0
    cp = 0
    tres = 0
    chol = 0
    fbs = 0
    rest = 0
    thalach = 0
    exang = 0
    oldpeak = 0
    slope = 0
    ca = 0
    thal = 0
    
    if request.method == 'POST':
        aa = int(request.form['aa'])
        ss = int(request.form.get('bb'))
        cp = int(request.form.get('cc'))
        tres = int(request.form['dd'])
        chol = int(request.form['ee'])
        fbs = int(request.form.get('ff'))
        rest = int(request.form.get('gg'))
        thalach = int(request.form['hh'])
        exang = int(request.form.get('ii'))
        oldpeak = float(request.form.get('jj'))
        slope = int(request.form.get('kk'))
        ca = int(request.form.get('ll'))
        thal = int(request.form.get('mm'))
        
        features = np.array([[aa, ss, cp, tres, chol, fbs, rest, thalach, exang, oldpeak, slope, ca, thal]])        
        prediction2 = model_heart.predict(features)
        
        print(f"Data to be sent: {prediction2}")
        return render_template('h_disease.html',data = prediction2)
    return render_template('h_disease.html', data = None)
    
        

if __name__ == '__main__':
    app.run(debug = True)
    
