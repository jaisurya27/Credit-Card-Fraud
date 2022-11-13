from flash import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
clf = pickle.load(open('clf.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    features=[float(x) for x in request.form.values()]
    final=[np.array(features)]
    pred1=clf.predict(final)
    pred2=model.predict(final)
    
    
    if pred1 == pred2 == 0:
        return render_template('index.html',pred='Normal transaction')
    elif pred1 == 0 and pred2==1:
        return render_template('index.html',pred='Maybe Fraudulent transaction')
    else:
        return render_template('index.html',pred='Fraudulent transaction')



if __name__ == '__main__':
    app.run()