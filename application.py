import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
# Here we will train the model for the meantime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
df = pd.read_csv('shoppers.csv')
y = df.iloc[:,-1]
features=['Administrative_Duration','ProductRelated_Duration','ExitRates','PageValues']
X = df.loc[:,features]

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scaler = StandardScaler()

sm = SMOTE(random_state=42)
smote_enn = SMOTEENN(smote = sm)
clf_log =LogisticRegression(C=0.000695193,class_weight='balanced',penalty='l1',solver='saga')


pipe_over_sample = Pipeline([('smote_enn', smote_enn),
                 ('scaler',scaler),
                ('clf_log', clf_log)])

pipe_over_sample.fit(X,y)

@app.route('/contact',methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/model',methods=['GET','POST'])
def model():
    return render_template('model.html')


@app.route('/predict',methods=['POST'])
def predict():
   try:
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = pipe_over_sample.predict(final_features)

        if prediction == True:
            output="Customer Will Purchase"
        else:
            output="Customer Will Not Purchase"
    except :
        output="Wrong Data Type Input"

    return render_template('model.html', prediction_text = output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
@app.route('/doc/feature')
def feature():
    return render_template('feature_selection.html')

@app.route('/doc/variables')
def variables():
    return render_template('variables.html')

@app.route('/doc/cross-val')
def cross_val():
    return render_template('cross_val.html')

@app.route('/doc/oversampling')
def oversampling():
    return render_template('oversampling.html')

@app.route('/doc/logistic')
def logistic():
    return render_template('logistic.html')
@app.route('/doc/random_forest')
def random_forest():
    return render_template('random_forest.html')
@app.route('/doc/AdaBoost')
def AdaBoost():
    return render_template('AdaBoost.html')

@app.route('/doc/KNN')
def KNN():
    return render_template('KNN.html')

@app.route('/doc/Neural')
def Neural():
    return render_template('Neural.html')

if __name__ == "__main__":
    app.run(debug=False)
