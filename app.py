from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
with open('predictionmodel.pkl' , 'rb') as file:
  clf = pickle.load(file)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    MolLogP = request.form.get('one')
    MolWt = request.form.get('two')
    NumRotatableBonds = request.form.get('three')
    AromaticProportion = request.form.get('four')
    five = request.form.get('five')

    input_query = np.array([[1.0	,100.0	,1300.0	,3470.0,	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    result = clf.predict(input_query)[0]



  # result = {'MolLogP': MolLogP, 'MolWt': MolWt, 'NumRotatableBonds': NumRotatableBonds, 'AromaticProporti: AromaticProportion}
    return jsonify('result is : ' + str(result))


if __name__ == '__main__':
    app.run(debug = True)

