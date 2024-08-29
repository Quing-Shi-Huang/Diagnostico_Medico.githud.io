from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Carga y prepara el modelo
train = pd.read_csv('Entrenado.csv').drop('Unnamed: 133', axis=1)
test = pd.read_csv('Examen1.csv')

P = train[["prognosis"]]
X = train.drop(["prognosis"], axis=1)
Y = test.drop(["prognosis"], axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(X, P, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
model_rf = rf.fit(xtrain, ytrain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json.get('symptoms', [])
    symptom_columns = X.columns
    input_vector = [1 if symptom in symptoms else 0 for symptom in symptom_columns]
    probabilities = model_rf.predict_proba([input_vector])[0]
    disease_probabilities = dict(zip(model_rf.classes_, probabilities))
    sorted_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)
    return jsonify(disease_probabilities=sorted_diseases)

if __name__ == '__main__':
    app.run(debug=True)
