import pickle
from flask import Flask, request, jsonify, render_template
from model import GPT2PPL
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import os
if not os.path.isfile('tfidf_vectorizer.pkl'):
   print('File tfidf_vectorizer.pkl not found')


app = Flask(__name__)
gpt2ppl = GPT2PPL()

with open('lsvm_model.pkl', 'rb') as f:
         lsvm_model = pickle.load(f)

with open('rf_model.pkl','rb') as f:
    rf_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
     tfidf_vectorizer = pickle.load(f)


with open('rf_vectorizer.pkl', 'rb') as f:
     rf_vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form['text']
    results, label = gpt2ppl(data)

    if not data:
        return jsonify({'results': results, 'label': 'Undefined - Please input some text!', 'lsvm_label': 'Undefined - Please input some text!', 'rf_label': 'Undefined - Please input some text!'})

    if (data != " "  and len(data)<100):
        return jsonify({'results': results, 'label': 'Undefined - Please input min 100 characters!', 'lsvm_label': 'Undefined - Please input min 100 characters!', 'rf_label': 'Undefined - Please input min 100 characters!'})
    
    
    data_tfidf = tfidf_vectorizer.transform([data])
    lsvm_label = lsvm_model.predict(data_tfidf).tolist()
    data_rf_tfidf = rf_vectorizer.transform([data])
    rf_label = rf_model.predict(data_rf_tfidf).tolist()
    
    print('LSVM',lsvm_label)
    print('RF',rf_label)
        
    return jsonify({'results': results, 'label': label,'lsvm_label':lsvm_label, 'rf_label' : rf_label})

if __name__ == '__main__':
    app.run(debug=True)
