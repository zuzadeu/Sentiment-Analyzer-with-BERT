from flask import Flask, request
from flask_cors import CORS, cross_origin
from common import model_inference, text_cleaning
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = tf.keras.models.load_model('model.h5')

@app.route('/score', methods = ['PUT'])
@cross_origin()
def score():
    print(request)
    text = request.json['text']
    df = pd.DataFrame(columns=['text']).append({'text': text}, ignore_index=True)
    df = text_cleaning(df, 'text')
    preproc_text = df.iloc[0][0]
    score = float(model_inference(model, [preproc_text]))
    return {"score": score}

if __name__ == '__main__':
    app.run()