from flask import Flask, jsonify, request
import pickle
import numpy as np
from helper.helper import predict_model as predict

PORT =  80
app = Flask(__name__)


@app.route('/analyze/price-home', methods=["POST"])
def analyze_price():
    """Predict a price house
        [payload]: {
	                "sqft_lot": int, -> param x for the model
	                "model_name":str -> in path "models" , name of binary
                    }
    """
    try:
        
        payload = request.json
        sqft_lot = payload['sqft_lot']
        model_name = payload['model_name']
        
        # transform sqft_lot 
        x_feature = np.array([sqft_lot]).reshape(1,1)
        
        # # read model (binary file)
        modelo_carregado = pickle.load(
                                    open(
                                        f'./models/{model_name}.sav','rb'
                                        ))
        
        # use predict method(linear regression)
        predict = modelo_carregado.predict(x_feature)[0]
        predict_repsonse = "{0:.2f}".format(round(float(predict),2))

        return jsonify({'price_predict':f'${predict_repsonse}','sqft_lot': sqft_lot}), 200
    except Exception as e:
        return jsonify({'message':e}), 400

@app.route('/analyze/luan', methods=["POST"])
def analyze_luan():
    """Predict a price house
        [payload]: {
	                "day": int, -> param x for the model
                    }
    """
    try:
        
        payload = request.json
        number_day = payload['number_day']
        model_name = 'luan_model'
        
        x_feature = np.array([number_day]).reshape(1,1)
        
        # read model (binary file)
        modelo_carregado = pickle.load(
                                    open(
                                        f'./models/{model_name}.sav','rb'
                                        ))
        
        # use predict method(linear regression)
        predict = modelo_carregado.predict(x_feature)[0]
        predict_repsonse = "{0:.2f}".format(round(float(predict),2))

        return jsonify({'price_predict':f'${predict_repsonse}','number_day': number_day}), 200
    except Exception as e:
        return jsonify({'message':e}), 400

@app.route('/analyze/nlp', methods=["POST"])
def analyze_nlp():
    """classifier sentence:
            - none
            - fazerTrocas
            - MeusPedidos
        [payload]: {
	                "sentence":"nao consigo concluir pedido"
                    "model":"modelo_nlp"
                    }
    """
    try:
        
        payload = request.json
        sentence = payload['sentence']
        model_name = payload['model_name']
        
        
        
        # use predict method(linear regression)
        predict_sentence = predict(sentence,
                                   model_name)

        return jsonify({'classifier':predict_sentence,'sentence': sentence}), 200
    except Exception as e:
        return jsonify({'message':e}), 400


if __name__ == '__main__':
    print(f'estou rodando na porta {PORT} 👹🌸')
    app.run(port=PORT, host='0.0.0.0')
