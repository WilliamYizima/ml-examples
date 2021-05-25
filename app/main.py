from flask import Flask, jsonify, request
import pickle
import numpy as np

PORT =  80
model_name = 'teste'
app = Flask(__name__)


@app.route('/analyze', methods=["POST"])
def analyze():
    try:
        payload = request.json
        sqft_lot = payload['sqft_lot']
        
        model_name = payload['model_name']
        
        x_feature = np.array([7242]).reshape(1,1)
        
        modelo_carregado = pickle.load(
                                    open(
                                        f'./models/{model_name}.sav','rb'
                                        ))
        predict = modelo_carregado.predict(x_feature)[0]

        return jsonify({'price_estimated':str(predict),'sqft_lot': sqft_lot}), 200
    except Exception as e:
        return jsonify({'message':e}), 400


if __name__ == '__main__':
    print(f'estou rodando na porta {PORT} 👹🌸')
    app.run(port=PORT, host='0.0.0.0')
