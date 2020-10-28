import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from keras.models import load_model
#from tensorflow.keras.models import load_model



app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict_model():

    # Read in the dictionary data from the curl request
    test = request.get_json(force=True)

    # Convert to dataframe
    test = pd.DataFrame.from_dict(test)
    test.reset_index(inplace=True)
    test = test.rename(columns = {'index':'model_variables'})

    """
    remove all variables except 'price_high'
    """
    test = test[test.model_variables.str.startswith('price_high_')]


    API_Output = test

    # Sort date in increasing time
    API_Output['sort'] = API_Output['model_variables'].str.extract('(\d+)', expand=False).astype(int)
    API_Output.sort_values('sort', inplace=True, ascending=True)
    API_Output = API_Output.drop('sort', axis=1)


    # transpose the dataframe
    API_Output = API_Output.T

    # Set the proper column names
    API_Output.columns = API_Output.iloc[0]
    API_Output = API_Output.drop(API_Output.index[0])

    # reshape input to be [samples, time steps, features] which is required for LSTM
    API_Output = API_Output.values.reshape(API_Output.shape[0], API_Output.shape[1], 1).astype('float32')


    # Passing data to model & loading the model from disk
    model = load_model(
        "model_LSTM.h5",
        custom_objects=None,
        compile=False
    )

    # Given,the input data, predict the results from the model
    API_Output = pd.DataFrame(model.predict(API_Output)).rename(columns={0: 'bitcoin_prediction'})
    # API_Output.set_index("0", inplace=True, drop=True)


    # Return the forecasted bitcoin price
    return jsonify(API_Output.to_dict(orient='records'))

# set the port to 8020
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8020)
