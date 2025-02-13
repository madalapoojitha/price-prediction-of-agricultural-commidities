from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)  # Corrected here

# Load dataset (This can be updated as per your path)
def load_data():
    return pd.read_excel('updatedprices.xlsx')

# Prepare dataset for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form['state']
    district = request.form['district']
    crop = request.form['item']
    months_to_predict = int(request.form['months_to_predict'])

    # Load and filter data
    df = load_data()
    filtered_data = df.loc[(df['State/UT'] == state) & (df['District'] == district) & (df['Commodity'] == crop)]

    if filtered_data.empty:
        return render_template('result.html', error=f"No data found for {state}, {district}, {crop}. Please check the inputs.")
    
    # Extract and preprocess prices
    prices = filtered_data.iloc[:, 5:].values.flatten()
    prices = prices[~np.isnan(prices)].astype('float32')

    if len(prices) == 0:
        return render_template('result.html', error="No valid prices found for the selected inputs.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

    # Prepare data for LSTM
    look_back = 24
    trainX, trainY = create_dataset(prices_scaled, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, look_back), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)

    # Forecast future prices
    future_inputs = prices_scaled[-look_back:]
    future_inputs = np.reshape(future_inputs, (1, 1, look_back))
    predicted_prices = []

    for _ in range(months_to_predict):
        future_price = model.predict(future_inputs)
        predicted_prices.append(scaler.inverse_transform(future_price)[0, 0])
        future_inputs = np.append(future_inputs[0, 0, 1:], future_price)
        future_inputs = np.reshape(future_inputs, (1, 1, look_back))

    predicted_prices = np.array(predicted_prices)

    # Calculate MSE for training data
    train_predict = model.predict(trainX)
    train_predict = scaler.inverse_transform(train_predict)
    trainY_inverse = scaler.inverse_transform([trainY])
    mse = mean_squared_error(trainY_inverse[0], train_predict[:, 0])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(prices)), scaler.inverse_transform(prices_scaled), label='Historical Prices', marker='o', linestyle='-')
    plt.plot(range(len(prices), len(prices) + months_to_predict), predicted_prices, label='Predicted Prices', marker='x', linestyle='--')
    plt.xlabel('Month Number')
    plt.ylabel(f'{crop} Prices (Rs/kg)')
    plt.title(f'Historical and Predicted {crop} Prices for {district}, {state}')
    plt.legend()

    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)  # Added transparent background to the graph
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

    return render_template('result.html', 
                           predicted_prices=predicted_prices, 
                           months=months_to_predict, 
                           mse=mse, 
                           graph_url=plot_url, 
                           state=state, 
                           district=district, 
                           crop=crop,
                           months_entered=months_to_predict)

if __name__ == '__main__':  # Corrected here
    app.run(debug=True)
