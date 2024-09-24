import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model  # Corrected import
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

model = load_model('D:\only coding\stcok prediction\Stock Predictions Model.keras') #hear set your train model path

st.header('Stock market predictor')

stock = st.text_input('Enter stock symbol', 'GOOG')
start = '2022-01-01'
end = '2023-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

st.subheader('MA50')
ma_50_days = data['Close'].rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(ma_50_days, 'r', label='MA50')
ax1.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data['Close'].rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(ma_50_days, 'r', label='MA50')
ax2.plot(ma_100_days, 'b', label='MA100')
ax2.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data['Close'].rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(ma_100_days, 'r', label='MA100')
ax3.plot(ma_200_days, 'b', label='MA200')
ax3.plot(data['Close'], 'g', label='Close Price')
plt.legend()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1 / scaler.scale_[0]
predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(y, 'g', label='Original Price')
ax4.plot(predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
