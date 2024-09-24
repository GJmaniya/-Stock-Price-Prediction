# 1. app.py: Stock Market Predictor using Streamlit
This Python script creates a stock market prediction web application using Streamlit and a pre-trained Keras model for stock price forecasting. It provides the following functionalities:
# Stock Data Retrieval:

->The application retrieves historical stock data from Yahoo Finance using the yfinance library.
->Users can input a stock symbol (default is 'GOOG') and retrieve data for the period from January 2022 to December 2023.
->The app displays raw stock data, including open, close, high, low, and adjusted prices for the selected period.

# Moving Averages (MA) Visualization:
->The app calculates the 50-day (MA50), 100-day (MA100), and 200-day (MA200) moving averages for the stock’s closing price.
->It generates three separate plots:
1. MA50 vs Close Price: Shows how the stock price compares to the 50-day moving average.
2. MA50, MA100 vs Close Price: Compares the stock’s price with the 50-day and 100-day moving averages.
3. MA100, MA200 vs Close Price: Compares the stock’s price with the 100-day and 200-day moving averages.

# Stock Price Prediction:
->The script uses a pre-trained Keras model (Stock Predictions Model.keras) to predict future stock prices based on the last 100 days of closing price data.
->The data is normalized using the MinMaxScaler to scale the input between 0 and 1, which is essential for neural network model performance.
->After prediction, the results are rescaled to the original range for comparison with actual stock prices.

# Comparison of Predicted and Actual Stock Prices:
->The app displays a graph comparing the original stock prices with the predicted prices over time.
->This allows users to visually assess how closely the model's predictions align with the actual market data.

# Tech Stack:
->Libraries: yfinance (for data fetching), pandas (data manipulation), numpy (array handling), MinMaxScaler (data normalization), Keras (model prediction), matplotlib (plotting).
->Streamlit: Provides a web interface, displaying user input fields, stock data, and interactive plots.


2. pro.ipynb: Jupyter Notebook (Unknown Content)
This Jupyter notebook file likely contains code related to the stock prediction model or further analysis. Typically, notebooks in this context may include:

->Data Preprocessing: Loading and cleaning stock data.
->Exploratory Data Analysis (EDA): Visualizations and statistics related to stock price trends.
->Model Training: Code for training the Keras model used in the app.py. This could include defining a neural network architecture, fitting the model on stock data, and evaluating its performance.
->Prediction and Evaluation: Code for predicting future stock prices using the trained model, along with plots comparing predicted prices with actual prices.










