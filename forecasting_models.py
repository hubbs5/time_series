# Time Series Forecasting Models
import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

def index_ext(index, forecast_horizon):
	try:
		if np.issubdtype(index[-1], np.datetime64):
			# Assuming days are the proper time frame
			day_index = index.astype("datetime64[D]")
			forecast_index = np.array([day_index.max() + i 
				for i in range(1, 1 + forecast_horizon)])
	except TypeError:
		if isinstance(index[-1], datetime):
			forecast_index = np.array([index.max() + timedelta(i) 
				for i in range(1, 1 + forecast_horizon)])
		else:
			max_index = index.max()
			forecast_index = np.array([max_index + i 
				for i in range(1, 1 + forecast_horizon)])
	return forecast_index

def naive_forecast(X, forecast_horizon):
	y_hat = np.repeat(X[-1], forecast_horizon)
	return y_hat

def drift_forecast(X, forecast_horizon):
	m = (X[-1] - X[0]) / len(X)
	y_hat = np.array([X[-1] + m * i
		for i in range(1, 1 + forecast_horizon)])
	return y_hat

def average_forecast(X, forecast_horizon):
	y_hat = np.repeat(X.mean(), forecast_horizon)
	return y_hat

def linear_reg_forecast(X, forecast_horizon):
	linreg = LinearRegression()
	linreg.fit(np.arange(len(X)).reshape(-1,1), X)
	y_hat = linreg.predict(np.arange(len(X), len(X) + forecast_horizon))

	return y_hat


