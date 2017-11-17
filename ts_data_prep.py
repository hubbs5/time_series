# Time Series Data Prep
import numpy as np 
import pandas as pd 

def data_prep(data, forecast_horizon):
    '''
    data_preprocess takes input data in the form of a pandas.DataFrame, 
    pandas.Series, or np.array and shapes the data for processing 
    through a RNN. 
    
    =============================================================
    Inputs
    data: np.array, pd.Series or DataFrame with observations in the
    rows and dimensions in the columns
    forecast_horizon: defines the number of periods for the forecast 
    (int)
    
    =============================================================
    Returns
    x_train: batched training data for input (np.array)
    y_train: batched training data for target (np.array)
    index_train: training data index (np.array)
    '''
    # Check data format
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        # Preserve index for plotting
        index = np.array(data.index)
        if np.issubdtype(index[-1], np.datetime64):
        	# Assuming days are the proper time frame
        	index = index.astype("datetime64[D]")
        data = data.values
    else:
    	index = np.arange(0, len(data))

    # Define input data dimensions
    n_dimensions = 1
    # Shift x and y data and reshape
    x_data = data[:len(data) - forecast_horizon]
    y_data = data[forecast_horizon:len(data)]
    x_index = index[:len(x_data)]
    y_index = index[forecast_horizon:len(index)]

    return(x_data, y_data, x_index, y_index)

def data_prep_rnn(data, batch_size, forecast_horizon, split=0.75):
    '''
    data_preprocess takes input data in the form of a pandas.DataFrame, 
    pandas.Series, or np.array and shapes the data for processing 
    through a RNN. 
    
    =============================================================
    Inputs
    data: np.array, pd.Series or DataFrame with observations in the
    rows and dimensions in the columns
    batch_size: defines the number of observations to be passed to 
    the network during training (int)
    forecast_horizon: defines the number of periods for the forecast 
    (int)
    split: the percentage of data to be returned as training data (float)
    
    =============================================================
    Returns
    x_train: batched training data for input (np.array)
    x_test: batched test data for input (np.array)
    y_train: batched training data for target (np.array)
    y_test: batched test data for target 
    index_train: training data index (np.array)
    index_test: test data index (np.array)
    '''
    # Check data format
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        # Preserve index for plotting
        index = data.index
        data = data.values
        
    
    # Define input data dimensions
    n_dimensions = 1
    # Reduce length of input data to match reshape requirements
    x_data = data[:len(data) - (len(data) % batch_size)].reshape(
        -1, batch_size, n_dimensions)
    y_data = data[forecast_horizon:len(data) - (len(data) % batch_size) + forecast_horizon].reshape(
        -1, batch_size, n_dimensions)
    
    # Split into test and training sets
    split_batch = int(x_data.shape[0] * split)
    x_train = x_data[:split_batch,:,:]
    x_test = x_data[split_batch:,:,:]
    y_train = y_data[:split_batch,:,:]
    y_test = y_data[split_batch:,:,:]
    
    # Return dates for easier plotting if they exist in index
    try:
        index_train = index[:split_batch * batch_size]
        index_test = index[split_batch * batch_size:len(data) - 
                           (len(data) % batch_size)]
       
    except NameError:
        index_train = np.arange(0, split_batch * batch_size)
        index_test = np.arange(split_batch * batch_size, 
                               len(data) - (len(data) % batch_size))
    
    return (x_train, x_test, y_train, y_test, index_train, index_test)