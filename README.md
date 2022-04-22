# PricePredictionForBinance



# DataRetrieval_PlotPrediction.py:

This script retrieves twitter and Binance data and write it into input_path(default: "input/data.csv") every minute; 

then it waits spark to output the prediction into output_path(default: "output/data.csv"), where the prediction column name should be "close";

then it plots the close price in the past 60 minutes and the predicted future close price in 5 minutes every minute.


### Adjustable parameters:

input_path(="input/data.csv"): where this script wrtites twitter and Binance data (The price should be in chronological order from top to bottom, i.e., oldest on the first row and newest on the last row)

output_path(="output/data.csv"): where spark writes predicted price (The price should be in chronological order from top to bottom, i.e., prediction for 1st minute after on the first row and prediction for 5th minute after on the last row)

wait(=5): the seconds needed for spark to predict


----------------------------------
