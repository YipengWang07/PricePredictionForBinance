# PricePredictionForBinance



# DataRetrieval_PlotPrediction.py:

This script retrieves twitter and Binance data and write it into input_path(default: "input/data.csv") every minute; 

then it waits spark to output the prediction into output_path(default: "output/data.csv"), where the prediction column name should be "close";

then it plots the close price in the past 60 minutes and the predicted future close price in 5 minutes every minute.


### Adjustable parameters:

input_path(="input/data.csv"): where this script wrtites twitter and Binance data  

output_path(="output/data.csv"): where spark writes predicted price

wait(=5): the seconds needed for spark to predict


----------------------------------
