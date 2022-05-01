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

## Methodologies
### Machine Learning - Decision Tree Ensembles
Codes implementing decision trees are stored in the folder _MachineLeanring_. There are four main code scripts:  
&emsp;&emsp;(1) **GBDT_model_ns.ipynb**; (2) **GBDT_model_s.ipynb**; (3) **Xgboost_model_ns.ipynb**; (4) **Xgboost_model_s.ipynb**.  

The first two scripts build GBDT models, and the other scripts two build Xgboost models. File name ended with "ns" indicates "no shulffling", which means training dataset is not shuffled but directly used in its original time order. File name ended with "s" indicates "shulffling", which means training dataset is shuffled.

In the folder _MachineLeanring_, we have another folder _Models_ that stores trained gbdt models. Xgboost models are not stored as we go with gbdt as our first choice and compare its perfomance with deep learning methods, but you can always train and save xgboost models with the provided code scripts. To train and save models, remember to adjust the following parameters:  
```
save_model = True
grid_search = True    # Optional

# If gird_search == True, it will try to find optimal hyperparameter combination. 
# If gird_search == False, it will go with default hyperparameter combination.
```

The script **predict_strategy_compare.ipynb** compares results of different training strategies. Generally, it compares peformance between:
(1) gbdt & xgboost models; (2) predict future price & predict difference between current price and future price; (3) with Twitter data & without Twitter data.

