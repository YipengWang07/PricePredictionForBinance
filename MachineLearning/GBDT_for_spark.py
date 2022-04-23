import numpy as np
import pandas as pd
import joblib
import os
import sys

# GBDT module
class GBDT_Module():
    def __init__(self, model_root_path: str, max_pred_pength: int = 5):
        """
        Initialization

        :param model_root_path: the root path that stores models
        :param max_pred_pength: the maximum length we want to predict
        """

        self.m = max_pred_pength
        self.model_list = list()
        if not model_root_path.endswith("/"): model_root_path = model_root_path + "/"

        for i in range(1, self.m + 1):
            model_full_path = model_root_path + "gbdt_" + str(i) + "_minutes_later.pkl"
            # See if the model exist, if not, stop searching
            if not os.path.exists(model_full_path):
                print("[Warning]: Cannot find model with predict length %d, further searches are ignored." % i)
                break
            # If exists, add the model to the model list
            self.model_list.append(joblib.load(model_full_path))

    def predict_with_one_sample(self, input_features: list) -> list:
        """
        Predict close price 1~5 minutes in future

        :param input_features: the input features of the one sample, it should have length of 12, which gives information about
                                [open_time, wgtavg, avg, open, high, low, close, volume, quote_asset_volume,
                                number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume]
        :return: a list with size 5, value at index 0 represents price 1min later, value at index 1 represents price 2mins later...
        """
        input_features_array = np.asarray(input_features)[1:].reshape(1, -1)
        print(input_features_array)
        return [(self.model_list[i].predict(input_features_array) + input_features[6])[0] for i in range(self.m)]

    def predict_with_one_hour_data(self, history_data: pd.DataFrame) -> list:
        """
        Predict close price 1~5 minutes in future

        :param history_data: values in the past hour, it should have shape of (60, 12), each row has information about
                                [open_time, wgtavg, avg, open, high, low, close, volume, quote_asset_volume,
                                number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume]
        :return: a list with size 5, value at index 0 represents price 1min later, value at index 1 represents price 2mins later...
        """
        weights = [(self.m - i) / np.sum(np.arange(self.m + 1)) for i in range(self.m)]
        price_pred = []

        for i in range(self.m):
            cur_sum, cur_portion = 0, 0
            for j in range(self.m):
                if j + self.m - i > 0:
                    cur_sum += weights[j] * (self.model_list[j].predict(
                        history_data.iloc[-((j + self.m) - i), 1:].values.reshape(1, -1)) +
                                             history_data.iloc[-((j + self.m) - i)]['close'])[0]
                    cur_portion += weights[j]
            price_pred.append(cur_sum / cur_portion if cur_portion != 0 else 0)

        return price_pred


if __name__ == "__main__":
    # Load the data
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Data/BTC-0408-0416.csv"
    # print(data_path)
    BTC_BUSD = pd.read_csv(data_path)

    models_path = os.getcwd() + "/Models/"
    # print(model_path)
    gbdt = GBDT_Module(model_root_path=models_path)

    # Test predict_with_one_sample
    input_features = list(BTC_BUSD.iloc[-1, :])
    print(input_features)
    res1 = gbdt.predict_with_one_sample(input_features)

    # Test predict_with_history_data
    input_df = BTC_BUSD
    res2 = gbdt.predict_with_one_hour_data(input_df)

    print("predict_with_one_sample: ", res1)
    print("predict_with_one_hour_data: ", res2)