from stockify.entity.artifact_entity import DataTransformationArtifact , DataIngestionArtifact
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
import json
import pandas_ta as pta


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self,
                data_transformation_artifact:DataTransformationArtifact,
                data_ingestion_artifact: DataIngestionArtifact
                ):

        self.data_transformation_artifact = data_transformation_artifact
        self.data_ingestion_artifact = data_ingestion_artifact    
    
    def LSTM_model(self, time_step =6):
        csv_files = [os.path.join(os.getcwd(), "raw_data", file) for file in os.listdir("raw_data") if file.endswith("data.csv")]   
        
        def X_Y(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step):
                a = dataset[i:(i + time_step), :]
                dataX.append(a)
                dataY.append(dataset[i + time_step, :])  
            return np.array(dataX), np.array(dataY)

        def get_recommendation(model, xtest, ytest, scaler, last_day_price):
            pred = model.predict(xtest)
            
            last_sequence = X_test[-1]
            # Reshape the data to match the input shape of the model
            last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
            # Use the model to predict the next day's 'Open' and 'Close' prices
            next_day_pred = model.predict(last_sequence)
            # Inverse scale the predicted values to get the prices in the original data scale
            next_day_pred_original_scale = scaler.inverse_transform(next_day_pred)

            next_hour_open_price = next_day_pred_original_scale[0, 0]  
            next_hour_high_price = next_day_pred_original_scale[0, 1] 
            next_hour_low_price = next_day_pred_original_scale[0, 2] 
            next_hour_close_price = next_day_pred_original_scale[0, 3] 

            accuracy = r2_score(ytest, pred)
            current_price = last_day_price[-1]
            
            def arrow_indicator(next_hour_close_price, current_price):
                if next_hour_close_price > current_price:
                    return "↑ Up"  # Up arrow
                else:
                    return "↓ Down"  # Down arrow
                
            recommendation = arrow_indicator(next_hour_close_price, current_price)
            

            return current_price, recommendation, accuracy, round(float(next_hour_close_price),2)

        result_data = []
        for file_path in csv_files:
            input_filename = os.path.basename(file_path).split(".")[0]
            df = pd.read_csv(file_path)
            # df['Datetime'] = pd.to_datetime(df['Datetime']).dt.date

            closedf = df[['Open', 'High', 'Low', 'Close']]

            # Normalize all columns in the DataFrame except for 'Date'
            scaler = MinMaxScaler(feature_range=(0, 1))
            closedf = scaler.fit_transform(closedf)  

            training_size = int(len(closedf) * 0.70)
            test_size = len(closedf) - training_size
            
            train_data = closedf[0:training_size, :]
            test_data = closedf[training_size:len(closedf), :]
            
            X_train, y_train = X_Y(train_data, time_step)
            X_test, y_test = X_Y(test_data, time_step)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 4)
            
            # Clear any existing models from memory
            tf.keras.backend.clear_session()

            model = Sequential()
            model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 4)))
            model.add(Dropout(0.2))  
            model.add(LSTM(32, return_sequences=True))
            model.add(Dropout(0.2)) 
            model.add(LSTM(32))
            model.add(Dropout(0.2)) 

            model.add(Dense(4))

            model.compile(loss='mean_squared_error', optimizer='adam')

            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, verbose=1)

            current_price, recommendation, accuracy,next_hour_close_price = get_recommendation(model, X_test, y_test, scaler, df['Close'].values)
            result_data.append({
                "stock_ticker": input_filename,
                "current_price": round(current_price,2),
                "recommendation": recommendation,
                "accuracy": f"{round(accuracy,2)*100}%",
                "next_hour_close_price": next_hour_close_price
            })    

        json.dump(result_data, open('Output/result_data.json', 'w'))
        return result_data


    def FinBert(self):
        model_name = "ProsusAI/finbert"  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        csv_files = self.data_ingestion_artifact.news_data
        try:
            with open('Output/news_sentiment.json', 'r') as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = []
        
        for file in csv_files:
            df = pd.read_csv(file,nrows=3)
            df_array = np.array(df)
            df_list = list(df_array[:, 3])
            inputs = tokenizer(df_list, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            max_probs, max_labels = torch.max(predictions, dim=1)
            max_labels = max_labels.tolist()

            labels = ["Positive", "Negative", "Neutral"]
            max_labels = [labels[label] for label in max_labels]
            
            input_filename = os.path.basename(file).split(".")[0]
            new_data = {
                "Stock_ticker" : input_filename,
                'Headline': df_list,
                "Max_Probability_Value": max_probs.tolist(),
                "Max_Probability_Label": max_labels,
            }
            found_ticker = False
            for item in existing_data:
                if item['Stock_ticker'] == input_filename:
                    for new in new_data['Headline']:
                        if new not in item['Headline']:  # Only add if it's not already in existing headlines
                            item['Headline'].append(new)
                            item['Max_Probability_Value'].append(new_data['Max_Probability_Value'][df_list.index(new)])
                            item['Max_Probability_Label'].append(new_data['Max_Probability_Label'][df_list.index(new)])
                    found_ticker = True
                    break

            if not found_ticker:
                existing_data.append(new_data)

        with open('Output/news_sentiment.json', 'w') as json_file:
            json.dump(existing_data, json_file)

        return existing_data
    
    
    def stock_data(self, years=1, interval="1h"):
        self.tickers = ["TCS.NS", "HDFCBANK.NS", "LT.NS", "TITAN.NS"]
        self.end_date = pd.Timestamp.now()
        self.start_date = self.end_date - pd.DateOffset(years=years)
    
        os.makedirs("raw_data", exist_ok=True)
        for ticker_symbol in self.tickers:
            self.stock_data = yf.download(ticker_symbol, start=self.start_date, end=self.end_date, interval=interval)
            filename = f"raw_data/{ticker_symbol.split('.')[0]}_data.csv"
            self.stock_data.to_csv(filename, encoding="utf-8")
        print("Data has been downloaded successfully")
            
            
    def ichimoku_recommendation(self, TS=12, KS=24, SS=120, CS=24, OS=0):
        csv_files = [os.path.join(os.getcwd(), "raw_data", file) for file in os.listdir("raw_data") if file.endswith("data.csv")]
        signals =[]
        for file_path in csv_files:
            input_filename = os.path.basename(file_path).split(".")[0]
            dataframe = pd.read_csv(file_path)
            dataframe['Datetime'] = pd.to_datetime(dataframe['Datetime']).dt.date
            dataframe['TenkanSan'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['ITS_12']
            dataframe['Kijun'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['IKS_24']
            dataframe['SenkanA'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['ISA_12']
            dataframe['SenkanB'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['ISB_24']
            dataframe['Chinkou'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['ICS_24']

            def trade_condition(dataframe):
                    signal_position = []
                    for i in range(len(dataframe)):
                        if (dataframe['Close'][i] > dataframe['TenkanSan'][i]) & (dataframe['Close'][i] > dataframe['Kijun'][i]):
                            signal_position.append('long')
                        elif  (dataframe['Close'][i] < dataframe['TenkanSan'][i]) & (dataframe['Close'][i] < dataframe['Kijun'][i]):
                            signal_position.append('short')
                        else:
                            signal_position.append('neutral')

                    return signal_position
            dataframe['position'] = trade_condition(dataframe)
            
            def trade_crossover(dataframe):
                long_crossover = []
                short_crossover = []
                marker = 0

                for i in range(len(dataframe)):
                    if dataframe['TenkanSan'][i] > dataframe['Kijun'][i]:
                        if (marker != 1):
                            long_crossover.append(dataframe['Close'][i])
                            short_crossover.append(np.NaN)
                            marker = 1
                        
                        else:
                            long_crossover.append(np.NaN)
                            short_crossover.append(np.NaN)

                    elif dataframe['TenkanSan'][i] < dataframe['Kijun'][i]:
                        if (marker != -1):
                            short_crossover.append(dataframe['Close'][i])
                            long_crossover.append(np.NaN)
                            marker = -1
                        
                        else:
                            long_crossover.append(np.NaN)
                            short_crossover.append(np.NaN)
                    else:
                        long_crossover.append(np.NaN)
                        short_crossover.append(np.NaN)

                return long_crossover, short_crossover
            crossover=trade_crossover(dataframe)
            dataframe['long_crossover']=crossover[0]
            dataframe['short_crossover']=crossover[1]
            def create_signal(dataframe):
                signal = []
                for i in range(len(dataframe)):
                    if pd.notnull(dataframe['long_crossover'][i]) & (dataframe['position'][i] == 'long'):
                        signal.append('buy')
                    elif pd.notnull(dataframe['short_crossover'][i]) & (dataframe['position'][i] == 'short'):
                        signal.append('sell')
                    else:
                        signal.append('hold')

                return signal
            signal=create_signal(dataframe)
            dataframe['Signal']=signal
            df_cleaned = dataframe.dropna(subset=['long_crossover'])
            last = df_cleaned.tail(1)
            indicator = last["Signal"].values[0]
            
            signals.append({
                "stock_ticker": input_filename,
                "Signal" : indicator
                
            })
            
            
        
        json.dump(signals, open(f'Output/signals.json', 'w'))            
        return signals



        





