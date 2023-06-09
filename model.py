import os
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.preprocessing import MinMaxScaler
import japanize_matplotlib
import base64
from io import BytesIO
from matplotlib.figure import Figure as fig
from matplotlib.figure import Figure


def carrot(predict_day):
    #データの読み込み
    def preprocess(vegetable):
        current_path = os.path.dirname(__file__)
        df = pd.read_csv(current_path + "/price.csv", header=0, usecols=["date", vegetable])

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        df = df.interpolate()

        original_data = df.values

        return original_data

    #モデルの定義
    def create_model(hidden_size, batch_size=None, time_steps=None, stateful=False):

        inputs = Input(batch_shape=(batch_size, None, 1))
        x = inputs
        x = LSTM (hidden_size, stateful=stateful, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        x = LSTM (hidden_size, stateful=stateful, return_sequences=False)(x)
        x = Dropout(0.5)(x)
        x = Dense(1, kernel_regularizer=l2(0.0001), activation='relu')(x)
        outputs = x

        model = Model(inputs=inputs, outputs=outputs)

        return model

    #パラーメタの設定と前処理
    # hyper parameters
    #training = True     # 訓練モード or 予測モード
    train_rate = 0.8    # 訓練データの割合
    time_steps = 30    # 一度の予測に使うデータ数

    predict_day = int(predict_day)    # 予測する日数

    hidden_size = 300    # 隠れ層の大きさ
    batch_size = 30     # バッチサイズ
    epochs = 50        # エポック数

    # データの読み込み
    current_path = os.path.dirname(__file__)
    original_data = preprocess(vegetable='carrot')

    # 訓練データとテストデータの分割と正規化   
    border = math.floor(len(original_data) * train_rate)
    train = original_data[:border]
    test  = original_data[border:]

    train_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaler.fit(train)

    scaled_train = train_scaler.transform(train)
    scaled_test  = train_scaler.transform(test)    # テストデータを正規化の基準に入れてはならない

    # 学習モード
    #if training:
    # 入力データとラベルデータの作成
    data_size = len(scaled_train) - time_steps - predict_day       
    x_train = np.zeros((data_size, time_steps))                    
    y_train = np.zeros((data_size, 1))                             
    for i in range(data_size):
        x_train[i] = scaled_train[i:i + time_steps].T              
        y_train[i] = scaled_train[i + time_steps + predict_day]    

    data_size = len(scaled_test) - time_steps - predict_day        
    x_test = np.zeros((data_size, time_steps))                     
    y_test = np.zeros((data_size, 1))                              
    for i in range(data_size):
        x_test[i] = scaled_test[i:i + time_steps].T                
        y_test[i] = scaled_test[i + time_steps + predict_day]      

    # モデルの入力に形状を合わせる
    x_train = x_train.reshape(len(x_train), time_steps, 1)  
    x_test  = x_test.reshape(len(x_test), time_steps, 1)    
    y_train = y_train.reshape(len(y_train), 1, 1)         
    y_test  = y_test.reshape(len(y_test), 1, 1)

    model = create_model(hidden_size=hidden_size,
                            time_steps=time_steps,    
                            stateful=False)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error", metrics=["mean_absolute_percentage_error"])
    checkpointer = ModelCheckpoint(filepath=current_path + '/best_model.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
    csv_logger   = CSVLogger(current_path + '/history.log')

    # 学習
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1,
                        shuffle=False,
                        callbacks=[csv_logger, checkpointer])

    # 学習の評価
    eval = model.evaluate(x=x_test,
                            y=y_test,
                            batch_size=batch_size,
                            verbose=1)
    #print('MAPE:{} %'.format(eval[1]))

    # 予測モード
    #else:
    model_sf = create_model(hidden_size=hidden_size,
                            batch_size=1,
                            stateful=True)           
    model_sf.load_weights(current_path + '/best_model.hdf5')

    # 状態を保存するため、最初100個のデータで予測(ここでの予測結果は使わない)
    inputs_pre = scaled_test[:time_steps]
    inputs_pre = inputs_pre.reshape(1, time_steps, 1)
    model_sf.predict(inputs_pre)

    # 100+30個目までは予測できないため、真値を入れておく
    predicted  = scaled_test[:time_steps+predict_day] 

    # 30日後までの予測
    # 予測開始
    i = 0
    while len(predicted) < len(scaled_test) + predict_day:  
        input_data = scaled_test[time_steps+i]
        input_data = input_data.reshape(1, 1, 1)
        score      = model_sf.predict(input_data)
        predicted  = np.append(predicted, score[0,0])
        i += 1

        # 自己回帰による追加予測
        # j = 0
        # while len(predicted) < len(scaled_test) + predict_day + 500:  
        #     input_data = predicted[-predict_day]
        #     input_data = input_data.reshape(1, 1, 1)
        #     score      = model_sf.predict(input_data)
        #     predicted  = np.append(predicted, score[0,0])
        #     j += 1

    predicted = train_scaler.inverse_transform(predicted.reshape(-1, 1)) 

    # 予測の評価
    y_pred = predicted[time_steps+predict_day:-predict_day] 
    y_true = test[time_steps+predict_day:]
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    #print('MAPE: {} %'.format(mape))

    # 結果出力のためのデータフレームを作成
    day = datetime.datetime.strptime('2022-08-24', '%Y-%m-%d')    
    days = []
    for _ in range(len(predicted)):
        day += datetime.timedelta(days=1)
        day_str = day.strftime("%Y/%m/%d")
        days.append(day_str)

    df_date = pd.DataFrame(days)
    df_date.columns = ['Date']
    df_pred = pd.concat([pd.DataFrame(predicted), df_date], axis=1)
    df_pred = df_pred.set_index('Date')

    # プロットの作成
    plt.figure(figsize = (15, 7))
    plt.plot(test, label = "実際の価格")
    plt.plot(df_pred, label = "予測価格")

    plt.axvspan(0, len(test[:time_steps+predict_day]), facecolor='Cyan', alpha=0.075)
    plt.axvspan(len(test[:time_steps+predict_day]), len(test), facecolor='magenta', alpha=0.075)
    plt.axvspan(len(test), len(predicted), facecolor='yellow', alpha=0.1)

    bbox_dict = dict(boxstyle='round', facecolor='None', edgecolor='red')
    plt.text(time_steps-50, 5, 'pre_input', fontsize=16, bbox=bbox_dict)
    plt.text(len(test)-90, 5, 'eval', fontsize=16, bbox=bbox_dict)
    plt.text(len(predicted)-30, 5, 'future', fontsize=16, bbox=bbox_dict)

    plt.xlabel("日付")
    plt.ylabel("価格", fontsize=16)
    plt.xticks(np.arange(0, len(df_pred), 90), rotation=45)
    plt.title("にんじんの価格", fontsize=16)
    plt.legend(loc='upper left', fontsize=16)

    #画像をバッファに保存
    buf = BytesIO()
    plt.savefig(buf, format="png")
    
    #画像データをBase64にエンコード
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return (predicted[-1] / 0.70), data
