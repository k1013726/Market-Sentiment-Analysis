我寫的LSTM用以預測 BTC-all.arff

請以命令列指令執行

可調整參數為
- training_ratio, 設定訓練資料比例, 例如設成0.9, 0.8, 0.7分別代表保留0.1, 0.2, 0.3當測試
- look_back, 設定由前幾期來預測本期資料, 例如設成 7是以7日資料來預測第 8 日
- batch_size, 深度神經網路訓練時的批次大小, 目前預設設成 4

命令列指令如下：

python 'Cryptocurrency forecasting with LSTM.py' -tr 0.8 -lb 7 -bs 4 > LSTMtr08lb7bs4.txt

程式中有用到一些套件可能都要另外安裝

- tensorflow 2.12
- keras 2.12
- keras_tuner 
- statsmodels==0.14.0

有一些統計檢定(test)我有安裝後來註解掉的, 你可以查看原始程式碼來瞭解

keras_tuner 會協助找到比較好的參數, 會存放在特定目錄, 重複執行時會詢問你是否要刪除該目錄

直接按 Enter 或按 y 再按 Enter就會刪除

如果按 n則是不刪除, 會直接用上次的參數速度會比較快

雖然有自動參數, 但目前我是用它的自動參數稍加修改來取代

也許還不是最好的但目前得到的是
<pre>
training_ratio=0.9
train MSE: 648411.50, RMSE: 805.24, MAE: 360.88
test MSE: 321746.77, RMSE: 567.23, MAE: 383.14
Total time: 2.0 minutes, 52.37 seconds.

training_ratio=0.8
train MSE: 538925.64, RMSE: 734.12, MAE: 296.88
test MSE: 1033367.17, RMSE: 1016.55, MAE: 643.62
Total time: 2.0 minutes, 31.94 seconds.

...</pre>

由於你之前LSTM一直執行不出理想的結果, 因此先提供這部份程式供你參考. 如目錄中的檔案.

你使用看看是否有問題.

由於這部份我花比較多時間且也比較關鍵, 因此這篇論文我會掛第一位作者, 還請理解
