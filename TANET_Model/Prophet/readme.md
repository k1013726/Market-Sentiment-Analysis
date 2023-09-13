Prophet 必須要調整參數及加人前幾期的regressor才能有比較好的測試結果

我花了不少時間在上面, 現終於解決...	

目前(BTC-all.arff)測試資料(0.1比例)的MAE已從7千多降至 4百多 (我跑出的是431.54，但windows下跑會高一些，好像是457.xx)

要留意我的程式是用 prophet 1.1.4而非 fbprophet
而且你的 matplotlib 也需更新至 3.4.1

程式已上傳至github, 
第一個程式Cryptocurrency forecasting with Prophet
- 純粹由調整週期及傅立葉次數來獲得最佳參數

第二個程式Cryptocurrency forecasting with Prophet
- 除調整週期及傅立葉次數來獲得最佳參數外, 另加入前三期的資料當regressor

你可以由修改資料檔來獲取其它資料集的結果，測試資料的比例可以透過修改 training_ratio來調整。
