import pandas as pd
from glob import glob
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 讀取檔案
files = glob('CoinText_final.csv')

# 合併檔案
df = pd.concat((pd.read_csv(file, usecols=['CoinDate','CoinText'], dtype={ 'CoinDate': str, 'CoinText':str}) for file in files))
df.reset_index()

# 建立日期迴圈
start = datetime.datetime.strptime("2022/1/01", "%Y/%m/%d")
end = datetime.datetime.strptime("2022/12/31", "%Y/%m/%d")
date_generated = pd.date_range(start, end)


# 載入模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

#建立情緒分數清單
CionScore=[]
Cion_score=0
CoinDate_n=[]
PCion_score=0
ScorePositive=[]
ScoreNegative=[]
ScoreNeutral=[]

positive_score=0
negative_score=0
neutral_score=0

i=0

# 設定要分類的標籤
labels = ['positive', 'negative', 'neutral']



for context_date in date_generated:
    #篩選時間
    mask1 = (df["CoinDate"] == str(context_date.strftime("%Y/%m/%d")))
    if len(df[mask1])==0:      
        CionScore.append(PCion_score*0.99)
        CoinDate_n.append(context_date.strftime("%Y/%m/%d"))
    else:
        for Context in df[mask1]['CoinText']:
            # 將文本轉換為tokens並加入特殊tokens
            inputs = tokenizer(str(Context), return_tensors='pt', padding=True, truncation=True)

            # 進行預測
            outputs = model(**inputs)
            predicted_scores = torch.softmax(outputs.logits, dim=1)
            ScorePositive.append()
            float(predicted_scores[0][2])

            # 挑選預測中得分最高的情緒類別
            max_score_idx = torch.argmax(predicted_scores)
            max_score_label = labels[max_score_idx]
            max_score = predicted_scores[0][max_score_idx]

            positive_score+=float(predicted_scores[0][0])
            negative_score+=float(predicted_scores[0][1])
            neutral_score+=float(predicted_scores[0][2])


        CoinDate_n.append(context_date.strftime("%Y/%m/%d"))
        ScorePositive.append(positive_score/len(df[mask1]))
        ScoreNegative.append(negative_score/len(df[mask1]))
        ScoreNeutral.append(neutral_score/len(df[mask1]))
        ScorePositive=0
        ScoreNegative=0
        ScoreNeutral=0

        
    i+=1
    if ((i%100)==0) &(i>0):
        lo={'CionScore':CionScore,'CoinDate':CoinDate_n}
        dfz = pd.DataFrame(lo)
        dfz.to_csv('Data_score/CionScore_'+str(i)+'.csv')
        print(str(i))


lo={'CionScore':CionScore,'CoinDate':CoinDate_n}
dfz = pd.DataFrame(lo)
dfz.to_csv('Data_score/CionScore_final.csv')