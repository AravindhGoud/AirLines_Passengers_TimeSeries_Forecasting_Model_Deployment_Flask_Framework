import pandas as pd
import numpy as np 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle

df=pd.read_csv(r"C:\Users\aravi\OneDrive\Desktop\ExcelR Assignments\18. Forecasting\Airlines+Data.csv")

df['Month']=pd.to_datetime(df['Month'],format='%b-%y' )

# Set the 'Month' column as the index and specify the frequency as 'MS'
df.set_index('Month', inplace=True)
df.index.freq = 'MS'

print(df)
df['Log_Passengers']=np.log(df['Passengers'])

hwe_model_mul_add = ExponentialSmoothing(df["Log_Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit() 

with open('model.pkl','wb') as f:
    pickle.dump(hwe_model_mul_add,f)

print(np.exp(hwe_model_mul_add.forecast(1)))
