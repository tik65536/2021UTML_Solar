import pickle
import pandas as pd
import os
import numpy as np
from datetime import datetime

begin_d = 20
begin_m = 11
begin_y = 2019

end_d = 20
end_m = 11
end_y = 2021

url = "https://meteo.physic.ut.ee/en/archive.php?do=data&begin%5Byear%5D=" + str(begin_y) + "&begin%5Bmon%5D=" + str(begin_m) + "&begin%5Bmday%5D=" + str(begin_d) + "&end%5Byear%5D=" + str(end_y) + "&end%5Bmon%5D=" + str(end_m) + "&end%5Bmday%5D=" + str(end_d) + "&14=1&ok=+Query+"
    
df = pd.read_csv(url)
df[['Date', 'Time']] = df['Times'].str.split(' ', 1, expand=True)
df = df.drop('Times', 1)
df = df.reindex(columns=['Date', 'Time', ' Irradiation flux'])
df = df.rename(columns={' Irradiation flux': 'Irradiation flux'})

df.to_csv('if_data_' + df.iloc[0,0] + '_' + df.iloc[-1,0] + '.csv')