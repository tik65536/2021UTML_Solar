#!/bin/python3
import pandas as pd
import pickle
from datetime import datetime
from datetime import timedelta

def hourlyData(maxdate,hour):
  lang = "en"
  mindate = "30.01.2004"
  #maxdate = "25.10.2021"
  #hour = 14
  url = fr'https://www.ilmateenistus.ee/ilm/ilmavaatlused/vaatlusandmed/tunniandmed/?lang={lang}&filter[minDate]={mindate}&filter[maxDate]={maxdate}&filter[hour]={hour}'
  tables = pd.read_html(url) # Returns list of all tables on page
  return tables

def cloudCoveraghourlyData(maxdate,hour):
  lang = "en"
  mindate = "30.01.2004"
  #maxdate = "25.10.2021"
  #hour = 14
  url = fr'https://www.ilmateenistus.ee/ilm/ilmavaatlused/vaatlusandmed/pilved/?lang={lang}&filter[minDate]={mindate}&filter[maxDate]={maxdate}&filter[hour]={hour}'
  tables = pd.read_html(url) # Returns list of all tables on page
  return tables

date_20190101=1563379200

day = datetime.fromtimestamp(date_20190101)

for i in range(1095):
    for j in range(0,24):
        print(f'Get cloud coverage :{day} {j}\r',end='')
        t=cloudCoveraghourlyData(day.strftime('%d.%m.%Y'),j)
        t[0]['time']=i
        f=open('./cloudCoverage_'+day.strftime('%d.%m.%Y')+'_'+str(j),'wb')
        pickle.dump(t[0],f)
    day = day + timedelta(days=1)

