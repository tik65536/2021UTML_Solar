#!/bin/python3
import pandas as pd
import pickle
from datetime import datetime
from datetime import timedelta

def hourlyData(date,hour):
  lang = "en"
  mindate = "30.01.2004"
  maxdate = datetime.today().strftime('%d.%m.%Y')
  #hour = 14
  url = fr'https://www.ilmateenistus.ee/ilm/ilmavaatlused/vaatlusandmed/tunniandmed/?lang={lang}&filter[minDate]={mindate}&filter[maxDate]={maxdate}&filter[date]={date}&filter[hour]={hour}'
  tables = pd.read_html(url) # Returns list of all tables on page
  return tables[0]

def cloudCoveraghourlyData(date,hour):
  lang = "en"
  mindate = "30.01.2004"
  maxdate = datetime.today().strftime('%d.%m.%Y')
  # Example :
  # lang=en&filter%5BmaxDate%5D=03.11.2021&filter%5BminDate%5D=30.01.2004&filter%5Bdate%5D=01.10.2021&filter%5Bhour%5D=23
  url = fr'https://www.ilmateenistus.ee/ilm/ilmavaatlused/vaatlusandmed/pilved/?lang={lang}&filter[minDate]={mindate}&filter[maxDate]={maxdate}&filter[date]={date}&filter[hour]={hour}'
  tables = pd.read_html(url) # Returns list of all tables on page
  return tables[0]


date_20190101=1546272000

day = datetime.fromtimestamp(date_20190101)

for i in range(1095):
    for j in range(0,24):
        print(f'Get cloud coverage :{day} {j}\r',end='')
        t=cloudCoveraghourlyData(day.strftime('%d.%m.%Y'),j)
        t['time']=i
        f=open('./cloudCoverage_'+day.strftime('%d.%m.%Y')+'_'+str(j),'wb')
        pickle.dump(t,f)
    day = day + timedelta(days=1)

