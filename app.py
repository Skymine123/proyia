#!/usr/bin/env python
# coding: utf-8
##Permite leer el archivo .ipynb
# In[1]:
from flask import Flask
import numpy as np
import pickle
import os
import joblib
# In[2]:
app = Flask(__name__)
# In[2]:
##Crea funcion para llamar al modelo
def predecir_Salario(rating, founded, sector, ownership, job_title, job_in_headquarters, job_seniority, job_skills):
 
 absolutepath = os.path.abspath(__file__)
 fileDirectory = os.path.dirname(absolutepath)
  
 prediction_input = list()
 print(fileDirectory)

 with open(fileDirectory+"\\sc_rating.pkl", "rb") as file:
  sc_rating = pickle.load(file) 

 with open(fileDirectory+"\\sc_founded.pkl", "rb") as file:
  sc_founded = pickle.load(file) 


 prediction_input.append(sc_rating.transform(np.array(rating).reshape(1, -1)))
 prediction_input.append(sc_founded.transform(np.array(founded).reshape(1, -1)))  

 sector_columns = ['sector_Health Care','sector_Business Services','sector_Information Technology']
 temp = list(map(int, np.zeros(shape=(1, len(sector_columns)))[0]))
 for index in range(0, len(sector_columns)):
    if sector_columns[index] == 'sector_' + sector:
      temp[index] = 1
      break
 prediction_input = prediction_input + temp


 if ownership == 'Private':
    prediction_input.append(1)
 else:
    prediction_input.append(0)
  

 job_title_columns = ['job_title_data scientist', 'job_title_data analyst']
 temp = list(map(int, np.zeros(shape=(1, len(job_title_columns)))[0]))
 for index in range(0, len(job_title_columns)):
    if job_title_columns[index] == 'job_title_' + job_title:
      temp[index] = 1
      break
 prediction_input = prediction_input + temp


 prediction_input.append(job_in_headquarters)


 job_seniority_map = {'other': 0, 'jr': 1, 'sr': 2}
 prediction_input.append(job_seniority_map[job_seniority])


 temp = list(map(int, np.zeros(shape=(1, 4))[0]))
 if 'excel' in job_skills:
    temp[0] = 1
 if 'python' in job_skills:
    temp[1] = 1
 if 'tableau' in job_skills:
    temp[2] = 1
 if 'sql' in job_skills:
    temp[3] = 1
 prediction_input = prediction_input + temp
 

 modelo = joblib.load(fileDirectory+'\\random_forest.joblib') 

 return modelo.predict([prediction_input])[0]

# In[ ]:
@app.route('/')
def pred():
 ##datos=sys.argv
 ##res=predecir_Isla(datos[1],datos[2],datos[3],datos[4],datos[5],datos[6],datos[7],datos[8],datos[9],datos[10],datos[11],datos[12],datos[13])
 salary=predecir_Salario(3.8, 1893, 'Health Care', 'Nonprofit Organization', 'Data Analyst', 1, 'sr', ['python', 'sql', 'tableau'])
 texto='Salario Estimado (rango): {}(USD) a {}(USD) por año.'.format((int(salary*1000)-9000), (int(salary*1000)+9000))
 print(texto)
 return str(texto)
#In[ ]:
if __name__ == '__main__':
    app.run()


#In[ ]:
