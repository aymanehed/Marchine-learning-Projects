# lab 21: prédiction du credit de logment
# Réaliser par: Rhahni Oussama - Emsi 2023 / 2024
# Ref: Code source https://www.kaggle.com/code/rodsonzepekinio/pr-vision-d-un-cr-dit-logement

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step1: Dataset/x_train, x_test, y_train, y_test
dt=pd.read_csv("datasets/train.csv")
print(dt.head())
print(dt.info())
print(dt.isna().sum())
# Nous allons  remplacer les variables manquantes categoriques par leurs  modes

def trans(data):
    for c in data.columns:
        if data[c].dtype=='int64' or data[c].dtype=='float64':
            data[c].fillna(data[c].median(),inplace=True)
        else:
              data[c].fillna(data[c].mode()[0],inplace=True)
                #modes=valeur plus frequense
                #mediane=la valeur centrale
                #moyenne
print(trans(dt))
print(dt.isna().sum())
# target exploration (loan_status)
#dt["Loan_Status"].value_counts()
print(dt)
print(dt["Loan_Status"].value_counts(normalize=True)*100)
# Data visualisation: px ou sns
#fig = px.histogram(dt, x="Loan_Status",title='Crédit accordé ou pas', color="Loan_Status",template= 'plotly_dark')
#fig.show(font = dict(size=17,family="Franklin Gothic"))
#fig = px.pie(dt, names="Dependents",title='Dependents',color="Dependents",template= 'plotly_dark')
#fig.show(font = dict(size=17,family="Franklin Gothic"))
#Property_Area
#fig = px.histogram(dt, x="Property_Area",title='Property_Area',color="Property_Area",template= 'plotly_dark')
#fig.show(font = dict(size=17,family="Franklin Gothic"))
var_num=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
#print(dt[var_num].describe())


dt['Credit_History'] = dt['Credit_History'].replace(1.0,'Yes')
dt['Credit_History'] = dt['Credit_History'].replace(0.0,'No')
#Les variables categoriques
var_cat=["Loan_Status","Gender","Married","Dependents","Education","Self_Employed","Property_Area","Credit_History"]
# Analyse bivariée (replace)

#Les variables categoriques
var_cat=["Loan_Status","Gender","Married","Dependents","Education","Self_Employed","Property_Area","Credit_History"]
fig, axes = plt.subplots(4, 2, figsize=(12, 15))
#creation de datasets
dt_num=dt[var_num]
dt_cat=dt[var_cat]
dt_cat=pd.get_dummies(dt_cat,drop_first=True)
dt_cat=pd.get_dummies(dt_cat,drop_first=True)
dt_cat
dt_encoded=pd.concat([dt_cat,dt_num],axis=1)
y=dt_encoded["Loan_Status_Y"]
x=dt_encoded.drop("Loan_Status_Y",axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

dt.to_csv("loan.csv")
# Step2: Model
model=LogisticRegression()

# Step3: Train
model.fit(x_train,y_train)

# Step4: Test
print("Votre Intelligence Arti est fiable à")
print(model.score(x_test,y_test)*100),print("%")
# Web model déployment with streamlit