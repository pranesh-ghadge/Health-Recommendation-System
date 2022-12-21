import pandas as pd
import numpy as np
from math import *
import tensorflow as tf
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

transactions = pd.read_csv('NewData.csv')
policies = pd.read_csv('Policy_Info.csv')


label_encoder = preprocessing.LabelEncoder()
transactions['Gender']= label_encoder.fit_transform(transactions['Gender'])
# transactions['Residence']= label_encoder.fit_transform(transactions['Residence'])

transactions['PolicyName']= transactions['PolicyName'].str.replace("Policy_", "").astype("int")

features = []
for i in range(1, len(transactions.columns) - 1):
    features.append(transactions.columns[i])

X = transactions.loc[:, features]
y = transactions.loc[:, ["PolicyName"]]

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, train_size = .75)

rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(X, y.values.ravel())

# -------------------------------------------------------------------------------------


# label_encoder = preprocessing.LabelEncoder()
# transactions['Gender']= label_encoder.fit_transform(transactions['Gender'])
# transactions['PolicyName']= transactions['PolicyName'].str.replace("Policy_", "").astype("int")

# # transactions['Age']= label_encoder.fit_transform(transactions['Age'])

# # transactions['Income']= label_encoder.fit_transform(transactions['Income'])


# features = []
# for i in range(0, len(transactions.columns) - 1):
#     features.append(transactions.columns[i])

# X = transactions.loc[:, features]
# y = transactions.loc[:, ["PolicyName"]]

# # from sklearn.model_selection import train_test_split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# from sklearn.linear_model import LogisticRegression
# rf_clf = LogisticRegression(solver='liblinear', random_state=0)

# rf_clf = LogisticRegression(multi_class='multinomial',solver='newton-cg',random_state=1)
# rf_clf.fit(X, y)


# ----------------------------------------------------------------------------------------

label_encoder = preprocessing.LabelEncoder()
policies['Maternity']= label_encoder.fit_transform(policies['Maternity'])
policies['OPD Benefits']= label_encoder.fit_transform(policies['OPD Benefits'])

policies = policies.drop(['Name', 'Insurer'], axis=1)

for col in ['Cover(lac)', 'Premium(annual)', 'Pre-Existing Waiting Period', 'ClaimSettlementRatio']:
  policies[col] = (policies[col] - policies[col].min()) / (policies[col].max() - policies[col].min())

policies = policies.set_index('PolicyName')

def euclidean_dist(x, y):
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def similarityIn(polA, polB):
  lA = []
  lB = []
  for i in policies.loc[polA]:
    lA.append(i)
  for i in policies.loc[polB]:
    lB.append(i)
  return euclidean_dist(lA, lB)

def nextTwo(policy):
  dict = {}
  for pol in policies.index:
    dict[pol] = similarityIn(pol, policy)
  topOne = max(dict, key=dict.get)
  dict.pop(topOne)
  topTwo = max(dict, key=dict.get)
  dict.pop(topTwo)
  ans = [topOne, topTwo]
  return ans



#------------------------------------------------------


from flask import Flask, render_template, request, redirect, url_for
from form import profileForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'xyzflaskapp'


@app.route('/')
def hello():
  # return str("Welcome Visiter")
  return render_template('expr.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
  form = profileForm()
  if form.is_submitted():
    age = form.age.data
    gender = form.gender.data
    income = form.income.data
    # residence = form.residence.data
    # diabetes = form.diabetes.data
    # heart = form.heart.data
    # bp = form.bp.data
    # other = form.other.data
    # surg = form.surg.data
    # covid = form.covid.data
    # pred = rf_clf.predict([[age, gender, income, residence, diabetes, heart, bp, other, surg, covid]])[0]
    pred = rf_clf.predict([[age, gender, income]])[0]
    pol1 = 'Policy_' + str(pred)
    topTwo = nextTwo(pol1)
    pol2 = topTwo[0]
    pol3 = topTwo[1]

    # with open ("Policy_Info.csv", "r") as source:
    #   reader = csv.reader(source)

    # policies = pd.read_csv('Policy_Info.csv')

    # row1 = policies.loc[pol1]
    # row2 = policies.loc[pol2]
    # row3 = policies.loc[pol3]

    # pName1=row1['Name'] 
    # pIns1=row1['Insurer']
    # pCov1=row1['Cover(lac)']
    # pPrem1=row1['Premium(annual)']
    # pWait1=row1['Pre-Existing Waiting Period']
    # pClaim1=row1['ClaimSettlementRatio']
    # pMat1=row1['Maternity']
    # pOPD1=row1['OPD Benefits']

    # pName2=row2['Name'] 
    # pIns2=row2['Insurer']
    # pCov2=row2['Cover(lac)']
    # pPrem2=row2['Premium(annual)']
    # pWait2=row2['Pre-Existing Waiting Period']
    # pClaim2=row2['ClaimSettlementRatio']
    # pMat2=row2['Maternity']
    # pOPD2=row2['OPD Benefits']

    # pName3=row3['Name'] 
    # pIns3=row3['Insurer']
    # pCov3=row3['Cover(lac)']
    # pPrem3=row3['Premium(annual)']
    # pWait3=row3['Pre-Existing Waiting Period']
    # pClaim3=row3['ClaimSettlementRatio']
    # pMat3=row3['Maternity']
    # pOPD3=row3['OPD Benefits']


    # return render_template('page2.html', pred1 = pol1, pred2 = pol2, pred3 = pol3, pName1=pName1, pIns1=pIns1, pCov1=pCov1, pPrem1=pPrem1, pWait1=pWait1, pClaim1=pClaim1, pMat1=pMat1, pOPD1=pOPD1,
    # pName2=pName2, pIns2=pIns2, pCov2=pCov2, pPrem2=pPrem2, pWait2=pWait2, pClaim2=pClaim2, pMat2=pMat2, pOPD2=pOPD2,
    # pName3=pName3, pIns3=pIns3, pCov3=pCov3, pPrem3=pPrem3, pWait3=pWait3, pClaim3=pClaim3, pMat3=pMat3, pOPD3=pOPD3)

    if pol1 == 'Policy_1':
      pName1= 'Activ Assure – Diamond'
      pIns1='AdityaBirla'
      pCov1='5'
      pPrem1='5197'
      pMat1='No'
    elif pol1=='Policy_2':
      pName1= 'Reassure'
      pIns1='NivaBupa'
      pCov1='5'
      pPrem1='8102'
      pMat1='No'
    elif pol1=='Policy_3':
      pName1= 'Health Guard Gold'
      pIns1='BajajAllianz'
      pCov1='10'
      pPrem1='9083'
      pMat1='No'
    elif pol1=='Policy_4':
      pName1= 'Optima Secure'
      pIns1='HDFC Ergo'
      pCov1='20'
      pPrem1='15340'
      pMat1='No'
    elif pol1=='Policy_5':
      pName1= 'Young Star Silver'
      pIns1='StarHealth'
      pCov1='20'
      pPrem1='9427'
      pMat1='No'
    elif pol1=='Policy_6':
      pName1= 'Shield'
      pIns1='MagmaHDI'
      pCov1='50'
      pPrem1='13938'
      pMat1='No'
    elif pol1=='Policy_7':
      pName1= 'Lifetime India'
      pIns1='Manipal Cigna'
      pCov1='75'
      pPrem1='12626'
      pMat1='No'
    elif pol1=='Policy_8':
      pName1= 'Premium'
      pIns1='MagmaHDI'
      pCov1='100'
      pPrem1='13233'
      pMat1='No'
    elif pol1=='Policy_9':
      pName1= 'Care Advantage'
      pIns1='CareHealth'
      pCov1='100'
      pPrem1='14608'
      pMat1='No'
    elif pol1=='Policy_10':
      pName1= 'Care Freedom'
      pIns1='CareHealth'
      pCov1='5'
      pPrem1='9304'
      pMat1='Yes'
    elif pol1=='Policy_11':
      pName1= 'Health Premia Silver'
      pIns1='NivaBupa'
      pCov1='5'
      pPrem1='10744'
      pMat1='Yes'
    elif pol1=='Policy_12':
      pName1= 'Activ Assure'
      pIns1='AdityaBirla'
      pCov1='10'
      pPrem1='11012'
      pMat1='Yes'
    elif pol1=='Policy_13':
      pName1= 'Health Premia Gold'
      pIns1='NivaBupa'
      pCov1='20'
      pPrem1='12812'
      pMat1='Yes'
    elif pol1=='Policy_14':
      pName1= 'Health Shield'
      pIns1='BajajAllianz'
      pCov1='40'
      pPrem1='14506'
      pMat1='Yes'
    elif pol1=='Policy_15':
      pName1= 'NCB Super Premium'
      pIns1='CareHealth'
      pCov1='60'
      pPrem1='17324'
      pMat1='Yes'
    elif pol1=='Policy_16':
      pName1= 'Star Platinum'
      pIns1='StarHealth'
      pCov1='100'
      pPrem1='36121'
      pMat1='Yes'


    if pol2 == 'Policy_1':
      pName2= 'Activ Assure – Diamond'
      pIns2='AdityaBirla'
      pCov2='5'
      pPrem2='5197'
      pMat2='No'
    elif pol2=='Policy_2':
      pName2= 'Reassure'
      pIns2='NivaBupa'
      pCov2='5'
      pPrem2='8102'
      pMat2='No'
    elif pol2=='Policy_3':
      pName2= 'Health Guard Gold'
      pIns2='BajajAllianz'
      pCov2='10'
      pPrem2='9083'
      pMat2='No'
    elif pol2=='Policy_4':
      pName2= 'Optima Secure'
      pIns2='HDFC Ergo'
      pCov2='20'
      pPrem2='15340'
      pMat2='No'
    elif pol2=='Policy_5':
      pName2= 'Young Star Silver'
      pIns2='StarHealth'
      pCov2='20'
      pPrem2='9427'
      pMat2='No'
    elif pol2=='Policy_6':
      pName2= 'Shield'
      pIns2='MagmaHDI'
      pCov2='50'
      pPrem2='13938'
      pMat2='No'
    elif pol2=='Policy_7':
      pName2= 'Lifetime India'
      pIns2='Manipal Cigna'
      pCov2='75'
      pPrem2='12626'
      pMat2='No'
    elif pol2=='Policy_8':
      pName2= 'Premium'
      pIns2='MagmaHDI'
      pCov2='100'
      pPrem2='13233'
      pMat2='No'
    elif pol2=='Policy_9':
      pName2= 'Care Advantage'
      pIns2='CareHealth'
      pCov2='100'
      pPrem2='14608'
      pMat2='No'
    elif pol2=='Policy_10':
      pName2= 'Care Freedom'
      pIns2='CareHealth'
      pCov2='5'
      pPrem2='9304'
      pMat2='Yes'
    elif pol2=='Policy_11':
      pName2= 'Health Premia Silver'
      pIns2='NivaBupa'
      pCov2='5'
      pPrem2='10744'
      pMat2='Yes'
    elif pol2=='Policy_12':
      pName2= 'Activ Assure'
      pIns2='AdityaBirla'
      pCov2='10'
      pPrem2='11012'
      pMat2='Yes'
    elif pol2=='Policy_13':
      pName2= 'Health Premia Gold'
      pIns2='NivaBupa'
      pCov2='20'
      pPrem2='12812'
      pMat2='Yes'
    elif pol2=='Policy_14':
      pName2= 'Health Shield'
      pIns2='BajajAllianz'
      pCov2='40'
      pPrem2='14506'
      pMat2='Yes'
    elif pol2=='Policy_15':
      pName2= 'NCB Super Premium'
      pIns2='CareHealth'
      pCov2='60'
      pPrem2='17324'
      pMat2='Yes'
    elif pol2=='Policy_16':
      pName2= 'Star Platinum'
      pIns2='StarHealth'
      pCov2='100'
      pPrem2='36121'
      pMat2='Yes'


    if pol1 == 'Policy_1':
      pName3= 'Activ Assure – Diamond'
      pIns3='AdityaBirla'
      pCov3='5'
      pPrem3='5197'
      pMat3='No'
    elif pol3=='Policy_2':
      pName3= 'Reassure'
      pIns3='NivaBupa'
      pCov3='5'
      pPrem3='8102'
      pMat3='No'
    elif pol3=='Policy_3':
      pName3= 'Health Guard Gold'
      pIns3='BajajAllianz'
      pCov3='10'
      pPrem3='9083'
      pMat3='No'
    elif pol3=='Policy_4':
      pName3= 'Optima Secure'
      pIns3='HDFC Ergo'
      pCov3='20'
      pPrem3='15340'
      pMat3='No'
    elif pol3=='Policy_5':
      pName1= 'Young Star Silver'
      pIns3='StarHealth'
      pCov3='20'
      pPrem3='9427'
      pMat3='No'
    elif pol3=='Policy_6':
      pName3= 'Shield'
      pIns3='MagmaHDI'
      pCov3='50'
      pPrem3='13938'
      pMat3='No'
    elif pol3=='Policy_7':
      pName3= 'Lifetime India'
      pIns3='Manipal Cigna'
      pCov3='75'
      pPrem3='12626'
      pMat3='No'
    elif pol3=='Policy_8':
      pName3= 'Premium'
      pIns3='MagmaHDI'
      pCov3='100'
      pPrem3='13233'
      pMat3='No'
    elif pol3=='Policy_9':
      pName3= 'Care Advantage'
      pIns3='CareHealth'
      pCov3='100'
      pPrem3='14608'
      pMat3='No'
    elif pol3=='Policy_10':
      pName3= 'Care Freedom'
      pIns3='CareHealth'
      pCov3='5'
      pPrem3='9304'
      pMat3='Yes'
    elif pol3=='Policy_11':
      pName3= 'Health Premia Silver'
      pIns3='NivaBupa'
      pCov3='5'
      pPrem3='10744'
      pMat3='Yes'
    elif pol3=='Policy_12':
      pName3= 'Activ Assure'
      pIns3='AdityaBirla'
      pCov3='10'
      pPrem3='11012'
      pMat3='Yes'
    elif pol3=='Policy_13':
      pName3= 'Health Premia Gold'
      pIns3='NivaBupa'
      pCov3='20'
      pPrem3='12812'
      pMat3='Yes'
    elif pol3=='Policy_14':
      pName3= 'Health Shield'
      pIns3='BajajAllianz'
      pCov3='40'
      pPrem3='14506'
      pMat3='Yes'
    elif pol3=='Policy_15':
      pName3= 'NCB Super Premium'
      pIns3='CareHealth'
      pCov3='60'
      pPrem3='17324'
      pMat3='Yes'
    elif pol3=='Policy_16':
      pName3= 'Star Platinum'
      pIns3='StarHealth'
      pCov3='100'
      pPrem3='36121'
      pMat3='Yes'

    return render_template('page2.html', pred1 = pol1, pred2 = pol2, pred3 = pol3, pName1=pName1, pIns1=pIns1, pCov1=pCov1, pPrem1=pPrem1, pMat1=pMat1, pName2=pName2, pIns2=pIns2, pCov2=pCov2, pPrem2=pPrem2, pMat2=pMat2, pName3=pName3, pIns3=pIns3, pCov3=pCov3, pPrem3=pPrem3, pMat3=pMat3)

    # return render_template('page2.html', pred1 = pol1, pred2 = pol2, pred3 = pol3)

  # return str("hi")
  return render_template('Form1.html', form=form)


if __name__ == '__main__':
    app.run()
