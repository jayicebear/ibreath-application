# profile to txt file

from google.colab import drive
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore
from firebase_admin import storage

drive.mount('/content/drive')

if not firebase_admin._apps:
    cred = credentials.Certificate('/content/drive/MyDrive/Mozzign ML team/ibreath2live-firebase-adminsdk-nkhmj-1bfdabd032.json') 
    default_app = firebase_admin.initialize_app(cred, {'databaseURL':'https://ibreath2live.firebaseio.com/'})

db = firestore.client()

# parsing database url 

users = db.collection(u'UserProfile')
docs = users.stream()

user_dic = {}

for doc in docs:
  dic = doc.to_dict()
  if 'id' in dic.keys() :
    user_dic[doc.get('id')] = [doc.get('airquality'),doc.get('ethnicity') ,doc.get('gender'),doc.get('height'),doc.get('res_dis'),doc.get('smoking'),doc.get('weather'),doc.get('weight'),doc.get('yob') ]

#print(user_dic)

breath = db.collection(u'BreathSounds')
docs = breath.stream()
breath_info = {}

for doc in docs:
  if(doc.get('enabled') and (not doc.get('isTest'))) :
    if doc.get('userid') in user_dic.keys() : 
      user_info = user_dic.get(doc.get('userid'))
      #print(user_info)
      breath_info[doc.get('path').replace('ibreath2live/','')] =user_info+[doc.get('mode')]
      
    
print(breath_info)
for key, info in breath_info.items() :
  print(key)
  print(info)


f = open("/content/drive/MyDrive/Mozzign ML team/info.txt", "w+")
for path,info in breath_info.items() : 
  f.write(path+ str(info) + "\n")

f.close()

# profiles to Data Frame for Machine learning modeling

import pandas as pd
users_ref = db.collection(u'UserProfile')
docs = users_ref.stream()

smoking = []
weight = []
height = []
airquality = []
gender = []
res_dis = []
yob = []
ethnicity = []
profid = []
weather = []
for doc in docs:
    smoking.append(doc.get('smoking'))
    weight.append(doc.get('weight'))
    height.append(doc.get('height'))
    airquality.append(doc.get('airquality'))
    gender.append(doc.get('gender'))
    res_dis.append(doc.get('res_dis'))
    yob.append(doc.get('yob'))
    ethnicity.append(doc.get('ethnicity'))
    weather.append(doc.get('weather'))
    
ethnicity = pd.get_dummies(ethnicity)
weather = pd.get_dummies(weather)

airquality = pd.DataFrame(airquality)
airquality.columns = ['air']

weather.columns = ['1','2','3','4']
weather['one'] = weather['1'] + weather['3']
weather['two'] = weather['2'] + weather['4']

weather = weather.drop(['1','2','3','4'],axis=1)


smoke = []
for i in smoking:
  if i == 'N':
    smoke.append(0)
  else:
    smoke.append(1)
smoke = pd.DataFrame(smoke)
smoke.columns = ['smoking']

resp = []
for i in res_dis:
  if i == 'N':
    resp.append(0)
  else:
    resp.append(1)
resp = pd.DataFrame(resp)
resp.columns = ['disease']

gend = []
for i in gender:
  if i == 'F':
    gend.append(0)
  else:
    gend.append(1)
gend = pd.DataFrame(gend)
gend.columns = ['gender']


yobt = []
for i in yob:
  yobt.append(2022-i)

yobt = pd.DataFrame(yobt)
yobt.columns = ['age']


combine = []
combine = pd.concat([ethnicity,weather,smoke,resp,gend,yobt,airquality],axis=1, join='inner')

breathinfo = {} 
combine





