# save the wav file from firebase

!pip install firebase

#saving wav file from firebase storage to g-drive 

from json import load
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore
from firebase_admin import storage
from google.colab import drive
from google.cloud import storage as gs
from google.colab import auth

drive.mount('/content/drive',force_remount=True)

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
      breath_info[doc.get('path').replace('ibreath2live/','').replace('.wav','')] =user_info+[doc.get('mode')]
      
    
print(breath_info)

bucket = firebase_admin.storage.bucket(name = 'ibreath2live.appspot.com')

num = 0
for i,blob in enumerate(bucket.list_blobs()) :
  if 'ibreath2live' in blob.name and '.wav' in blob.name :
    name = blob.name.replace('ibreath2live/','').replace('.wav','')
    if name in breath_info.keys() :
      num = num + 1
      print( str(num) + ' name : ' + name)
      with open("/content/drive/MyDrive/Mozzign ML team/breathfolder/"+name+".wav", "wb") as file_obj:
        blob.download_to_file(file_obj)
