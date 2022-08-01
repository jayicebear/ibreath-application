from google.colab import drive
drive.mount('/content/drive',force_remount=True)
path = '/content/drive/MyDrive/Mozzign ML team/'

import tensorflow as tf
import os
from keras.models import load_model
from os import listdir
from os.path import isfile, join
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation, MaxPooling1D, Dropout
from tensorflow.keras.utils import plot_model,to_categorical


class Diagnosis():
  def __init__ (self, id, diagnosis, image_path):
    self.id = id
    self.diagnosis = diagnosis 
    self.image_path = image_path  
    
def get_wav_files():
  audio_path = '/content/drive/MyDrive/Mozzign ML team/breathfolder/'
  files = [f for f in listdir(audio_path) if isfile(join(audio_path, f))]  #Gets all files in dir
  wav_files = [f for f in files if f.endswith('.wav')]  # Gets wav files 
  wav_files = sorted(wav_files)
  return wav_files, audio_path

wav_files, audio_path = get_wav_files()
print(wav_files)
print(len(wav_files))


import csv

def diagnosis_data():
  diagnosis = pd.read_csv('/content/drive/MyDrive/Mozzign ML team/breath_info.csv', header = None)
  diag_dict = { }    
  for index , row in diagnosis.iterrows():
    diag_dict[row[0]] = row[1]     
    print(row[0] + " " + row[1])

  c = 1
  diagnosis_list = []

  print(audio_path)
  for f in wav_files:
    if diag_dict.get(f.replace('.wav','')) is None : 
      print(f)
    else :
      diagnosis_list.append(Diagnosis(c, diag_dict.get(f.replace('.wav','')), audio_path+f))  
      c+=1

  return diagnosis_list

print(len(wav_files))
diagnosis_data()
    
    
def audio_features(filename): 
  count = 0
  sound, sample_rate = librosa.load(filename)
  if len(sound) < 250000 :
    return 
  count += 1
  print('sample rate : ' + str(sample_rate) + ' length : ' + str(len(sound)))
  stft = np.abs((librosa.stft(sound)))  
  
  print('lengt stft : ' + str(len(stft)))
  mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40),axis=1)
  chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
  mel = np.mean(librosa.feature.melspectrogram(sound, sr=sample_rate),axis=1)
  contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate),axis=1)
  tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate),axis=1)
    
  concat = np.concatenate((mfccs,chroma,mel,contrast,tonnetz))
  return concat

def data_points():
  labels = []
  images = []

  to_hot_one = {"normal1":0, "phystr1":1, "mstr1":2, "deep1":3, "relax1":4}

  count = 1
  for f in diagnosis_data():
    print(str(f.id) + " " + str(f.diagnosis) + " " + str(f.image_path))
    if f.diagnosis in to_hot_one and f.image_path: 
      if audio_features(f.image_path) is not None : 
        labels.append(to_hot_one[f.diagnosis]) 
        images.append(audio_features(f.image_path))
        count+=1

  return np.array(labels), np.array(images)

labels, images = data_points()
print(len(images))
print(len(labels))


def preprocessing(labels, images):    

  # Remove Asthma and LRTI
  #images = np.delete(images, np.where((labels == 7) | (labels == 6))[0], axis=0) 
  #labels = np.delete(labels, np.where((labels == 7) | (labels == 6))[0], axis=0)      

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=10)

  # Hot one encode the labels
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)  

  # Format new data
  print('length y_train : ' + str(len(y_train)) + ' y_train.shape[0] : ' + str((y_train.shape[0])))
  y_train = np.reshape(y_train, (y_train.shape[0], 5))
  
  print('length X_train : ' + str(len(X_train)) + ' X_train.shape[0] : ' + str((X_train.shape[0])))
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  
  print('length y_test : ' + str(len(y_test)) + ' y_test.shape[0] : ' + str((y_test.shape[0])))
  y_test = np.reshape(y_test, (y_test.shape[0], 5))
  
  print('length X_test : ' + str(len(X_test)) + ' X_test.shape[0] : ' + str((X_test.shape[0])))
  X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1],  1))

  return X_train, X_test, y_train, y_test
  
  
start = timer()

labels, images = data_points()
X_train, X_test, y_train, y_test = preprocessing(labels, images)

print('Time taken: ', (timer() - start))
    
    
model = Sequential()
model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=(193, 1)))

model.add(Conv1D(128, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(2)) 

model.add(Conv1D(256, kernel_size=1, activation='relu'))

model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(512, activation='relu'))   
model.add(Dense(5, activation='softmax'))
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#              loss='mse',
#              metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70, batch_size=200, verbose=1)


# save the ML model (h5 format)
from keras.models import load_model
model.save(path + 'ibreath_model1.h5')

# import model
model = load_model(path + 'ibreath_model1.h5')
model.summary()


# predict one example using the model 
xhat = '/content/drive/MyDrive/Mozzign ML team/breathfolder/0157f52e-11bd-4863-8c06-4949dcf89ae5.wav'
yhat = model.predict_classes(xhat)
yhat



score = model.evaluate(X_test, y_test, batch_size=60, verbose=0)
print('Accuracy: {0:.0%}'.format(score[1]/1))
print("Loss: %.4f\n" % score[0])

# Plot accuracy and loss graphs
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label = 'training acc')
plt.plot(history.history['val_accuracy'], label = 'validation acc')
plt.legend()

plt.subplot(1,2,2)
plt.title('Loss')
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.legend()


matrix_index = ["normal1", "phystr1", "mstr1", "deep1", "relax1"]

preds = model.predict(X_test)
classpreds = np.argmax(preds, axis=1) # predicted classes 
y_testclass = np.argmax(y_test, axis=1) # true classes

cm = confusion_matrix(y_testclass, classpreds)
print(classification_report(y_testclass, classpreds, target_names=matrix_index))

# Get percentage value for each element of the matrix
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)


# Display confusion matrix 
df_cm = pd.DataFrame(cm, index = matrix_index, columns = matrix_index)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(10,7))
sn.heatmap(df_cm, annot=annot, fmt='')


diagnosis = pd.read_csv('/content/drive/MyDrive/Mozzign ML team/breath_info.csv', header = None)
diag_dict = { }    
for index , row in diagnosis.iterrows():
  diag_dict[row[0]] = row[1]     

to_hot_one = {0 :"normal1", 1:"phystr1", 2:"mstr1", 3:"deep1", 4:"relax1"}

total = 0
accuracy = 0

for file in wav_files :
  xhat = path +'breathfolder/' +file
  x = audio_features(xhat)
  if x is not None :
    total = total + 1
    X = np.reshape(x, (1,x.shape[0], 1))

    original = diag_dict.get(file.replace('.wav',''))
    y_prob = model.predict(X) 
    predicted = y_prob.argmax(axis=-1)

    if original == to_hot_one.get(predicted[0]) :
      accuracy = accuracy + 1
    print('original : ' + original)
    print('predicted : ' + to_hot_one.get(predicted[0]))

print(accuracy/total*100)


print('total : ' + str(total))
print('accuracy : ' + str(accuracy))

