import numpy as np
import quaternions
import math

from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras import backend as K

def qfe(euler, axis='rxyz'):
  # assuming euler is given in degrees
  euler = euler/180*math.pi
  return quaternions.quaternion_from_euler(euler[0], euler[1], euler[2], axis)
def efq(quaternion, axis='rxyz'):
  # returns euler angles in degrees
  euler = np.array(quaternions.euler_from_quaternion(quaternion, axis))
  return euler*180/math.pi
  
def convert_to_quaternion(y):
  if type(y) is list:
    y = np.concatenate(y, axis=1)
  q = np.zeros((y.shape[0],4))
  for i in range(y.shape[0]):
    q[i,:] = qfe(y[i,:])
  return q

def convert_to_euler(q):
  y = np.zeros((q.shape[0],3))
  for i in range(q.shape[0]):
    y[i,:] = efq(q[i,:])
  return y
  
def mae_per_angle(y_pred, y_true):
  if type(y_true) is list:
    y_true = np.concatenate(y_true, axis=1)
  return np.abs(y_pred-y_true).mean(axis=0)

def readLabels(path, keys):
  with open(path) as f:
    f.readline()
    label_names = f.readline().strip().split(' ')
    lines = [line.strip().replace('  ',' ').split(' ')
                              for line in f]

  keys = set(keys)
  img_to_labels = []
  for l in range(len(label_names)):  
    img_to_labels.append({line[0][:-4] : float(line[l+1]) for line in lines 
                              if line[0][:-4] in keys})
  return img_to_labels, label_names
  
def getxy(X, keys, split, img_to_labels, part):
  split_ind = np.array([i for i in range(len(keys)) if 
                          split[keys[i]] == part])
  x = X[split_ind,:]
  y = np.zeros((x.shape[0],3))
  y[:,0] = [img_to_labels[0][keys[i]] for i in split_ind]
  y[:,1] = [img_to_labels[1][keys[i]] for i in split_ind]
  y[:,2] = [img_to_labels[2][keys[i]] for i in split_ind]
  return x, y
  
def split_train_test(X, keys, split, img_to_labels):
  TRAIN, VAL, TEST = 0, 1, 2    

  x_train, y_train = getxy(X, keys, split, img_to_labels, TRAIN)
  x_val, y_val =  getxy(X, keys, split, img_to_labels, VAL)
  x_test, y_test =  getxy(X, keys, split, img_to_labels, TEST)

#  print 'calculating mean and std per coordinate'  
  train_mean = x_train.mean()#axis=0)
  train_std = x_train.std()#axis=0)
  def nor(x):
    return (x-train_mean)/train_std
    
  x_train = nor(x_train)
  x_val = nor(x_val)
  x_test = nor(x_test)
  
  return x_train, y_train, x_val, y_val, x_test, y_test
 
 
def run(x_train, y_train, x_val, y_val, x_test, 
      y_test, nb_epoch=10, use_normalization_layer=False):

  batch_size = 128    

  vgg_face = Input(shape=(x_train.shape[1],))
  emb = Dense(1024, activation='relu')(vgg_face)
  if use_normalization_layer:
    def normalize(x):
      return K.l2_normalize(x, axis=1)
    emb = Dense(y_train.shape[1])(emb)
    predictor = Lambda(normalize)(emb)
  else:    
    predictor = Dense(y_train.shape[1])(emb)
  
#  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model = Model(input=[vgg_face], output=[predictor])
  model.compile(optimizer='adam', loss='mse')
  model.fit(x_train, y_train, batch_size=batch_size, 
            validation_data=(x_val,y_val),
            nb_epoch=nb_epoch,  verbose=1)

  y_pred = model.predict(x_test)

  return model, y_pred