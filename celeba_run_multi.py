import numpy as np

from keras.models import Model
from keras.layers import Dense, Input

def readLabels(path, keys):
  with open(path) as f:
    f.readline()
    label_names = f.readline().strip().split(' ')
    lines = [line.strip().replace('  ',' ').split(' ')
                              for line in f]

  keys = set(keys)
  img_to_labels = []
  for l in range(len(label_names)):  
    img_to_labels.append({line[0][:-4] : int(line[l+1]) for line in lines 
                              if line[0][:-4] in keys})
  return img_to_labels, label_names
  
def getxy(X, keys, split, img_to_labels, part):
  split_ind = np.array([i for i in range(len(keys)) if 
                          split[keys[i]] == part])
  x = X[split_ind,:]
  ys = []
  for l in range(len(img_to_labels)):
    y = np.array([img_to_labels[l][keys[i]] for i in split_ind])
    y [y==-1] = 0
    y = (y[:,None] == np.arange(2)).astype(np.int32)
    ys.append(y)
  return x, ys
  
def split_train_test(X, keys, split, img_to_labels, label_names):
  TRAIN, VAL, TEST = 0, 1, 2    

  x_train, y_train = getxy(X, keys, split, img_to_labels, TRAIN)
  x_val, y_val =  getxy(X, keys, split, img_to_labels, VAL)
  x_test, y_test =  getxy(X, keys, split, img_to_labels, TEST)

# 1- positive, 0-negative
# in one-hot it becomes [1,0] - negative, [0,1] - positive    
  class_weights = []
  for i in range(len(img_to_labels))  :
    percent_positive = float(y_train[i][:,1].sum())/y_train[0].shape[0]
    class_weights.append({1:1-percent_positive, 0:percent_positive})
    print 'class ', label_names[i], ' negative and positive examples: ' , y_train[i].sum(axis=0),
    print ', percentage:', y_train[i].sum(axis=0).astype(np.float)/y_train[0].shape[0]
#  print 'calculating mean and std per coordinate'  
  train_mean = x_train.mean()#axis=0)
  train_std = x_train.std()#axis=0)
  def nor(x):
    return (x-train_mean)/train_std
    
  x_train = nor(x_train)
  x_val = nor(x_val)
  x_test = nor(x_test)
  
  return x_train, y_train, x_val, y_val, x_test, y_test, class_weights
 
 
def run(x_train, y_train, x_val, y_val, x_test, 
      y_test, label_names, nb_epoch=10):
  batch_size = 128    

  vgg_face = Input(shape=(x_train.shape[1],))
  emb = Dense(1024, activation='relu')(vgg_face)
  classifiers = []
  for i in range(len(y_train)):
    classifiers.append(Dense(2, activation='softmax', name=label_names[i])(emb))
  
  model = Model(input=[vgg_face], output=classifiers)
  model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, 
            validation_data=(x_val,y_val),
            nb_epoch=nb_epoch, verbose=1)

  score = model.evaluate(x_test, y_test, verbose=1)

  print 'Test accuracy:', score[41:]
  print 'Average:', np.mean(score[41:])

  return model, score[41:]