import numpy as np
import scipy.io as io

from keras.models import Sequential#, Model
from keras.layers.core import Dense
#from keras.optimizers import SGD, Adam, RMSprop

def readData(path, emb, keys_to_emb, split_train_test):
  print 'loading data...', 
  X = dict()
  io.loadmat(path + emb, X)
  X = np.concatenate((X.values()[:4]))
  keys = dict()
  io.loadmat(path + keys_to_emb, keys)
  keys = keys['all_keys'][0]
  keys = [str(k[0]) for k in keys]

  set_keys = set(keys)
  with open(path+split_train_test) as f:
    lines = [line.split(' ') for line in f]
  split = {l[0][:-4] : int(l[1].strip()) for l in lines 
                                  if l[0][:-4] in set_keys}
  
  print 'done!'
  return X, keys, split

def readLabels(path, label_name, keys):
  with open(path) as f:
    f.readline()
    label_names = f.readline().strip().split(' ')
    lines = [line.strip().replace('  ',' ').split(' ')
                              for line in f]
  if label_name not in label_names:
    print 'incorrect label name', label_name
    return None
  ind = label_names.index(label_name)
  keys = set(keys)
  img_to_label = {l[0][:-4] : int(l[ind+1]) for l in lines 
                              if l[0][:-4] in keys}
  return img_to_label    
  
def getxy(X, keys, split, img_to_label, part, num_samples=-1):
  split_ind = np.array([i for i in range(len(keys)) if 
                          split[keys[i]] == part])
  x = X[split_ind,:]
  y = np.array([img_to_label[keys[i]] for i in split_ind])
  y [y==-1] = 0
  y = (y[:,None] == np.arange(2)).astype(np.int32)
  if num_samples > 0:
    negative = np.random.permutation(np.where(y[:,0] == 1)[0])[:num_samples]
    positive = np.random.permutation(np.where(y[:,1] == 1)[0])[:num_samples]
    new_ind = np.concatenate([negative, positive])
    np.random.shuffle(new_ind)
    x = x[new_ind,:]
    y = y[new_ind,:]
  return x, y
  
def split_train_test(X, keys, split, img_to_label, num_samples=-1):
  TRAIN, VAL, TEST = 0, 1, 2    

  x_train, y_train = getxy(X, keys, split, img_to_label, TRAIN, num_samples)
  x_val, y_val =  getxy(X, keys, split, img_to_label, VAL)
  x_test, y_test =  getxy(X, keys, split, img_to_label, TEST)

# 1- positive, 0-negative
# in one-hot it becomes [1,0] - negative, [0,1] - positive    
  percent_positive = float(y_train[:,1].sum())/y_train.shape[0]
  class_weight = {1:1-percent_positive, 0:percent_positive}
  print 'negative and positive examples: ' , y_train.sum(axis=0),
  print ', percentage:', y_train.sum(axis=0).astype(np.float)/y_train.shape[0]
#  print 'calculating mean and std per coordinate'  
  train_mean = x_train.mean()#axis=0)
  train_std = x_train.std()#axis=0)
  def nor(x):
    return (x-train_mean)/train_std
    
  x_train = nor(x_train)
  x_val = nor(x_val)
  x_test = nor(x_test)
  
  return x_train, y_train, x_val, y_val, x_test, y_test, class_weight
 
 
def run(x_train, y_train, x_val, y_val, x_test, 
      y_test, nb_epoch=10, class_weight=[.5, .5]):
  np.random.seed(1337)  # for reproducibility

  batch_size = 128    

  model = Sequential()
  model.add(Dense(1024, activation='relu', input_shape=(x_train.shape[1],)))
  model.add(Dense(2, activation='softmax'))
 
#  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  
  model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, 
            validation_data=(x_val,y_val),
            nb_epoch=nb_epoch, class_weight=class_weight,
            show_accuracy=True, verbose=1)

  score = model.evaluate(x_test, y_test, show_accuracy=True,
                         verbose=1)
  
  print 'Test score:', score[0]
  print 'Test accuracy:', score[1]

  return model