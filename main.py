import os
import celeba_run as cr
import celeba_run_multi as crm
import celeba_run_pose as crp
import numpy as np

path = '../../data/celeba/for_keras/'
emb =  'embeddings_32_split'
keys_to_emb = 'good_keys'
split_train_test = 'list_eval_partition.txt'
X, keys, split = cr.readData(path, emb, keys_to_emb, split_train_test)

def classify(label,subfolder='', num_samples=-1):
  out_file = 'results/'+subfolder+'/'+label+'.txt'
  if os.path.exists(out_file):
    with open(out_file) as f:
      print out_file, 'exists'
      return None, float(f.read())
  img_to_label = cr.readLabels(path + 'list_attr_celeba.txt', label, keys)
  x_train, y_train, x_val, y_val, x_test, y_test, class_weight = cr.split_train_test(X, keys, split, img_to_label, num_samples)
  model = cr.run(x_train, y_train, x_val, y_val, x_test, y_test, nb_epoch=5, class_weight=class_weight)
  score = model.evaluate(x_test, y_test, show_accuracy=True,verbose=1)
  with open(out_file,'w') as f:
      f.write(str(score[1]))
  return model, score[1]
    
celeba_acc = {
  '5_o_Clock_Shadow':91.0 
, 'Arched_Eyebrows':79.0
, 'Attractive':81.0
, 'Bags_Under_Eyes':79.0
, 'Bald':98.0
, 'Bangs':95.0
, 'Big_Lips':68.0
, 'Big_Nose':78.0
, 'Black_Hair':88.0
, 'Blond_Hair':95.0
, 'Blurry':84.0
, 'Brown_Hair':80.0
, 'Bushy_Eyebrows':90.0
, 'Chubby':91.0
, 'Double_Chin':92.0
, 'Eyeglasses':99.0
, 'Goatee': 95.0
, 'Gray_Hair':97.0
, 'Heavy_Makeup':90.0
, 'High_Cheekbones':87.0
, 'Male': 98.0
, 'Mouth_Slightly_Open':92.0
, 'Mustache':95.0
, 'Narrow_Eyes':81.0
, 'No_Beard':95.0
, 'Oval_Face':66.0
, 'Pale_Skin':91.0
, 'Pointy_Nose':72.0
, 'Receding_Hairline':89.0
, 'Rosy_Cheeks':90.0
, 'Sideburns':96.0
, 'Smiling':92.0
, 'Straight_Hair':73.0
, 'Wavy_Hair':80.0
, 'Wearing_Earrings':82.0
, 'Wearing_Hat':99.0
, 'Wearing_Lipstick':93.0
, 'Wearing_Necklace':71.0
, 'Wearing_Necktie':93.0
, 'Young':87.0
  }
  
my_acc = celeba_acc;

for k in celeba_acc.keys():
  my_acc[k] = 0.0
      
def classify_all(subfolder='', num_samples=-1, multi_classifier=False):
  if not os.path.exists('results/' + subfolder ):
    os.mkdir('results/' + subfolder )  
  if multi_classifier:
    img_to_labels, label_names = crm.readLabels(path + 'list_attr_celeba.txt', keys)
    x_train, y_train, x_val, y_val, x_test, y_test, class_weights = crm.split_train_test(X, keys, split, img_to_labels, label_names)
    model, score = crm.run(x_train, y_train, x_val, y_val, x_test, y_test, label_names, nb_epoch=2, class_weight=class_weights)
    return score
    
  for k in celeba_acc.keys():
    print 'working on ', k, '(', subfolder,')\n\n\n'
    _, my_score = classify(k, subfolder, num_samples)
    my_acc[k] = my_score
    print '\n\n\n My accuracy for: ', k, ' is ', my_score, '\n\n\n\n'
      
  return my_acc
  
def predict_pose(subfolder='', use_quaternions=False, use_normalization_layer=False):
  if not os.path.exists('results/pose/' + subfolder ):
    os.mkdir('results/pose/' + subfolder )  
  print 'working on pose', '(', subfolder,')\n'
  out_file = 'results/pose/'+subfolder+'/res.txt'
  if os.path.exists(out_file):
    with open(out_file) as f:
      print out_file, 'exists'
      return None, f.read()
  img_to_label,_ = crp.readLabels(path + 'pose.txt', keys)
  has_pose = [i for i in range(X.shape[0]) if not np.isnan(img_to_label[0][keys[i]])]
  keys_filt = [keys[i] for i in range(len(keys)) 
            if not np.isnan(img_to_label[0][keys[i]])]
  X_filt = X[has_pose,:]
  for i in range(3):
    img_to_label[i] = {k : img_to_label[i][k] for k in img_to_label[i].keys() if not np.isnan(img_to_label[i][k])} 
    
  x_train, y_train, x_val, y_val, x_test, y_test = crp.split_train_test(X_filt, keys_filt, split, img_to_label)
  print y_train.shape
  if use_quaternions:
    e2q = crp.convert_to_quaternion
    q_train, q_val, q_test = e2q(y_train), e2q(y_val), e2q(y_test)
    model, q_pred = crp.run(x_train, q_train, x_val, q_val, x_test, q_test, 
                            nb_epoch=5, use_normalization_layer=use_normalization_layer)
    y_pred = crp.convert_to_euler(q_pred)
  else :
    model, y_pred = crp.run(x_train, y_train, x_val, y_val, x_test, y_test, nb_epoch=5)

  mae = crp.mae_per_angle(y_pred, y_test)
  print 'Mean absolute error:', mae  
  with open(out_file,'w') as f:
      f.write(str(mae))
  return model, y_pred

 classify_all('test_classification', multi_classifier=True)
 predict_pose('test_pose', True, True)