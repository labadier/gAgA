#%%
 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier#*
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import pandas
import os

splits = 5

enc_train = torch.load('../data/train_bt_mtl.pt').cpu().numpy()
enc_test = torch.load('../data/test_bt_mtl.pt').cpu().numpy()


cols = ['misogynous']
df = pandas.read_csv(os.path.join('../data', 'training.csv'), sep='\t', usecols=cols).to_numpy()

data = {'text': enc_train, 'labels':  df[:,0].reshape(len(df), 1)}
#%%
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)

overall_acc = 0
last_printed = None

out_test = None
for i, (train_index, test_index) in enumerate(skf.split(data['text'], data['labels'])):

    model = SVC()
    print(data['labels'][train_index].shape)
    model.fit(data['text'][train_index], data['labels'][train_index])
    output = model.predict(data['text'][test_index])
    if out_test is None:
      out_test = model.predict(enc_test)
    else: out_test += model.predict(enc_test)
    acc = accuracy_score(data['labels'][test_index], output)
    metrics = classification_report(output, data['labels'][test_index], target_names=['No Misogeny', 'Misogeny'],  digits=4, zero_division=1)        
    print('Report Split: {} - acc: {}{}'.format(i+1, np.round(acc, decimals=2), '\n'))
    print(metrics)
    overall_acc += acc

pred = np.where(out_test > 2, 1, 0)
# %%
images_path = pandas.read_csv(os.path.join('../data', 'Test.csv'), sep='\t', usecols=['file_name']).to_numpy()

dictionary = {'id': [i[0] for i in images_path],  'misogynous':pred}  
df = pandas.DataFrame(dictionary) 
df.to_csv('../preds.csv', sep='\t', index=False, header=False)

  # %%
