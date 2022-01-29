#%%
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import pandas
import os
from utils import load_data

splits = 5
multitask='_mtl'
representations = 'visualbert'

#%%
_, _, labels = load_data('data', 'training.csv', True, multitask=True, dataless=True)
images_path = pandas.read_csv(os.path.join('data', 'Test.csv'), sep='\t', usecols=['file_name']).to_numpy()

datatrain = {'labels':labels, 'enc':{}}
datatest = {'images_path':images_path, 'enc':{}}

for rep in ['bt', 'visualbert', 'vit']:
  if rep == 'bt':
    datatrain['enc']['all'] = torch.load(f'data/train_{rep}{multitask}.pt').cpu().numpy()
    datatest['enc']['all'] = torch.load(f'data/test_{rep}{multitask}.pt').cpu().numpy()
  else: 
    datatrain['enc']['all'] = np.concatenate([datatrain['enc']['all'], torch.load(f'data/train_{rep}{multitask}.pt').cpu().numpy()], axis=-1)
    datatest['enc']['all'] = np.concatenate([datatest['enc']['all'], torch.load(f'data/test_{rep}{multitask}.pt').cpu().numpy()], axis=-1)


representations = 'all'
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)

dictionary = {'id': [i[0] for i in datatest['images_path']]}
for taskindex, task in enumerate(['misogynous', 'shaming',	'stereotype',	'objectification',	'violence']):

  out_test = None
  overall_acc = 0
  for i, (train_index, test_index) in enumerate(skf.split(datatrain['enc'][representations], datatrain['labels'][:,taskindex])):

    model = SVC()
    
    model.fit(datatrain['enc'][representations][train_index], datatrain['labels'][train_index,taskindex])
    output = model.predict(datatrain['enc'][representations][test_index])
    
    if out_test is None:
      out_test = model.predict(datatest['enc'][representations])
    else: out_test += model.predict(datatest['enc'][representations])

    acc = accuracy_score(datatrain['labels'][test_index,taskindex], output)
    metrics = classification_report(output, datatrain['labels'][test_index,taskindex], 
                                    target_names=[f'No {task}', task],  digits=4, zero_division=1)        
    print('Report Split: {} - acc: {}{}'.format(i+1, np.round(acc, decimals=2), '\n'))
    print(metrics)
    overall_acc += acc
  print(task , f'overall accuracy: {acc}')

  pred = np.where(out_test > 2, 1, 0)
  dictionary[task] = pred
  del out_test
  
df = pandas.DataFrame(dictionary) 
df.to_csv('answer.txt', sep='\t', index=False, header=False)

os.system(f'zip {representations}.zip answer.txt')

# %%  Gated Multimodal Unit
from utils import load_data
import pandas, torch

# %%
