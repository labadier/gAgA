from numpy.random import f
import torch, pandas, numpy as np, os
from models.LXMERT import LXMERT
from models.VisualBERT import VisualBERT
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, fbeta_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from utils import bcolors
import time
import random

VISUAL_MODELS = {'lxmert': LXMERT, 'visualbert': VisualBERT}
MODELS = {'lxmert': LXMERT, 'visualbert': VisualBERT}

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

class MultimodalData(Dataset):
  def __init__(self, text, images, label=None):

    self.text = text
    self.images = images
    self.label = label

  def __len__(self):
    return self.text.shape[0]

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    text  = self.text[idx] 
    images = self.images[idx]
    if self.label is not None:
      labels = self.label[idx]
      return {'text': text, 'images': images, 'labels':labels}
    return {'text': text, 'images': images}

def train_model(model_name, model, trainloader, devloader, epoches, lr, decay, split=1):
  
  eloss, eacc, edev_loss, edev_acc = [], [], [], []
  
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
  batches = len(trainloader)

  for epoch in range(epoches):

    running_loss = 0.0
    start_time = time.time()
    perc = 0
    acc = 0
    
    model.train()
    last_printed = ''

    for j, data in enumerate(trainloader, 0):

      torch.cuda.empty_cache()         
      text, images, labels = data['text'], data['images'], data['labels'].to(model.device)      
      
      optimizer.zero_grad()
      outputs = model(text, images)
      loss = model.loss_criterion(outputs, labels)
      
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          acc = ((1.0*(torch.max(outputs, 1).indices == labels)).sum()/len(labels)).cpu().numpy()
          running_loss = loss.item()
        else: 
          acc = (acc + ((1.0*(torch.max(outputs, 1).indices == labels)).sum()/len(labels)).cpu().numpy())/2.0
          running_loss = (running_loss + loss.item())/2.0

      if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
        perc = (1+j)*100.0/batches
        eta = ((time.time()-start_time)*batches)/(j+1)

        last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
        remaining_time = f'ETA: {int(eta/3600):2d}:{int(int(eta%3600)/60):2d}:{int(int(eta%3600)%60):2d}'
        print('f{last_printed} {remaining_time}', end="")
    
    model.eval()
    eloss.append(running_loss)
    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        text, images, labels = data['text'], data['images'], data['labels'].to(model.device) 

        dev_out = model(text, images)
        if k == 0:
          out = dev_out
          log = labels
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, labels), 0)

      dev_loss = model.loss_criterion(out, log).item()
      dev_acc = ((1.0*(torch.max(out, 1).indices == log)).sum()/len(log)).cpu().numpy() 
      eacc.append(acc)
      edev_loss.append(dev_loss)
      edev_acc.append(dev_acc) 

    band = False
    if model.best_acc is None or model.best_acc < dev_acc:
      model.save(f'{model_name}_split_{split}.pt')
      model.best_acc = dev_acc
      band = True

    ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)[0]:.3f}'

    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(ep_finish_print)

  return {'loss': eloss, 'acc': eacc, 'dev_loss': edev_loss, 'dev_acc': edev_acc}


def train_model_CV(model_name, data, frcnn_cpu=False, splits = 5, epoches = 4, batch_size = 8, max_length = 120, 
                    interm_layer_size = 64, lr = 1e-5,  decay=2e-5, **kwargs):

  '''
    kwargs:

    for image + text models in the image embedding from the featrures R-CNN

    min_edge, max_edge: minimun an maximun length for the height and wide of the images to scale in the preprocess stage
     
    for detectron2 models in ROI heads
    min_boxes, max_boxes: mimimun and maximum amount of boxes to keep from the region proposal network
  '''

  params = {'max_edge': 600, 'min_edge': 400, 'min_boxes':10, 'max_boxes':100}

  if model_name in VISUAL_MODELS.keys() and kwargs['max_edge'] != None :
    params.update({'max_edge':kwargs['max_edge'], 'min_edge':kwargs['min_edge']})

  if model_name == 'VisualBERT' and kwargs['min_boxes'] != None:
    params.update({'min_boxes':kwargs['min_boxes'], 'max_boxes':kwargs['max_boxes']})

  history = []
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  
  for i, (train_index, test_index) in enumerate(skf.split(data['text'], data['labels'])):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = MODELS[model_name](interm_layer_size, max_length, **params)
    
    trainloader = DataLoader(MultimodalData(data['text'][train_index], data['images'][train_index], data['labels'][train_index]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = DataLoader(MultimodalData(data['text'][test_index], data['images'][test_index], data['labels'][test_index]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

    history.append(train_model(model_name, model, trainloader, devloader, epoches, lr, decay, i+1))
      
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del model
    del devloader
    break
  return history

def train_with_dev(model_name, datatrain, datadev, frcnn_cpu=False, epoches = 4, batch_size = 8,
                   max_length = 120, interm_layer_size = 64, lr = 1e-5,  decay=2e-5, 
                   validation_rate=0.1, **kwargs):


  params = {'max_edge': 600, 'min_edge': 400, 'min_boxes':10, 'max_boxes':100}

  if model_name in VISUAL_MODELS.keys() and kwargs['max_edge'] != None :
    params.update({'max_edge':kwargs['max_edge'], 'min_edge':kwargs['min_edge']})

  if model_name == 'VisualBERT' and kwargs['min_boxes'] != None:
    params.update({'min_boxes':kwargs['min_boxes'], 'max_boxes':kwargs['max_boxes']})

  if datadev == {} and validation_rate != 0:
    ttrain, ttest, itrain, itest, ltrain,ltest = train_test_split(datatrain["text"], datatrain["images"], datatrain["labels"], test_size=validation_rate, stratify=datatrain["labels"].to_numpy())
    datatrain = {"text":ttrain,'images':itrain,'labels':ltrain}
    datadev = {"text":ttest,'images':itest,'labels':ltest}

  model = MODELS[model_name](interm_layer_size, max_length, **params)

  trainloader = DataLoader(MultimodalData(datatrain['text'], datatrain['images'], datatrain['labels']), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
  devloader = DataLoader(MultimodalData(datadev['text'], datadev['images'], datadev['labels']), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

  history = [train_model(model_name, model, trainloader, devloader, epoches, lr, decay)]
  
  del trainloader
  del model
  del devloader
  return history


def predict(model_name, model, data, batch_size, output, images_path, split = 1):
  devloader = DataLoader(MultimodalData(data['text'], data['images']), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
  model.eval()
  model.load(f'logs/{model_name}_split_{split}.pt')
  with torch.no_grad():
    out = None
    for k, data in enumerate(devloader, 0):
      text, images = data['text'], data['images']     
      dev_out = model(text, images)
      if k == 0:
          out = dev_out
      else:  out = torch.cat((out, dev_out), 0)
  y_hat = np.int32(np.round(torch.argmax(torch.nn.functional.softmax(out, dim=-1), axis=-1).cpu().numpy(), decimals=0))

  if os.path.isdir(output) == False:
      os.system(f'mkdir {output}')

  dictionary = {'id': np.array([i.split('/')[-1] for i in images_path]),  'misogynous':y_hat}  
  df = pandas.DataFrame(dictionary) 
  df.to_csv(os.path.join(output, 'preds.csv'))