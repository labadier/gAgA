import torch, pandas, numpy as np, os, math
from models.LXMERT import LXMERT
from models.VisualBERT import VisualBERT
from models.ViT import ViT
from models.GMU import GMU
from models.seqModel import SeqModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, fbeta_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from utils import bcolors, compute_eta
import time
import random

VISUAL_MODELS = {'lxmert': LXMERT, 'visualbert': VisualBERT}
MODELS = {'lxmert': LXMERT, 'visualbert': VisualBERT, 'deberta': SeqModel, 'bertweet': SeqModel, 'vit':ViT, 'multimodal':GMU}
# VISUAL_MODELS = {}
# MODELS = {'multimodal':GMU}

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

class MultimodalData(Dataset):
  def __init__(self, data, label=None):

    self.data = data
    self.label = label

  def __len__(self):
    for i in self.data.keys():
      return self.data[i].shape[0]
    return self.label.shape[0]

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()

    ret = {}
    for i in self.data.keys():
      ret[i] = self.data[i][idx]

    if self.label is not None:
      ret['labels'] = self.label[idx]

    return ret

def sigmoid( z ):
  return 1./(1 + torch.exp(-z))

def compute_acc(ground_truth, predictions, multitask):

  if multitask == False:
    return((1.0*(torch.max(predictions, 1).indices == ground_truth)).sum()/len(ground_truth)).cpu().numpy()

  predictions = torch.where(sigmoid(predictions) > 0.5, 1, 0)

  acc = []
  for i in range(ground_truth.shape[1]):
    acc.append( ((1.0*(predictions[:,i] == ground_truth[:,i])).sum()/ground_truth.shape[0]).cpu().numpy() )
  return np.array(acc)


def train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, split=1, multitask=False):
  
  eloss, eacc, edev_loss, edev_acc = [], [], [], []
  
  optimizer = model.makeOptimizer(lr=lr, decay=decay)
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
      labels = data['labels'].to(model.device)     
      
      optimizer.zero_grad()
      outputs = model(data)
      loss = model.loss_criterion(outputs, labels)
   
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        if j == 0:
          acc = compute_acc(labels, outputs, multitask)
          running_loss = loss.item()
        else: 
          acc = (acc + compute_acc(labels, outputs, multitask))/2.0
          running_loss = (running_loss + loss.item())/2.0

      if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
        
        perc = (1+j)*100.0/batches
        last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
        
        print(last_printed , end="")#+ compute_eta(((time.time()-start_time)*batches)//(j+1))

    model.eval()
    eloss.append(running_loss)
    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        labels = data['labels'].to(model.device) 

        dev_out = model(data)
        if k == 0:
          out = dev_out
          log = labels
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, labels), 0)

      dev_loss = model.loss_criterion(out, log).item()
      dev_acc = compute_acc(log, out, multitask)
      eacc.append(acc)
      edev_loss.append(dev_loss)
      edev_acc.append(dev_acc) 

    band = False

    measure = dev_acc
    if multitask == True:
      measure = dev_acc[0]

    if model.best_acc is None or model.best_acc < measure:
      model.save(os.path.join(output, f'{model_name}_split_{split}.pt'))
      model.best_acc = measure
      band = True

    # ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)[0]:.3f}'
    ep_finish_print = f' acc: {acc} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)}'

    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(last_printed + ep_finish_print)

  return {'loss': eloss, 'acc': eacc, 'dev_loss': edev_loss, 'dev_acc': edev_acc}


def train_model_CV(model_name, data, splits = 5, epoches = 4, batch_size = 8, max_length = 120, 
                    interm_layer_size = 64, lr = 1e-5,  decay=2e-5, output='./logs', multitask=False, **kwargs):

  '''
    kwargs:

    for image + text models in the image embedding from the featrures R-CNN

    min_edge, max_edge: minimun an maximun length for the height and wide of the images to scale in the preprocess stage
     
    for detectron2 models in ROI heads
    min_boxes, max_boxes: mimimun and maximum amount of boxes to keep from the region proposal network
  '''


  params = {'max_edge': 600, 'min_edge': 400, 'min_boxes':10, 'max_boxes':100, 'multitask':multitask}

  if model_name in VISUAL_MODELS.keys() and kwargs['max_edge'] != None :
    params.update({'max_edge':kwargs['max_edge'], 'min_edge':kwargs['min_edge']})

  if model_name == 'VisualBERT' and kwargs['min_boxes'] != None:
    params.update({'min_boxes':kwargs['min_boxes'], 'max_boxes':kwargs['max_boxes']})
  
  if model_name not in VISUAL_MODELS.keys():
    params.update({'model':model_name, 'mode':kwargs['mode']})

  history = []
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  
  tmplb = None
  if data['labels'].ndim == 1:
    tmplb = data['labels']
  else:
    tmplb = data['labels'][:,0]

  for i, (train_index, test_index) in enumerate(skf.split(data['text'], tmplb)):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = MODELS[model_name](interm_layer_size, max_length, **params)
    
    datatrain = {}
    datadev = {}

    for j in data.keys():
      if j != 'labels':
        datatrain[j] = data[j][train_index].reshape(8000, -1)
        datadev[j] = data[j][test_index]
    print(data['labels'][train_index].shape)
    trainloader = DataLoader(MultimodalData(datatrain, data['labels'][train_index]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = DataLoader(MultimodalData(datadev, data['labels'][test_index]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

    history.append(train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, i+1, multitask=multitask))
      
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del model
    del devloader
    break
  return history

def train_with_dev(model_name, datatrain, datadev, epoches = 4, batch_size = 8,
                   max_length = 120, interm_layer_size = 64, lr = 1e-5,  decay=2e-5, 
                   validation_rate=0.1, output='./logs', multitask=False, **kwargs):


  params = {'max_edge': 600, 'min_edge': 400, 'min_boxes':10, 'max_boxes':100, 'multitask':multitask}

  if model_name in VISUAL_MODELS.keys() and kwargs['max_edge'] != None :
    params.update({'max_edge':kwargs['max_edge'], 'min_edge':kwargs['min_edge']})

  if model_name == 'VisualBERT' and kwargs['min_boxes'] != None:
    params.update({'min_boxes':kwargs['min_boxes'], 'max_boxes':kwargs['max_boxes']})

  if model_name not in VISUAL_MODELS.keys():
    params.update({'model':model_name, 'mode':kwargs['mode']})

  if datadev == {} and validation_rate != 0:
    ttrain, ttest, itrain, itest, ltrain,ltest = train_test_split(datatrain["text"], datatrain["images"], datatrain["labels"], test_size=validation_rate, stratify=datatrain["labels"].to_numpy())
    datatrain = {"text":ttrain,'images':itrain,'labels':ltrain}
    datadev = {"text":ttest,'images':itest,'labels':ltest}

  model = MODELS[model_name](interm_layer_size, max_length, **params)
  train = {}
  dev = {}

  for j in datatrain.keys():
    if j != 'labels':
      train[j] = datatrain[j]
      dev[j] = datadev[j]


  trainloader = DataLoader(MultimodalData(train, datatrain['labels']), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
  devloader = DataLoader(MultimodalData(dev, datadev['labels']), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)

  history = [train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, multitask=multitask)]
  
  del trainloader
  del model
  del devloader
  return history


def predict(model_name, model, data, batch_size, output, images_path, wp,  multitask = False, split = 1):
  devloader = DataLoader(MultimodalData(data), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
  model.eval()
  model.load(os.path.join(wp, f'{model_name}_split_{split}.pt'))
  with torch.no_grad():
    out = None
    for k, data in enumerate(devloader, 0):   
      dev_out = model(data)
      if k == 0:
          out = dev_out
      else:  out = torch.cat((out, dev_out), 0)
  
  if os.path.isdir(output) == False:
      os.system(f'mkdir {output}')

  if multitask == False:
    y_hat = np.int32(np.round(torch.argmax(torch.nn.functional.softmax(out, dim=-1), axis=-1).cpu().numpy(), decimals=0))
    dictionary = {'id': np.array([i.split('/')[-1] for i in images_path]),  'misogynous':y_hat}  
    df = pandas.DataFrame(dictionary) 
    df.to_csv(os.path.join(output, 'preds.csv'), sep='\t', index=False, header=False)
  else:
    y_hat = torch.where(sigmoid(out) > 0.5, 1, 0).cpu().numpy()
    dictionary = {'id': np.array([i.split('/')[-1] for i in images_path]),  'misogynous':y_hat[:,0], 'shaming':y_hat[:,1],	'stereotype':y_hat[:,2], 'objectification':y_hat[:,3],	'violence': y_hat[:,4]}  
    df = pandas.DataFrame(dictionary) 
    df.to_csv(os.path.join(output, 'preds.csv'), sep='\t', index=False, header=False)




def save_encodings(model_name, model, data, batch_size, output, images_path, wp, split = 1):
  devloader = DataLoader(MultimodalData(data), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
  model.eval()
  model.load(os.path.join(wp, f'{model_name}_split_{split}.pt'))
  with torch.no_grad():
    out = None
    for k, data in enumerate(devloader, 0):   
      dev_out = model(data, get_encoding=True)
      
      if k == 0:
          out = dev_out
      else:  out = torch.cat((out, dev_out), 0)

  if os.path.isdir(output) == False:
      os.system(f'mkdir {output}')

  torch.save(out, os.path.join(output, 'preds.pt'))