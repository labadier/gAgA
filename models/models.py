import torch, pandas, cv2, numpy as np, os
from transformers import LxmertTokenizer, LxmertModel
from lxmert.processing_image import Preprocess
from lxmert.modeling_frcnn import GeneralizedRCNN
from lxmert.utils import Config
import lxmert.utils
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from utils import bcolors
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class LXMERT(torch.nn.Module):

  def __init__(self, interm_size=64, max_length=120, fcrnn_cpu=False, **kwargs):
    '''
    kwargs min_edge, max_edge
    '''
    super(LXMERT, self).__init__()
		
    self.best_acc = None
    self.max_length = max_length
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.fcrnn_cpu = fcrnn_cpu
    self.interm_neurons = interm_size
    self.encoder = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
    self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

    self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    self.frcnn_cfg.input.max_size_test = kwargs['max_edge']
    self.frcnn_cfg.input.min_size_test = kwargs['min_edge']
    
    if self.fcrnn_cpu == False:
      self.frcnn_cfg.MODEL.DEVICE = self.device
    else: self.frcnn_cfg.MODEL.DEVICE = 'cpu'
    self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)
    self.image_preprocess = Preprocess(self.frcnn_cfg)

    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU())
    self.classifier = torch.nn.Linear(in_features=self.interm_neurons, out_features=2)
    self.loss_criterion = torch.nn.CrossEntropyLoss()
    
    self.to(device=self.device)
    if self.fcrnn_cpu == True:
      self.frcnn.to(device='cpu')


  def forward(self, text, images_path):

    text = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)
    images, sizes, scales_yx = self.image_preprocess(images_path, single_image=False)
    
    self.frcnn.eval()
    # print(self.frcnn.device, images.device, sizes.device, scales_yx.device)
    if self.fcrnn_cpu == False:
      output_dict = self.frcnn(images.to(device=self.device), sizes.to(device=self.device), scales_yx=scales_yx.to(device=self.device), padding="max_detections", max_detections=self.frcnn_cfg.max_detections, return_tensors="pt", location='cuda')
    else: output_dict = self.frcnn(images, sizes, scales_yx=scales_yx, padding="max_detections", max_detections=self.frcnn_cfg.max_detections, return_tensors="pt", location='cpu')
   
    X = self.encoder(input_ids = text['input_ids'], 
              attention_mask = text['attention_mask'],
              visual_feats = output_dict['roi_features'].to(self.device),
              visual_pos = output_dict['normalized_boxes'].to(self.device),
              token_type_ids = text['token_type_ids'],
              return_dict = True)
    
    enc = self.intermediate(X['pooled_output'])
    return self.classifier(enc) 

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    if os.path.exists('./logs') == False:
      os.system('mkdir logs')
    torch.save(self.state_dict(), os.path.join('logs', path))
   
  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):
    return torch.optim.RMSprop(self.parameters(), lr, weight_decay=decay)

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


def train_model(data, frcnn_cpu, splits = 5, epoches = 4, batch_size = 8, max_length = 120, interm_layer_size = 64, lr = 1e-5,  decay=2e-5, edges ={'max_edge':600, 'min_edge':400}):

  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)

  history = []
  for i, (train_index, test_index) in enumerate(skf.split(data['text'], data['labels'])):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    model = LXMERT(interm_layer_size, max_length, max_edge=edges["max_edge"], min_edge=edges["min_edge"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    trainloader = DataLoader(MultimodalData(data['text'][train_index], data['images'][train_index], data['labels'][train_index]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = DataLoader(MultimodalData(data['text'][test_index], data['images'][test_index], data['labels'][test_index]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    batches = len(trainloader)

    for epoch in range(epoches):

      running_loss = 0.0
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
          last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
          print(last_printed, end="")
      
      model.eval()
      history[-1]['loss'].append(running_loss)
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
        history[-1]['acc'].append(acc)
        history[-1]['dev_loss'].append(dev_loss)
        history[-1]['dev_acc'].append(dev_acc) 

      band = False
      if model.best_acc is None or model.best_acc < dev_acc:
        model.save('bestmodel_split_{}.pt'.format(i+1))
        model.best_acc = dev_acc
        band = True

      ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)[0]:.3f}'

      if band == True:
        print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
      else: print(ep_finish_print)

      
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del model
    del devloader
    break
  return history

def predict(model, data, batch_size, output, images_path):

  devloader = DataLoader(MultimodalData(data['text'], data['images']), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)

  model.eval()
  model.load('logs/bestmodel_split_1.pt')
  with torch.no_grad():
    out = None
    ids = None
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