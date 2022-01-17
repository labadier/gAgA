from tkinter.tix import Tree
import torch, sys
sys.path.append('../')
from transformers import AutoTokenizer, AutoModel
import random, numpy as np
from utils import bcolors


def HuggTransformer(model):
  
  if model == "bertweet":
    model = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", do_lower_case=False, TOKENIZERS_PARALLELISM=True)
  elif model == "deberta":
    model = AutoModel.from_pretrained("microsoft/deberta-base")
    tokenizer = AutoTokenizer.from_pretrained( "microsoft/deberta-base", do_lower_case=False, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MultiTaskLoss(torch.nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

    def sigmoid(self, z ):
      return 1./(1 + torch.exp(-z))

    def forward(self, outputs, labels):

        outputs = self.sigmoid(outputs) 
        outputs = -(labels*torch.log(outputs) + (1. - labels)*torch.log(1. - outputs))                     
        return torch.sum(outputs)
  
class SeqModel(torch.nn.Module):

  def __init__(self, interm_size, max_length, **kwargs):

    super(SeqModel, self).__init__()
		
    self.mode = kwargs['mode']
    self.model = kwargs['model']
    self.best_acc = None
    self.max_length = max_length
    self.interm_neurons = interm_size
    self.transformer, self.tokenizer = HuggTransformer(self.model)
    self.intermediate = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU())
    
    if kwargs['multimodal'] == True:
      self.classifier = torch.nn.Linear(in_features=self.interm_neurons, out_features=5)
      self.loss_criterion = MultiTaskLoss()
    else: 
      self.classifier = torch.nn.Linear(in_features=self.interm_neurons, out_features=2)
      self.loss_criterion = torch.nn.CrossEntropyLoss()
    
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, data, get_encoding=False):

    ids = self.tokenizer(data['text'], return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)

    X = self.transformer(**ids)[0]

    X = X[:,0]
    enc = self.intermediate(X)
    output = self.classifier(enc)
    if get_encoding == True:
      return enc, output

    return output 

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):

    if self.mode == 'static':
      return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)

    params = []
    for l in self.transformer.encoder.layer:

      params.append({'params':l.parameters(), 'lr':lr*multiplier}) 
      multiplier += increase

    try:
      params.append({'params':self.transformer.pooler.parameters(), 'lr':lr*multiplier})
    except:
      print(f'{bcolors.WARNING}Warning: No Pooler layer found{bcolors.ENDC}')

    params.append({'params':self.intermediate.parameters(), 'lr':lr*multiplier})
    params.append({'params':self.classifier.parameters(), 'lr':lr*multiplier})

    return torch.optim.RMSprop(params, lr=lr*multiplier, weight_decay=decay)

  # def get_encodings(self, text, batch_size):

  #   self.eval()    
  #   text = pd.DataFrame({'tweets': text, 'label': np.zeros((len(text),))})
  #   devloader = DataLoader(RawDataset(text, dataframe=True), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
 
  #   with torch.no_grad():
  #     out = None
  #     log = None
  #     for k, data in enumerate(devloader, 0):
  #       torch.cuda.empty_cache() 
  #       inputs = data['tweet']

  #       dev_out, dev_log = self.forward(inputs, True)
  #       if k == 0:
  #         out = dev_out
  #         log = dev_log
  #       else: 
  #         out = torch.cat((out, dev_out), 0)
  #         log = torch.cat((log, dev_log), 0)

  #   out = out.cpu().numpy()
  #   log = torch.max(log, 1).indices.cpu().numpy() 
  #   del devloader
  #   return out, log
