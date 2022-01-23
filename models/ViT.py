import torch, os, sys
sys.path.append('../')
import numpy as np, pandas as pd
from transformers import ViTFeatureExtractor, ViTModel
from torch.utils.data import Dataset, DataLoader, dataloader
from sklearn.model_selection import StratifiedKFold
import random
from utils import bcolors
from PIL import Image
import cv2

def HuggTransformer(model):
  
  if model == "vit":
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    

  return model, feature_extractor

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
  
class ViT(torch.nn.Module):

  def __init__(self, interm_size, max_length, **kwargs):

    super(ViT, self).__init__()
		
    self.model = kwargs['model']
    self.best_acc = None
    self.interm_neurons = interm_size
    self.ViTModel, self.feature_extractor = HuggTransformer(self.model)
    self.intermediate = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU())
    self.classifier = torch.nn.Linear(in_features=self.interm_neurons, out_features=2)
    self.loss_criterion = torch.nn.CrossEntropyLoss()
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, data, get_encoding=False):
    
    features = self.feature_extractor([cv2.imread(i) for i in data['image']], return_tensors='pt').to(device=self.device)

    X = self.ViTModel(**features).pooler_output

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

    return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
