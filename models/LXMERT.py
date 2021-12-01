import torch, os
from transformers import LxmertTokenizer, LxmertModel
from lxmert.processing_image import Preprocess
from lxmert.modeling_frcnn import GeneralizedRCNN
from lxmert.utils import Config

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
    torch.save(self.state_dict(), path)
