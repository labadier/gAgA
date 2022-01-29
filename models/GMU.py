import torch, os


class MultiTaskLoss(torch.nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

    def sigmoid(self, z ):
      return 1./(1 + torch.exp(-z))

    def forward(self, outputs, labels):

        outputs = self.sigmoid(outputs) 
        outputs = -(labels*torch.log(outputs) + (1. - labels)*torch.log(1. - outputs))                     
        return torch.sum(outputs)
        
class Aditive_Attention(torch.nn.Module):

  def __init__(self, units=32, input=64, usetanh=False):
    super(Aditive_Attention, self).__init__()
    self.units = units
    self.aditive = torch.nn.Linear(in_features=input, out_features=1)
    self.usetanh=usetanh

  def forward(self, x, getattention=False):

    attention = self.aditive(x)
    attention = torch.nn.functional.softmax(torch.squeeze(attention), dim=-1)
    if self.usetanh == True:
      attention = torch.tanh(x)*torch.unsqueeze(attention, -1)
    else: attention = x*torch.unsqueeze(attention, -1)
    
    weighted_sum = torch.sum(attention, axis=1)
    if getattention == True:
      return weighted_sum, attention
    return weighted_sum


class GMU(torch.nn.Module):

  def __init__(self,interm_size=64, max_length=120, **kwargs):

    super(GMU, self).__init__()

    self.best_acc = None
    self.gmu = Aditive_Attention(input=64, usetanh=True)

    if kwargs['multitask'] == True:
      self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=64, out_features=interm_size), torch.nn.LeakyReLU(), torch.nn.Linear(in_features=interm_size, out_features=5))
      self.loss_criterion = MultiTaskLoss()
    else: 
      self.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=64, out_features=interm_size), torch.nn.LeakyReLU(), torch.nn.Linear(in_features=interm_size, out_features=2))
      self.loss_criterion = torch.nn.CrossEntropyLoss()

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, A, attention=False):
      
    if attention == True:
      _, att = self.gmu(A['text'].to(device=self.device), getattention=True)
      return att

    X = self.gmu(A['text'].to(device=self.device))
    return self.classifier(X)


  def load(self, path):
      self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)
  
  def makeOptimizer(self, lr, decay):
    return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)