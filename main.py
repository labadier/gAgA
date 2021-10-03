import argparse, sys, os, numpy as np, torch, random
from matplotlib.pyplot import axis
from models.models import LXMERT, train_model, predict
from sklearn.metrics import classification_report, accuracy_score
from utils import bcolors, load_data, plot_training
import params

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-arch', metavar='architecture', help='Architecture')
  parser.add_argument('-phase', metavar='phase', help='Phase')
  parser.add_argument('-output', metavar='output', help='Output Path')
  parser.add_argument('-lr', metavar='lrate', default = params.LR , type=float, help='learning rate')
  parser.add_argument('-decay', metavar='decay', default = params.DECAY, type=float, help='learning rate decay')
  parser.add_argument('-splits', metavar='splits', default = params.SPLITS, type=int, help='spits cross validation')
  parser.add_argument('-ml', metavar='max_length', default = params.ML, type=int, help='Maximun Tweets Length')
  parser.add_argument('-interm_layer', metavar='int_layer', default = params.IL, type=int, help='Intermediate layers neurons')
  parser.add_argument('-epoches', metavar='epoches', default=params.EPOCHES, type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=params.BS, type=int, help='Batch Size')
  parser.add_argument('-dp', metavar='data_path', help='Data Path')
  parser.add_argument('-min_edge', metavar='min_edge', default=params.MIN_EDGE, type=int, help='Minimun Edge')
  parser.add_argument('-max_edge', metavar='max_edge', default=params.MAX_EDGE, type=int, help='Maximun Edge')
  parser.add_argument('-dt', metavar='data_test', help='Get Data for test')
  parser.add_argument('-gf', metavar='gold_file', help='Data Anotation Files')
  parser.add_argument('-gft', metavar='gold_file_test', help='Data Anotation FIles')
  return parser.parse_args(args)

if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  learning_rate, decay = parameters.lr,  parameters.decay
  splits = parameters.splits
  interm_layer_size = parameters.interm_layer
  max_length = parameters.ml
  min_edge = parameters.min_edge
  max_edge = parameters.max_edge
  batch_size = parameters.bs
  epoches = parameters.epoches
  data_path = parameters.dp
  test_path = parameters.dt
  phase = parameters.phase
  output = parameters.output
  arch = parameters.arch
  gf = parameters.gf
  gft = parameters.gft

  if arch == 'lxmert':

    if phase == 'train':

      if os.path.exists('./logs') == False:
        os.system('mkdir logs')

      images_path, text, labels = load_data(data_path, gf)
      data = {'text':text, 'images':images_path, 'labels':labels}
      history = train_model(data, frcnn_cpu=False, splits = 5, epoches = 4, batch_size = 3, max_length = 120, interm_layer_size = 64, lr = 1e-5,  decay=2e-5, edges ={'max_edge':300, 'min_edge':300})
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Training Finished{bcolors.ENDC}")
      plot_training(history[-1], 'xlmert', 'acc')
      exit(0)

    if phase == 'eval':
      
      images_path, text, labels = load_data(data_path, gft, labeled = False)
      data = {'text':text, 'images':images_path, 'labels':labels} 
      model = LXMERT(interm_layer_size=64, max_length=120, max_edge=300, min_edge=300)
      predict(model, data, 3, output)
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Predictions Saved{bcolors.ENDC}")


