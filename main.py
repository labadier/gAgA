import argparse, sys, os, numpy as np, torch, random
from models.LXMERT import LXMERT
from models.VisualBERT import VisualBERT
from models.models import save_encodings, train_model_CV, train_with_dev, predict
from models.models import VISUAL_MODELS, MODELS
from utils import bcolors, load_data, plot_training
import params

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')


  parser.add_argument('-modeltype', metavar='modeltype', help='Type of Architecture, e.g. Multimodal', choices=['multimodal', 'text', 'images'])
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
  parser.add_argument('-min_boxes', metavar='min_boxes', default=params.MIN_EDGE, type=int, help='Minimun Boxes for Detectron Model')
  parser.add_argument('-max_boxes', metavar='max_boxes', default=params.MAX_EDGE, type=int, help='Minimun Boxes for Detectron Model')
  parser.add_argument('-val_rate', metavar='val_rate', default=params.VAL_RATE, type=float, help='Validation Rate')
  parser.add_argument('-tf', metavar='train_file', help='Data Anotation Files for Training')
  parser.add_argument('-df', metavar='dev_file', help='Data Anotation Files for Development', default=None)
  parser.add_argument('-gf', metavar='test_file', help='Data Anotation Files for Testing')
  parser.add_argument('-tm', metavar='training_mode', default=params.TRAIN_MODE, help='Weights Update mode', choices=['dynamic', 'static'])
  parser.add_argument('-wp', metavar='weigths_path', default=params.TRAIN_MODE, help='Saved Weights Path')
  parser.add_argument('-mtl', metavar='multitask', default=params.MULTITASK, help='Multitask Leatning')

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
  phase = parameters.phase
  output = parameters.output
  arch = parameters.arch
  
  tf = parameters.tf
  df=parameters.df
  gf=parameters.gf
  val_rate=parameters.val_rate
  min_boxes = parameters.min_boxes
  max_boxes =  parameters.max_boxes
  modeltype = parameters.modeltype
  training_mode = parameters.tm
  weights_path = parameters.wp
  multitask = (parameters.mtl == 'mtl')
  # textF, imageF, labelF ="preprotext", "images","irony"
  
  if modeltype == 'multimodal':

    if phase == 'train':

      output = os.path.join(output, 'logs')

      if os.path.exists(output) == False:
        os.system(f'mkdir {output}')

      images_path, text, labels = load_data(data_path, tf, True, multitask=multitask)
      data = {'text':text, 'images':images_path, 'labels':labels}
      
      if df != None:
        dimages_path, dtext, dlabels = load_data(data_path, df, True, multitask=multitask)#, imageF, textF, labelF)
        datadev = {'text':dtext, 'images':dimages_path, 'labels':dlabels}
        history = train_with_dev(arch, datatrain=data, datadev=datadev, epoches = epoches, 
                            batch_size = batch_size, max_length = max_length, interm_layer_size = interm_layer_size,
                            lr = learning_rate, decay=decay, output=output, validation_rate=val_rate, multitask=multitask, max_edge = max_edge, 
                            min_edge = min_edge, min_boxes = min_boxes, max_boxes = max_boxes)
      else:
        history = train_model_CV(arch, data, splits = splits, epoches = epoches, 
                            batch_size = batch_size, max_length = max_length, interm_layer_size = interm_layer_size,
                            lr = learning_rate,  decay=decay, output=output, multitask=multitask, max_edge = max_edge, 
                            min_edge = min_edge, min_boxes = min_boxes, max_boxes = max_boxes)
      
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Training Finished for {arch.upper()} Model{bcolors.ENDC}")
      plot_training(history[-1], arch, output, 'acc')
      exit(0)

    if phase == 'eval':
      
      images_path, text = load_data(data_path, gf, labeled = False, multitask=multitask)
      data = {'text':text, 'images':images_path} 

      params = {'max_edge': max_edge, 'min_edge': min_edge, 'min_boxes':min_boxes, 'max_boxes':max_boxes, 'model':arch, 'mode':'static', 'multitask':multitask}
      model = MODELS[arch](interm_layer_size=interm_layer_size, max_length=max_length, **params)

      predict(arch, model, data, batch_size, output, images_path, weights_path, multitask=multitask)
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Predictions Saved{bcolors.ENDC}")
    exit(0)
  
  if modeltype == 'text':

    if phase == 'train':

      output = os.path.join(output, 'logs')

      if os.path.exists(output) == False:
        os.system(f'mkdir {output}')

      _, text, labels = load_data(data_path, tf, True, multitask=multitask)
      data = {'text':text, 'labels':labels}
      
      if df != None:
        _, dtext, dlabels = load_data(data_path, df, True, multitask=multitask)
        datadev = {'text':dtext, 'labels':dlabels}
        history = train_with_dev(arch, datatrain=data, datadev=datadev, epoches = epoches, 
                            batch_size = batch_size, max_length = max_length, interm_layer_size = interm_layer_size,
                            lr = learning_rate, decay=decay, output=output, validation_rate=val_rate, mode=training_mode,multitask=multitask)
      else:
        history = train_model_CV(arch, data, splits = splits, epoches = epoches, 
                            batch_size = batch_size, max_length = max_length, interm_layer_size = interm_layer_size,
                            lr = learning_rate,  decay=decay, output=output, mode=training_mode, multitask=multitask)
      
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Training Finished for {arch.upper()} Model{bcolors.ENDC}")
      plot_training(history[-1], arch, output, 'acc')

    if phase == 'eval':
    
      images_path, text = load_data(data_path, gf, labeled = False, multitask=multitask)
      data = {'text':text} 

      params = {'model':arch, 'mode':'static', 'multitask':multitask}
      model = MODELS[arch](interm_layer_size, max_length, **params)

      predict(arch, model, data, batch_size, output, images_path, weights_path,multitask=multitask)
      save_encodings(arch, model, data, batch_size, output, images_path, weights_path)
      
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Predictions Saved{bcolors.ENDC}")
    
    exit(0)

  if modeltype == 'images':

    if phase == 'train':

      output = os.path.join(output, 'logs')

      if os.path.exists(output) == False:
        os.system(f'mkdir {output}')

      images_path, text, labels = load_data(data_path, tf, True, multitask=multitask)
      data = {'text':text, 'image':images_path, 'labels':labels}
      
      if df != None:
        dimages_path, dtext, dlabels = load_data(data_path, df, True, multitask=multitask)
        datadev = {'text':dtext, 'image':dimages_path, 'labels':dlabels}
        history = train_with_dev(arch, datatrain=data, datadev=datadev, epoches = epoches, 
                            batch_size = batch_size, max_length = max_length, interm_layer_size = interm_layer_size,
                            lr = learning_rate, decay=decay, output=output, validation_rate=val_rate, mode=training_mode, 
                            multitask=multitask)
      else:
        history = train_model_CV(arch, data, splits = splits, epoches = epoches, 
                            batch_size = batch_size, max_length = max_length, interm_layer_size = interm_layer_size,
                            lr = learning_rate,  decay=decay, output=output, mode=training_mode, 
                            multitask=multitask)
      
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Training Finished for {arch.upper()} Model{bcolors.ENDC}")
      plot_training(history[-1], arch, output, 'acc')
    
    if phase == 'eval':
    
      images_path, text = load_data(data_path, gf, labeled = False, multitask=multitask)
      data = {'image':images_path} 

      params = {'model':arch, 'multitask':multitask}
      model = MODELS[arch](interm_layer_size, max_length, **params)

      predict(arch, model, data, batch_size, output, images_path, weights_path, multitask=multitask)
      save_encodings(arch, model, data, batch_size, output, images_path, weights_path)
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Predictions Saved{bcolors.ENDC}")
    
    exit(0)