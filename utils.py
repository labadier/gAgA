from operator import truediv
import pandas, numpy as np, os, math
from matplotlib import pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_data(path, gold_file, labeled = True, multitask=False, imageField="file_name", textField="Text Transcription", labelField="misogynous"):

  cols = [imageField, labelField, textField]
  if multitask == True:
    cols = [imageField, 'misogynous',	'shaming', 'stereotype',	'objectification',	'violence', textField]

  if labeled == True:
    df = pandas.read_csv(os.path.join(path, gold_file), sep='\t', usecols=cols).to_numpy()
  else: df = pandas.read_csv(os.path.join(path, gold_file), sep='\t', usecols=[imageField, textField]).to_numpy()
  
  labels,text,images = [], [],[]
  for i in range(len(df)):
    pic = os.path.join(path, df[i,0])
    
    if os.path.exists(pic):
      images.append(pic)
      text.append(df[i,-1])
      if labeled == True:
        labels.append(np.array([ df[i,j] for j in range(1, df.shape[1]-1)]))
  
  if labeled == True:
    return np.array(images), np.array(text), np.array(labels)
  return np.array(images), np.array(text)

def compute_eta(eta):
  h = math.floor(eta/3600)
  m = math.floor(int(eta%3600)/60)
  s = int(int(eta%3600)%60)
  
  return' ETA: {}{}:{}{}:{}{}'.format('0'*(1 - int(math.log10(h + 0.99))), h, '0'*(1 - int(math.log10(m+1+ 0.99))), m, '0'*(1 - int(math.log10(s+0.99))), s)
        

def plot_training(history, model, output, measure='loss'):
    
    plotdev = 'dev_' + measure

    plt.plot(history[measure])
    plt.plot(history['dev_' + measure])
    plt.legend(['train', 'dev'], loc='upper left')
    plt.ylabel(measure)
    plt.xlabel('Epoch')
    if measure == 'loss':
        x = np.argmin(history['dev_loss'])
    else: x = np.argmax(history['dev_acc'])

    plt.plot(x,history['dev_' + measure][x], marker="o", color="red")

    if os.path.exists('./logs') == False:
        os.system('mkdir logs')

    plt.savefig(os.path.join(output, f'train_history_{model}.png'))

if __name__=="__main__":
    pass