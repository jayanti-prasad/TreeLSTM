import os
from logutil import get_logger 
import torch.optim as optim

class Config:

    def __init__(self, cfg_parser, logger=None):
        self.cfg_parser = cfg_parser
        self.logger = get_logger(self.log_name(),\
             self.log_dir(), self.log_file(), self.log_level(),\
             self.log_to_console())

    def random_seed(self):
        return self.cfg_parser.getint('setting', 'random_seed')

    def input_dir(self):
        return self.cfg_parser.get('setting', 'input_dir')
  
    def model_name(self):
        return self.cfg_parser.get('setting', 'model_name')

    def workspace_dir(self):
        return self.cfg_parser.get('setting', 'workspace_dir')
    
    def glove_dir(self):
        return self.cfg_parser.get('setting', 'glove_dir')
    
    def input_dim(self):
        return self.cfg_parser.getint('setting', 'input_dim')

    def mem_dim(self):
        return self.cfg_parser.getint('setting', 'mem_dim')
  
    def hidden_dim(self):
        return self.cfg_parser.getint('setting', 'hidden_dim')

    def num_classes(self):
        return self.cfg_parser.getint('setting', 'num_classes')
  
    def freeze_embed(self):
        return self.cfg_parser.getboolean('setting', 'freeze_embed')

    def num_epochs(self):
        return self.cfg_parser.getint('training', 'num_epochs')
 
    def batch_size(self):
        return self.cfg_parser.getint('training', 'batch_size')
 
    def learning_rate(self):
        return self.cfg_parser.getfloat('training', 'learning_rate')
 
    def weight_decay(self):
        return self.cfg_parser.getfloat('training', 'weight_decay')
 
    def sparse(self):
        return self.cfg_parser.getboolean('training', 'sparse')
 
    def use_cuda(self):
        return self.cfg_parser.getboolean('training', 'use_cuda')

    def device(self):
        return self.cfg_parser.getboolean('training', 'device')

    def optimizer(self, model):
       optm = self.cfg_parser.get('training', 'optim')
       lr = self.learning_rate()
       wd = self.weight_decay()  

       if optm == 'adam':
          return  optim.Adam(filter(lambda p: p.requires_grad,
             model.parameters()), lr=lr, weight_decay=wd)
       elif optm == 'adagrad':
           return optim.Adagrad(filter(lambda p: p.requires_grad,
             model.parameters()), lr=lr, weight_decay=wd)
       elif optm == 'sgd':
           return optim.SGD(filter(lambda p: p.requires_grad,
             model.parameters()), lr=lr, weight_decay=wd)

    def device(self):
        return self.cfg_parser.get('training', 'device')

    def output_dir(self):
        tmp_dir = self.workspace_dir() + os.sep + "output"
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def scratch_dir(self):
        tmp_dir = self.workspace_dir() + os.sep + "scratch"
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def log_dir(self):
        tmp_dir = self.workspace_dir() + os.sep + "log"
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir
   
    def log_name(self):
        return self.cfg_parser.get('log', 'log_name') 

    def log_file(self):
        return self.cfg_parser.get('log', 'log_file')          

    def log_level(self):
        return self.cfg_parser.get('log', 'log_level')          

    def log_to_console(self):
        return self.cfg_parser.get('log', 'log_to_console')          


