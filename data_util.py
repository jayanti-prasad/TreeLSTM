from __future__ import division
from __future__ import print_function
import os
import sys
import torch
import Constants
from vocab import Vocab
import utils
from  dataset import SICKDataset

def get_embd(cfg, vocab):
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(cfg.input_dir(), 'sick_embed.pth')

    if os.path.isfile(emb_file):
       emb = torch.load(emb_file)
    else:
       # load glove embeddings and vocab
       glove_vocab, glove_emb = utils.load_word_vectors(
       os.path.join(cfg.glove_dir(), 'glove.840B.300d'))
       cfg.logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
       emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=cfg.device())
       emb.normal_(0, 0.05)
       # zero out the embeddings for padding and other special words if they are absent in vocab
       for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
           Constants.BOS_WORD, Constants.EOS_WORD]):
           emb[idx].zero_()
       for word in vocab.labelToIdx.keys():
           if glove_vocab.getIndex(word):
               emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
               torch.save(emb, emb_file)

    return emb


class DataUtil:
    def __init__(self, cfg):
        self.cfg = cfg 
        self.vocab = None 

        train_dir = self.cfg.input_dir() + os.sep + "train" 
        test_dir = self.cfg.input_dir() + os.sep + "test" 
        dev_dir = self.cfg.input_dir() + os.sep + "dev" 

        self.source_dir = [train_dir, dev_dir, test_dir]  
        self.build_vocab() 
 
 
    def build_vocab (self):

        sick_vocab_file = os.path.join(self.cfg.input_dir(), 'sick.vocab')
        if not os.path.isfile(sick_vocab_file):
            token_files_b = [os.path.join(split, 'b.toks') for split in self.source_dir]
            token_files_a = [os.path.join(split, 'a.toks') for split in self.source_dir]
            token_files = token_files_a + token_files_b
            utils.build_vocab(token_files, sick_vocab_file)

        # get vocab object from vocab file previously written
        self.vocab = Vocab(filename=sick_vocab_file,
            data=[Constants.PAD_WORD, Constants.UNK_WORD,
            Constants.BOS_WORD, Constants.EOS_WORD])


    def get_data(self, flag):
        data_dir = os.path.join(self.cfg.input_dir(), flag)

        # load SICK dataset splits
        data_file = os.path.join(data_dir, 'sick_' + flag + '.pth')
        #   dataset = torch.load(data_file)
        #else:
        dataset = SICKDataset(data_dir, self.vocab, self.cfg.num_classes())
        torch.save(dataset, data_file)
    
        return dataset 
