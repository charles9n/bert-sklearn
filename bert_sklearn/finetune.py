import numpy as np
import os
import random
import sys
import time
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split 
from torch.utils.data import RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from bert_sklearn.model import BertPlusMLP
from bert_sklearn.data import TextFeaturesDataset

def get_model(bert_model,
              do_lower_case,
              num_labels,
              model_type,
              num_mlp_layers=0,              
              num_mlp_hiddens=500,
              local_rank=-1):
    """
    Get a BertPlusMLP model and BertTokenizer.
    
    Args:
        bert_model (string): one of SUPPORTED_MODELS 
            i.e 'bert-base-uncased','bert-large-uncased'
        do_lower_case (bool): use lower case with tokenizer
        num_labels (int): for a classifier, this is the number of distinct classes.
            For a regressor his will be 1.        
        model_type (string): specifies 'classifier' or 'regressor' model  
        num_mlp_layers (int): the number of mlp layers. If set to 0, then defualts 
            to the linear classifier/regresor as in the original Google code. 
        num_mlp_hiddens (int): the number of hidden neurons in each layer of the mlp.
        local_rank (int): local_rank for distributed training on gpus
    """
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(bert_model,do_lower_case=do_lower_case)

    print("Loading %s model..."%(bert_model))
    model = BertPlusMLP.from_pretrained(bert_model,
            cache_dir = PYTORCH_PRETRAINED_BERT_CACHE/'distributed_{}'.format(local_rank),
            num_labels = num_labels, 
            model_type = model_type,
            num_mlp_hiddens = num_mlp_hiddens,
            num_mlp_layers = num_mlp_layers)

    return model, tokenizer

def get_dataloaders(X1,X2,y,train_batch_size,eval_batch_size,
                    model_type,label2id,max_seq_length,tokenizer,
                    val_frac,local_rank,train_sampler,drop_last_batch):
    """
    Get train and validation dataloaders.
    
    Args:

        X1 (list of strings): text_a for input data
        X2 (list of strings, None) : text_b for input data text pairs
        y (list of string or list of floats):  labels/targets for data
        train_batch_size (int): batch size for training
        eval_batch_size (int): batch_size for eval
        model_type (string): specifies 'classifier' or 'regressor' model  
        label2id ( dict map of string to int): label map for labels
        max_seq_length (int): maximum length of input text sequence (text_a + text_b)
        tokenizer (BertTokenizer): standard word tokenizer followed by WordPiece Tokenizer
        val_frac (float): fraction of training set to use for validation
        local_rank (int): local_rank for distributed training on gpus
        train_sampler (string): if 'random' then uses random sampler, else None.
    """
    ds = TextFeaturesDataset(X1,X2,y,
        model_type,
        label2id,
        max_seq_length,
        tokenizer)

    val_len = int(len(ds) * val_frac)
    if val_len > 0:
        train_ds,val_ds = random_split(ds,[len(ds) - val_len,val_len])
        val_dl = DataLoader(val_ds, batch_size=eval_batch_size,
                            num_workers=5,shuffle=False)
    else:
        val_dl = None
        train_ds = ds

    if local_rank==-1:
        train_sampler = RandomSampler(train_ds) if train_sampler=='random' else None
    else:
        train_sampler = DistributedSampler(train_ds)
        
    train_dl = DataLoader(train_ds,sampler=train_sampler,
                          batch_size=train_batch_size,num_workers=5,
                          drop_last=drop_last_batch,
                          shuffle=False)   
    
    return train_dl,val_dl
  

def get_device(local_rank,use_cuda):
    """
    Get torch device and number of gpus.
    
    Args:
        local_rank (int): local_rank for distributed training on gpus
        use_cuda (bool): use cuda if available
    """
    if local_rank == -1 or not use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will 
        # take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        
    return device, n_gpu

def prepare_model(model,fp16,device,local_rank,n_gpu):
    """
    Prepare model for training.
    """        
    if fp16:
        model.half()
        
    model.to(device)
    
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from \
            https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    return model

def get_optimizer(model,num_train_steps,local_rank,fp16,
            loss_scale,learning_rate,warmup_proportion):
    """
    Get and prepare Bert Adam optimizer.
    """    
    # get optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]  

    t_total = num_train_steps
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
        
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=t_total) 
                             
    return optimizer, t_total       

def fit(model,
        tokenizer,
        X1, X2, y,
        model_type,        
        epochs=1,
        max_seq_length=128,
        learning_rate = 2e-5 ,
        warmup_proportion = 0.1,
        train_batch_size = 32,
        eval_batch_size = 8,
        label2id = None,
        gradient_accumulation_steps=1,
        fp16 = False,
        loss_scale=0,
        local_rank=-1,
        use_cuda = True,
        train_sampler='random',
        drop_last_batch=False,
        val_frac=0.15,
        logger=None):
    
    """
    Finetune pretrained Bert model.
    
    A training wrapper based on:
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py
    
    Args:
     
        Bert model input are triples of: (text_a,text_b,label/target). 
        For single text tasks text_b = None
     
        model (BertPlusMLP): pretrained Bert model with a MLP head
        tokenizer (BertTokenizer): standard word tokenizer followed by WordPiece Tokenizer
        X1 (list of strings): text_a for input text data
        X2 (list of strings, optional) : text_b input text pair data
        y (list of string/floats):  labels/targets for input text data
        model_type (string): specifies 'classifier' or 'regressor' model  
        epochs (int): number of training epochs
        max_seq_length (int): maximum length of input text sequence (text_a + text_b)
        learning_rate (float): inital learning rate ofr Bert Optimizer
        warmup_proportion (float): proportion of training to perform learning rate warmup
        train_batch_size (int): batch size for training
        eval_batch_size (int): batch_size for eval
        label2id ( dict map of string to int): label map for labels
        gradient_accumulation_steps (int): number of updates steps to accumulate
            before performing a backward/update pass
        fp16 (bool): whether to use 16-bit float precision instead of 32-bit
        loss_scale (float): loss scaling to improve fp16 numeric stability. 
            Only used when fp16 set to True
        local_rank (int): local_rank for distributed training on gpus
        use_cuda (bool): use cuda if available
        train_sampler (string): use 'random or 'sequential
        val_frac (float): fraction of training set to use for validation
        logger (logging.Logger): logger to use for logging info
    """
    
    def log(msg,logger=logger,console=True):
        if logger: logger.info(msg)
        if console:
            print(msg)
            sys.stdout.flush()
            
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x 
    
    global_step = 0 
                  
    train_batch_size = int(train_batch_size / gradient_accumulation_steps)
                 
    train_dl, val_dl = get_dataloaders(X1,X2,y,train_batch_size,eval_batch_size,
                    model_type,label2id,max_seq_length,tokenizer,
                    val_frac,local_rank,train_sampler,drop_last_batch)
    
    log("train data size: %d, validation data size: %d"%
        (len(train_dl.dataset),len(val_dl.dataset) if val_dl else 0))
        
    device, n_gpu = get_device(local_rank,use_cuda)
        
    model = prepare_model(model,fp16,device,local_rank,n_gpu)
    
    num_train_steps = int(
            len(train_dl.dataset) / train_batch_size / gradient_accumulation_steps * epochs)         
        
    optimizer, t_total = get_optimizer(model,num_train_steps,local_rank,
                                    fp16,loss_scale,learning_rate,warmup_proportion)
        
    # main training loop
    for epoch in range(int(epochs)):
        model.train()
        losses=[]
        batch_iter = tqdm(train_dl, desc="Training",leave=True)
        for step, batch in enumerate(batch_iter):
            batch = tuple(t.to(device) for t in batch)
            loss,_ = model(*batch)
            
            # we need this now since reduction='none' is in model loss
            loss = loss.mean() 
            
            # this is not needed since the loss is already averaged:        
            #if gradient_accumulation_steps > 1:
            #    loss = loss / gradient_accumulation_steps

            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
                
            if (step + 1) % gradient_accumulation_steps == 0:
            
                # modify learning rate with special warm up BERT uses
                lr_this_step = learning_rate * warmup_linear(global_step/t_total, warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step 
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                              
            losses.append(loss.item())
            batch_iter.set_postfix(loss=np.mean(losses)) 
                
        if val_dl is not None:        
            test_loss,test_accuracy = eval_model(model,val_dl,device,model_type)
            epoch_str = "Epoch %d, Train loss : %0.04f, Val loss: %0.04f, Val accy = %0.02f%%"
            epoch_str = epoch_str%(epoch+1,np.mean(losses),test_loss,test_accuracy)
            log(epoch_str)
        
    return model
    
class OnlinePearson():
    """
    Online pearson stats calculator
    
    Ref: https://stats.stackexchange.com/questions/23481/\
    are-there-algorithms-for-computing-running-linear-or-logistic-regression-param
    """
    def __init__(self):
        self.n = 0.
        self.meanX = self.meanY = 0.
        self.varX = self.varY = self.covXY = 0.
        self.pearson = 0.
        
    def add(self,x,y):
        self.n +=1
        n = self.n
        dx = x - self.meanX
        dy = y - self.meanY
        self.varX += (((n-1)/n)*dx*dx - self.varX)/n
        self.varY += (((n-1)/n)*dy*dy - self.varY)/n

        self.covXY += (((n-1)/n)*dx*dy - self.covXY)/n
        self.meanX += dx/n
        self.meanY += dy/n
        
        if self.varX * self.varY != 0 :
            self.pearson = self.covXY/ np.sqrt((self.varX * self.varY))
            
def eval_model(model,dataloader,device,model_type="classification",desc="Validation"):
    """
    Evaluate model on validation data.
    
    Args:
        model (BertPlusMLP): Bert model plus mlp head
        dataloader (Dataloader): validation dataloader
        device (torch.device): device to run validation on
        model_type (string): specifies 'classifier' or 'regressor' model      
    """
    regression_stats = OnlinePearson()
    model.to(device)
    model.eval()   
    loss = accy =  0.
    batch_iter = tqdm(dataloader,desc=desc,leave=False)
    for eval_steps, batch in enumerate(batch_iter):
        batch = tuple(t.to(device) for t in batch)
        _, _, _, y = batch
        with torch.no_grad():
            tmp_eval_loss, output = model(*batch) 
        loss += tmp_eval_loss.mean().item()
            
        if model_type=="classifier":
            _, y_pred = torch.max(output, 1)
            accy += torch.sum(y_pred==y)
        elif model_type=="regressor":
            y_pred = output
            # add each (y,y_pred) to stats counter to calc pearson
            for xi,yi in zip(y.detach().cpu().numpy(),y_pred.detach().cpu().numpy()):
                regression_stats.add(xi,yi)

    loss = loss/(eval_steps+1)
    
    if model_type=="classifier":
        accy = 100 * accy.item()/len(dataloader.dataset)
    elif model_type=="regressor":
        accy = 100 * regression_stats.pearson
    
    return loss,accy
