"""
Module for finetuning BERT.

Overall flow:
-------------

    # Input data to BERT consists of text pairs and labels:
    X1, X2, y = (texts_a, texts_b, label)

    # get pretrined BERT and tokenizer
    model, tokenizer = get_model('bert-base-uncased',...)

    # set tokenizer and training parameters in config
    config = FinetuneConfig(tokenizer=tokenizer, epochs=2,...)

    # finetune model
    model = fit(model, X1, X2, y, config)
"""

import sys

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
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

    Parameters
    ----------
    bert_model : string
        one of SUPPORTED_MODELS i.e 'bert-base-uncased','bert-large-uncased'
    do_lower_case : bool
        use lower case with tokenizer
    num_labels : int
        For a classifier, this is the number of distinct classes.
        For a regressor his will be 1.
    model_type : string
        specifies 'classifier' or 'regressor' model
    num_mlp_layers : int
        The number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor as in the original Google code.
    num_mlp_hiddens : int
        The number of hidden neurons in each layer of the mlp.
    local_rank : (int)
        local_rank for distributed training on gpus

    Returns
    -------
    model : BertPlusMLP
        BERT model plus mlp head
    tokenizer : BertTokenizer
        Wordpiece tokenizer to use with BERT
    """
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    print("Loading %s model..."%(bert_model))
    model = BertPlusMLP.from_pretrained(bert_model,
                                        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE\
                                        /'distributed_{}'.format(local_rank),
                                        num_labels=num_labels,
                                        model_type=model_type,
                                        num_mlp_hiddens=num_mlp_hiddens,
                                        num_mlp_layers=num_mlp_layers)

    return model, tokenizer


def get_device(local_rank, use_cuda):
    """
    Get torch device and number of gpus.

    Parameters
    ----------
    local_rank : int
        local_rank for distributed training on gpus
    use_cuda : bool
        use cuda if available
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


def prepare_model_and_device(model, config):
    """
    Prepare model for training and get torch device

    Parameters
    ----------
    model : BertPlusMLP
        BERT model plud mlp head

    len_train_data : int
        length of training data

    config : FinetuneConfig
        Parameters for finetuning BERT
    """
    device, n_gpu = get_device(config.local_rank, config.use_cuda)

    if config.fp16:
        model.half()

    model.to(device)

    if config.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from \
            https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, device


def get_dataloaders(X1, X2, y, config):
    """
    Get train and validation dataloaders.

    Parameters
    ----------
    X1 : list of strings
        text_a for input data
    X2 : list of strings
        text_b for input data text pairs
    y : list of string or list of floats)
        labels/targets for data
    config : FinetuneConfig
        Parameters for finetuning BERT
    """
    dataset = TextFeaturesDataset(X1, X2, y,
                                  config.model_type,
                                  config.label2id,
                                  config.max_seq_length,
                                  config.tokenizer)

    val_len = int(len(dataset) * config.val_frac)
    if val_len > 0:
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])
        val_dl = DataLoader(val_ds, batch_size=config.eval_batch_size,
                            num_workers=5, shuffle=False)
    else:
        val_dl = None
        train_ds = dataset

    if config.local_rank == -1:
        train_sampler = RandomSampler(train_ds) if config.train_sampler == 'random' else None
    else:
        train_sampler = DistributedSampler(train_ds)

    train_dl = DataLoader(train_ds, sampler=train_sampler,
                          batch_size=config.train_batch_size, num_workers=5,
                          drop_last=config.drop_last_batch,
                          shuffle=False)

    return train_dl, val_dl


def get_optimizer(params, len_train_data, config):
    """
    Get and prepare Bert Adam optimizer.

    Parameters
    ----------
    params :
        model parameters to be optimized
    len_train_data : int
        length of training data
    config : FinetuneConfig
        Parameters for finetuning BERT

    Returns
    -------
    optimizer : FusedAdam or BertAdam
        Optimizer for training model
    num_opt_steps : int
        number of optimization training steps
    """

    num_opt_steps = len_train_data / config.train_batch_size
    num_opt_steps = int(num_opt_steps / config.gradient_accumulation_steps) * config.epochs

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]

    if config.local_rank != -1:
        num_opt_steps = num_opt_steps // torch.distributed.get_world_size()

    if config.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/\
                                nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(grouped_params,
                              lr=config.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)

        if config.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=config.loss_scale)
    else:
        optimizer = BertAdam(grouped_params,
                             lr=config.learning_rate,
                             warmup=config.warmup_proportion,
                             t_total=num_opt_steps)

    return optimizer, num_opt_steps


def update_learning_rate(optimizer, global_step, num_opt_steps, config):
    """Update learning rate for optimizer for special warm up BERT uses

    if args.fp16 is False, BertAdam is used that handles this automatically
    """
    lr, warmup = config.learning_rate, config.warmup_proportion
    if config.fp16:
        lr_this_step = lr * warmup_linear(global_step/num_opt_steps, warmup)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step


def fit(model, X1, X2, y, config):
    """
    Finetune pretrained Bert model.

    A training wrapper based on: https://github.com/huggingface/\
    pytorch-pretrained-BERT/blob/master/examples/run_classifier.py

    Parameters
    ----------
    Bert model inputs are triples of: (text_a,text_b,label).
    For single text tasks text_b = None

    model : BertPlusMLP
        pretrained Bert model with a MLP classifier/regressor head

    X1 : list of strings
        First of a pair of input text data, texts_a

    X2 : list of strings
        Second(optional) of a pair of input text data, texts_b

    y : list of string/floats
        labels/targets for input text data

    config : FinetuneConfig
        Parameters for finetuning BERT

    Returns
    --------
    model : BertPlusMLP
        finetuned BERT model plus mlp head

    """

    def log(msg, logger=config.logger, console=True):
        if logger:
            logger.info(msg)
        if console:
            print(msg)
            sys.stdout.flush()

    grad_accum_steps = config.gradient_accumulation_steps

    # change batch_size if do gradient accumulation
    config.train_batch_size = int(config.train_batch_size / grad_accum_steps)

    # build dataloaders from input texts and labels
    train_dl, val_dl = get_dataloaders(X1, X2, y, config)
    log("train data size: %d, validation data size: %d"%
        (len(train_dl.dataset), len(val_dl.dataset) if val_dl else 0))

    # prepare model i.e multiple gpus and fpint16
    model, device = prepare_model_and_device(model, config)

    # get and prepare BertAdam optimizer
    params = list(model.named_parameters())
    optimizer, num_opt_steps = get_optimizer(params, len(train_dl.dataset), config)
    log("Number of train optimization steps is : %d"%(num_opt_steps), console=False)

    #=========================================================
    #                 main training loop
    #=========================================================
    global_step = 0

    for epoch in range(int(config.epochs)):

        model.train()
        losses = []
        batch_iter = tqdm(train_dl, desc="Training", leave=True)

        for step, batch in enumerate(batch_iter):
            batch = tuple(t.to(device) for t in batch)
            loss, _ = model(*batch)

            loss = loss.mean()

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

            if config.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            # step the optimizer every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                update_learning_rate(optimizer, global_step, num_opt_steps, config)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            losses.append(loss.item() * grad_accum_steps)
            batch_iter.set_postfix(loss=np.mean(losses))

        if val_dl is not None:
            test_loss, test_accy = eval_model(model, val_dl, device, config.model_type)
            msg = "Epoch %d, Train loss : %0.04f, Val loss: %0.04f, Val accy = %0.02f%%"
            msg = msg%(epoch+1, np.mean(losses), test_loss, test_accy)
            log(msg)
        else:
            msg = "Epoch %d, Train loss : %0.04f"%(epoch+1, np.mean(losses))
            log(msg, console=False)

    return model


class OnlinePearson():
    """
    Online pearson stats calculator

    Calculates online pearson coefficient via running covariance
    ,variance, and mean  estimates.

    Ref: https://stats.stackexchange.com/questions/23481/\
    are-there-algorithms-for-computing-running-linear-or-logistic-regression-param
    """
    def __init__(self):
        self.num_points = 0.
        self.mean_X = self.mean_Y = 0.
        self.var_X = self.var_Y = self.cov_XY = 0.
        self.pearson = 0.

    def add(self, x, y):
        """Add data point to online calculation"""
        self.num_points += 1
        n = self.num_points
        delta_x = x - self.mean_X
        delta_y = y - self.mean_Y
        self.var_X += (((n - 1)/n) * delta_x * delta_x - self.var_X)/n
        self.var_Y += (((n - 1)/n) * delta_y * delta_y - self.var_Y)/n

        self.cov_XY += (((n - 1)/n) * delta_x * delta_y - self.cov_XY)/n
        self.mean_X += delta_x/n
        self.mean_Y += delta_y/n

        if self.var_X * self.var_Y != 0:
            self.pearson = self.cov_XY/ np.sqrt((self.var_X * self.var_Y))


def eval_model(model, dataloader, device, model_type="classifier", desc="Validation"):
    """
    Evaluate model on validation data.

    Parameters
    ----------
    model : BertPlusMLP
        Bert model plus mlp head
    dataloader : Dataloader
        validation dataloader
    device : torch.device
        device to run validation on
    model_type : string
        specifies 'classifier' or 'regressor' model

    Returns
    -------
    loss : float
        Loss calculated on eval data
    accy : float
        Classification accuracy for classifiers.
        Pearson coorelation for regressors.
    """
    regression_stats = OnlinePearson()
    model.to(device)
    model.eval()
    loss = accy = 0.
    batch_iter = tqdm(dataloader, desc=desc, leave=False)
    for eval_steps, batch in enumerate(batch_iter):
        batch = tuple(t.to(device) for t in batch)
        _, _, _, y = batch
        with torch.no_grad():
            tmp_eval_loss, output = model(*batch)
        loss += tmp_eval_loss.mean().item()

        if model_type == "classifier":
            _, y_pred = torch.max(output, 1)
            accy += torch.sum(y_pred == y)
        elif model_type == "regressor":
            y_pred = output

            # add each (y,y_pred) to stats counter to calc pearson
            for xi, yi in zip(y.detach().cpu().numpy(),
                              y_pred.detach().cpu().numpy()):
                regression_stats.add(xi, yi)

    loss = loss/(eval_steps+1)

    if model_type == "classifier":
        accy = 100 * (accy.item() / len(dataloader.dataset))
    elif model_type == "regressor":
        accy = 100 * regression_stats.pearson

    return loss, accy
