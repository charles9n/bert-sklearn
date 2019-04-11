"""sklearn interface to finetuning BERT."""

import logging
import random

import statistics as stats
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier
from sklearn.model_selection import ParameterGrid
from pytorch_pretrained_bert import BertTokenizer

from bert_sklearn.data import TextFeaturesDataset
from bert_sklearn.finetune import get_model
from bert_sklearn.finetune import get_device
from bert_sklearn.finetune import fit
from bert_sklearn.finetune import eval_model
from bert_sklearn.model import BertPlusMLP
from bert_sklearn.config import FinetuneConfig


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

SUPPORTED_MODELS = ('bert-base-uncased', 'bert-large-uncased', 'bert-base-cased',
                    'bert-large-cased', 'bert-base-multilingual-uncased',
                    'bert-base-multilingual-cased', 'bert-base-chinese')


def set_random_seed(seed=42, use_cuda=True):
    """Seed all random number generators to enable repeatable runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def get_logger(logname, no_stdout=True):
    logger = logging.getLogger()
    handler = logging.StreamHandler(open(logname, "a"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if no_stdout:
        logger.removeHandler(logger.handlers[0])

    return logger


def to_numpy(X):
    """
    Convert input to numpy ndarray
    """
    if hasattr(X, 'iloc'):              # pandas
        return X.values
    elif isinstance(X, list):           # list
        return np.array(X)
    elif isinstance(X, np.ndarray):     # ndarray
        return X
    else:
        raise ValueError("Unable to handle input type %s"%str(type(X)))


def unpack_text_pairs(X):
    """
    Unpack text pairs
    """
    if X.ndim == 1:
        texts_a = X
        texts_b = None
    else:
        texts_a = X[:, 0]
        texts_b = X[:, 1]

    return texts_a, texts_b


class BaseBertEstimator(BaseEstimator):
    """
    Base Class for Bert Classifier and Regressor.

    Parameters
    ----------
    bert_model : string
        one of SUPPORTED_MODELS, i.e 'bert-base-uncased', 'bert-large-uncased'...
    num_mlp_hiddens : int
        the number of hidden neurons in each layer of the mlp
    num_mlp_layers : int
        the number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor in the original Google paper and code
    restore_file : string
        file to restore model state from previous savepoint
    epochs : int
        number of finetune training epochs
    max_seq_length : int
        maximum length of input text sequence (text_a + text_b)
    train_batch_size : int
        batch size for training
    eval_batch_size : int
        batch_size for validation
    label_list :list of strings
        list of classifier labels. For regressors this is None.
    learning_rate :float
        inital learning rate of Bert Optimizer
    warmup_proportion : float
        proportion of training to perform learning rate warmup
    gradient_accumulation_steps : int
        number of update steps to accumulate before performing a backward/update pass
    fp16 : bool
        whether to use 16-bit float precision instead of 32-bit
    loss_scale : float
        loss scaling to improve fp16 numeric stability. Only used when
        fp16 set to True
    local_rank : int
        local_rank for distributed training on gpus
    use_cuda : bool
        use GPU(s) if available
    random_state : intt
        seed to initialize numpy and torch random number generators
    validation_fraction : float
        fraction of training set to use for validation
    logname : string
        path name for logfile

    """
    def __init__(self, label_list=None, bert_model='bert-base-uncased',
                 num_mlp_hiddens=500, num_mlp_layers=0, restore_file=None,
                 epochs=3,
                 max_seq_length=128,
                 train_batch_size=32,
                 eval_batch_size=8,
                 learning_rate=2e-5,
                 warmup_proportion=0.1,
                 gradient_accumulation_steps=1,
                 fp16=False,
                 loss_scale=0,
                 local_rank=-1,
                 use_cuda=True,
                 random_state=42,
                 validation_fraction=0.1,
                 logfile='bert_sklearn.log'):

        self.id2label, self.label2id = {}, {}
        self.input_text_pairs = None

        if restore_file is not None:
            # finish loading a previously finetuned model
            self.restore_finetuned_model(restore_file)
        else:
            # a bert model will be loaded during fit()
            self.label_list = label_list
            self.bert_model = bert_model
            self.num_mlp_hiddens = num_mlp_hiddens
            self.num_mlp_layers = num_mlp_layers
            self.restore_file = restore_file
            self.epochs = epochs
            self.max_seq_length = max_seq_length
            self.train_batch_size = train_batch_size
            self.eval_batch_size = eval_batch_size
            self.learning_rate = learning_rate
            self.warmup_proportion = warmup_proportion
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.fp16 = fp16
            self.loss_scale = loss_scale
            self.local_rank = local_rank
            self.use_cuda = use_cuda
            self.random_state = random_state
            self.validation_fraction = validation_fraction
            self.logfile = logfile

        # use lower case for uncased models
        self.do_lower_case = True if 'uncased' in self.bert_model else False

        self._validate_hyperparameters()

        self.logger = get_logger(logfile)

        if is_classifier(self):
            print("Building sklearn classifier...")
            self.model_type = "classifier"
        else:
            print("Building sklearn regressor...")
            self.model_type = "regressor"
            self.num_labels = 1

        self.logger.info("Loading model:\n" + str(self))

    def load_bert(self):
        """
        Load a BertPlusMLP model from a pretrained checkpoint.

        This will be an pretrianed BERT ready to be finetuned.
        """

        # load a vanilla bert model ready to finetune:
        # pretrained bert LM + untrained classifier/regressor
        self.model, self.tokenizer = get_model(self.bert_model,
                                               self.do_lower_case,
                                               self.num_labels,
                                               self.model_type,
                                               self.num_mlp_layers,
                                               self.num_mlp_hiddens,
                                               self.local_rank)

    def _validate_hyperparameters(self):
        """
        Check hyperpameters are within allowed values.
        """
        if self.bert_model not in SUPPORTED_MODELS:
            raise ValueError("The bert model '%s' is not supported. Supported "
                             "models are %s." % (self.bert_model, SUPPORTED_MODELS))

        if (not isinstance(self.num_mlp_hiddens, int) or self.num_mlp_hiddens < 1):
            raise ValueError("num_mlp_hiddens must be an integer >= 1, got %s"%
                             self.num_mlp_hiddens)

        if (not isinstance(self.num_mlp_layers, int) or self.num_mlp_layers < 0):
            raise ValueError("num_mlp_layers must be an integer >= 0, got %s"%
                             self.num_mlp_layers)

        if (not isinstance(self.epochs, int) or self.epochs < 1):
            raise ValueError("epochs must be an integer >= 1, got %s" %self.epochs)

        if (not isinstance(self.max_seq_length, int) or self.max_seq_length < 2 or \
                           self.max_seq_length > 512):
            raise ValueError("max_seq_length must be an integer >=2 and <= 512, "
                             "got %s" %self.max_seq_length)

        if (not isinstance(self.train_batch_size, int) or self.train_batch_size < 1):
            raise ValueError("train_batch_size must be an integer >= 1, got %s" %
                             self.train_batch_size)

        if (not isinstance(self.eval_batch_size, int) or self.eval_batch_size < 1):
            raise ValueError("eval_batch_size must be an integer >= 1, got %s" %
                             self.eval_batch_size)

        if self.learning_rate < 0 or self.learning_rate >= 1:
            raise ValueError("learning_rate must be >= 0 and < 1, "
                             "got %s" % self.learning_rate)

        if self.warmup_proportion < 0 or self.warmup_proportion >= 1:
            raise ValueError("warmup_proportion must be >= 0 and < 1, "
                             "got %s" % self.warmup_proportion)

        if (not isinstance(self.gradient_accumulation_steps, int) or \
                self.gradient_accumulation_steps > self.train_batch_size or \
                self.gradient_accumulation_steps < 1):
            raise ValueError("gradient_accumulation_steps must be an integer"
                             " >= 1 and <= train_batch_size, got %s" %
                             self.gradient_accumulation_steps)

        if not isinstance(self.fp16, bool):
            raise ValueError("fp16 must be either True or False, got %s." %
                             self.fp16)

        if not isinstance(self.use_cuda, bool):
            raise ValueError("use_cuda must be either True or False, got %s." %
                             self.fp16)

        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, "
                             "got %s" % self.validation_fraction)

    def fit(self, X, y, load_at_start=True):
        """
        Finetune pretrained Bert model.


        Bert model input are triples : (text_a, text_b, label/target).

        For text pair classification/regression tasks:

            X = [(text_a, text_b)]

        For single text classification/regression tasks:

            X = [text]

        Parameters
        ----------

        X : 1D or 2D Array like list of strings
            Input text or text pair data features

        y : list or Array like  list of string or list of floats):
            Labels/targets for text data

        load_at_start : bool
            load model from saved checkpoint file at the start of the fit

        """
        set_random_seed(self.random_state, self.use_cuda)

        X = np.squeeze(to_numpy(X))
        y = np.squeeze(to_numpy(y))

        texts_a, texts_b = unpack_text_pairs(X)
        labels = y

        self.input_text_pairs = not texts_b is None

        if is_classifier(self):

            # if the label_list not specified, then infer it from y
            if self.label_list is None:
                self.label_list = np.unique(labels)

            self.num_labels = len(self.label_list)
            for (i, label) in enumerate(self.label_list):
                self.label2id[label] = i
                self.id2label[i] = label

        if load_at_start:
            self.load_bert()

        # to fix BatchLayer1D prob in rare case last batch is a singlton w MLP
        drop_last_batch = False if self.num_mlp_layers == 0 else True

        # create a finetune config
        config = FinetuneConfig(
            tokenizer=self.tokenizer,
            model_type=self.model_type,
            epochs=self.epochs,
            max_seq_length=self.max_seq_length,
            learning_rate=self.learning_rate,
            warmup_proportion=self.warmup_proportion,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            label2id=self.label2id,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=self.fp16,
            loss_scale=self.loss_scale,
            local_rank=self.local_rank,
            use_cuda=self.use_cuda,
            train_sampler='random',
            drop_last_batch=drop_last_batch,
            val_frac=self.validation_fraction,
            logger=self.logger)

        # check lengths
        assert len(texts_a) == len(labels)
        if texts_b is not None:
            assert len(texts_a) == len(texts_b)

        # finetune model
        self.model = fit(self.model, texts_a, texts_b, labels, config)

        return self

    def setup_eval(self, texts_a, texts_b, labels, use_cuda=True):
        """
        Get dataloader and device for eval.
        """

        device, _ = get_device(self.local_rank, use_cuda)

        dataset = TextFeaturesDataset(texts_a, texts_b, labels,
                                      self.model_type,
                                      self.label2id,
                                      self.max_seq_length,
                                      self.tokenizer)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 self.eval_batch_size,
                                                 num_workers=5)
        self.model.to(device)
        self.model.eval()
        return dataloader, device

    def score(self, X, y, verbose=True):
        """
        Score model on test/eval data.
        """
        X = np.squeeze(to_numpy(X))
        y = np.squeeze(to_numpy(y))

        texts_a, texts_b = unpack_text_pairs(X)
        labels = y

        dataloader, device = self.setup_eval(texts_a, texts_b,
                                             labels, self.use_cuda)

        res = eval_model(self.model, dataloader, device, self.model_type, "Testing")

        if verbose:
            print("\nTest loss: %0.04f, Test accuracy = %0.02f%%"%res)
        return res[1]

    def save(self, filename):
        """
        Save model state to disk.
        """
        # Only save the model it-self
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        state = {
            'params': self.get_params(),
            'class_name' : type(self).__name__,
            'model_type' : self.model_type,
            'num_labels' : self.num_labels,
            'id2label'   : self.id2label,
            'label2id'   : self.label2id,
            'state_dict' : model_to_save.state_dict(),
            'input_text_pairs' : self.input_text_pairs
        }
        torch.save(state, filename)

    def restore_finetuned_model(self, restore_file):
        """
        Restore a previously finetuned model from a restore_file

        This is called from the BertClassifier or BertRegressor. The saved model
        is a finetuned BertPlusMLP
        """

        print("Loading model from %s..."%(restore_file))
        state = torch.load(restore_file)

        params = state['params']

        bert_model = params['bert_model']
        num_mlp_layers = params['num_mlp_layers']
        num_mlp_hiddens = params['num_mlp_hiddens']

        model_type = state['model_type']
        num_labels = state['num_labels']

        do_lower_case = True if 'uncased' in bert_model else False

        self.tokenizer = BertTokenizer.from_pretrained(bert_model,
                                                       do_lower_case=do_lower_case)
        model_state_dict = state['state_dict']

        self.model = BertPlusMLP.from_pretrained(bert_model,
                                                 state_dict=model_state_dict,
                                                 num_labels=num_labels,
                                                 model_type=model_type,
                                                 num_mlp_layers=num_mlp_layers,
                                                 num_mlp_hiddens=num_mlp_hiddens)
        params = state['params']
        self.set_params(**params)
        self.input_text_pairs = state['input_text_pairs']
        self.id2label = state['id2label']
        self.label2id = state['label2id']

    def tune_params(self,
                    X_train, y_train,
                    X_val, y_val,
                    learning_rate=None,
                    epochs=None,
                    train_batch_size=None,
                    max_seq_length=None,
                    num_mlp_layers=None,
                    rand_states=None):
        """
        Tune model over parameter grid.

        The Google BERT paper recommends searching over:
            train_batch_size = [16, 32]
            learning rate = [5e-5, 3e-5, 2e-5]
            epochs = [3, 4]

        For larger datasets I think its worth trying the MLP with:
            num_mlp_layers = [0, 1, 4]
        """

        if learning_rate is None:
            learning_rate = [5e-5, 3e-5, 2e-5]

        if epochs is None:
            epochs = [3, 4]

        if train_batch_size is None:
            train_batch_size = [16]

        if max_seq_length is None:
            max_seq_length = [96]

        if num_mlp_layers is None:
            num_mlp_layers = [0]

        if rand_states is None:
            rand_states = [42, 134, 6, 8]

        params = ParameterGrid({
            'learning_rate': learning_rate,
            'epochs': epochs,
            'train_batch_size': train_batch_size,
            'max_seq_length': max_seq_length,
            'num_mlp_layers': num_mlp_layers,
        })

        scores = {}

        # loop over all parameter combinations
        for param in params:
            param_list = tuple(param.items())
            scores[param_list] = []
            print("="*60)
            for k, v in param.items():
                print("%s: %r"%(k, v))
            print("="*60)
            # loop over random states
            for r in rand_states:
                print("Using random seed :", r)
                if self.model_type == 'classifier':
                    model = BertClassifier(**param)
                elif self.model_type == 'regressor':
                    model = BertRegressor(**param)

                model.validation_fraction = 0
                model.random_state = r

                #fit and score model
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores[param_list].append(score)
                print("score: %0.2f\n"%(score))

        # print results
        for param, score in scores.items():
            if len(score) > 1:
                mean = stats.mean(score)
                std = stats.stdev(score)
                print("%0.3f (+/-%0.03f) for %s"% (mean, std * 2, dict(param)))

        # find best model
        lis = [(stats.mean(score), dict(param), score) for param, score in scores.items()]
        best_score, best_param, best_scores = max(lis, key=lambda item: item[0])
        print("Best mean score is %0.2f, with params: %s"%(best_score, str(best_param)))

        #refit model with best params
        if self.model_type == 'classifier':
            model = BertClassifier(**best_param)
        elif self.model_type == 'regressor':
            self.model = BertRegressor(**best_param)
        model.validation_fraction = 0

        # choose the best random_state
        idx = best_scores.index(max(best_scores))
        model.random_state = rand_states[idx]

        # fit best model
        model.fit(X_train, y_train)

        return {'best_model': model, 'best_param': best_param,
                'best_score': best_score, 'scores': scores}


class BertClassifier(BaseBertEstimator, ClassifierMixin):
    """
    A classifier built on top of a pretrained Bert model.
    """

    def predict_proba(self, X, use_cuda=True):
        """
        Make class probability predictions.
        """

        X = np.squeeze(to_numpy(X))
        texts_a, texts_b = unpack_text_pairs(X)

        dataloader, device = self.setup_eval(texts_a, texts_b, None, use_cuda)
        probs = []
        batch_iter = tqdm(dataloader, desc="Predicting", leave=False)
        for batch in batch_iter:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = self.model(*batch)
                prob = F.softmax(logits, dim=-1)
            prob = prob.detach().cpu().numpy()
            probs.append(prob)
        return np.vstack(tuple(probs))

    def predict(self, X, use_cuda=True):
        """
        Predict most probable class.
        """
        y_pred = np.argmax(self.predict_proba(X, use_cuda=use_cuda), axis=1)
        y_pred = np.array([self.id2label[y] for y in y_pred])
        return y_pred


class BertRegressor(BaseBertEstimator, RegressorMixin):
    """
    A regressor built on top of a pretrained Bert model.
    """

    def predict(self, X, use_cuda=True):
        """
        Predict method for regression.
        """

        X = np.squeeze(to_numpy(X))
        texts_a, texts_b = unpack_text_pairs(X)

        dataloader, device = self.setup_eval(texts_a, texts_b, None, use_cuda)

        ypred_list = []
        batch_iter = tqdm(dataloader, desc="Predicting", leave=False)
        for batch in batch_iter:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                y_pred = self.model(*batch)
            ypred_list.append(y_pred.detach().cpu().numpy())
        y_pred = np.vstack(tuple(ypred_list)).reshape(-1,)
        return y_pred


def load_model(filename):
    """
    Load BertClassifier or BertRegressor from a disk file.
    """
    state = torch.load(filename)
    class_name = state['class_name']

    classes = {
        'BertClassifier': BertClassifier,
        'BertRegressor' : BertRegressor}

    # call the constructor to load the model
    model_ctor = classes[class_name]
    model = model_ctor(restore_file=filename)
    return model
