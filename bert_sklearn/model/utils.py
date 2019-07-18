import torch

from .pytorch_pretrained import BertTokenizer, BasicTokenizer
from .pytorch_pretrained import PYTORCH_PRETRAINED_BERT_CACHE

from .model import BertPlusMLP

def get_basic_tokenizer(do_lower_case):
    """
    Get a  basic tokenizer(punctuation splitting, lower casing, etc.).
    """
    return BasicTokenizer(do_lower_case=do_lower_case)


def get_tokenizer(bert_model='bert-base-uncased',
                  bert_vocab_file=None,
                  do_lower_case=False):
    """
    Get a BERT wordpiece tokenizer.

    Parameters
    ----------
    bert_model : string
        one of SUPPORTED_MODELS i.e 'bert-base-uncased','bert-large-uncased'
    bert_vocab_file: string
        Optional pathname to vocab file to initialize BERT tokenizer
    do_lower_case : bool
        use lower case with tokenizer

    Returns
    -------
    tokenizer : BertTokenizer
        Wordpiece tokenizer to use with BERT
    """
    if bert_vocab_file is not None:
        return BertTokenizer(bert_vocab_file, do_lower_case=do_lower_case)
    else:
        return BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)


def get_model(bert_model='bert-base-uncased',
              bert_config_json=None,
              from_tf=False,
              num_labels=2,
              model_type='classifier',
              num_mlp_layers=0,
              num_mlp_hiddens=500,
              state_dict=None,
              local_rank=-1):
    """
    Get a BertPlusMLP model.

    Parameters
    ----------
    bert_model : string
        one of SUPPORTED_MODELS i.e 'bert-base-uncased','bert-large-uncased'
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
    state_dict : collections.OrderedDict object
         an optional state dictionnary
    local_rank : (int)
        local_rank for distributed training on gpus

    Returns
    -------
    model : BertPlusMLP
        BERT model plus mlp head
    """

    cache_dir = PYTORCH_PRETRAINED_BERT_CACHE/'distributed_{}'.format(local_rank)

    if bert_config_json is not None:
        # load from a tf checkpoint file, pytorch checkpoint file,
        # or a pytorch state dict
        model = BertPlusMLP.from_model_ckpt(config_file_or_dict=bert_config_json,
                                            weights_path=bert_model,
                                            state_dict=state_dict,
                                            from_tf=from_tf,
                                            num_labels=num_labels,
                                            model_type=model_type,
                                            num_mlp_hiddens=num_mlp_hiddens,
                                            num_mlp_layers=num_mlp_layers)
    else:
        # Load from pre-trained model archive
        print("Loading %s model..."%(bert_model))
        model = BertPlusMLP.from_pretrained(bert_model,
                                            cache_dir=cache_dir,
                                            state_dict=state_dict,
                                            num_labels=num_labels,
                                            model_type=model_type,
                                            num_mlp_hiddens=num_mlp_hiddens,
                                            num_mlp_layers=num_mlp_layers)

    return model
