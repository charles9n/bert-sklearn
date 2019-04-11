"""Configuration parameters for finetuning."""

class FinetuneConfig:
    """
    Parameters used in finetuning BERT.

    Parameters
    ----------

    data input processing params
    =============================
    tokenizer : BertTokenizer
        Wordpiece tokenizer to use with BERT
    max_seq_length : int
        maximum length of input text sequence (text_a + text_b)
    train_sampler : string
        training sampling strategy
    drop_last_batch : bool
        drop last batch in training
    val_frac : float
        fraction of training set to use for validation
    label2id : dict
        label to id dict for classifiers

    model params
    ============
    model_type : string
        "classifier" or "regressor"

    training params
    ===============
    epochs : int
        number of finetune training epochs
    learning_rate :float
        inital learning rate of Bert Optimizer
    warmup_proportion : float
        proportion of training to perform learning rate warmup
    train_batch_size : int
        batch size for training
    eval_batch_size : int
        batch_size for validationn
    gradient_accumulation_steps : int
        number of update steps to accumulate before performing a backward/update pass

    device params
    =============
    local_rank : int
        local_rank for distributed training on gpus
    fp16 : bool
        whether to use 16-bit float precision instead of 32-bit
    loss_scale : float
        loss scaling to improve fp16 numeric stability. Only used when
        fp16 set to True
    use_cuda : bool
        use GPU(s) if available

    Other
    =======
    logger : python logger
        logger to send logging messages to

    """
    def __init__(self, tokenizer=None, max_seq_length=64, train_sampler='random',
                 drop_last_batch=False, val_frac=0.15, label2id=None,
                 model_type="classifier", epochs=1, learning_rate=2e-5,
                 warmup_proportion=0.1, train_batch_size=32, eval_batch_size=8,
                 gradient_accumulation_steps=1, local_rank=-1, fp16=False,
                 loss_scale=0, use_cuda=True, logger=None):

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_sampler = train_sampler
        self.drop_last_batch = drop_last_batch
        self.val_frac = val_frac
        self.label2id = label2id
        self.model_type = model_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.local_rank = local_rank
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.use_cuda = use_cuda
        self.logger = logger

    def __repr__(self):
        attrs = ["%s=%s"%(key, val) for key, val in vars(self).items()]
        attrs = ",".join(attrs)
        return f'{self.__class__.__name__}({attrs})'
