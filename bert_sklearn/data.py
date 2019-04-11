import torch
from torch.utils.data import Dataset


class TextFeatures:
    """
    A single set of input features for the Bert model.
    """
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.

    From pytorch-pretrained-BERT/examples/run_classifier.py
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_text_to_features(text_a, text_b, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s.

    Adapted from 'convert_examples_to_features' in
    pytorch-pretrained-BERT/examples/run_classifier.py

    """

    tokens_a = tokenizer.tokenize(text_a)

    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return TextFeatures(input_ids, input_mask, segment_ids)


class TextFeaturesDataset(Dataset):
    """
    A pytorch dataset for Bert text features.

    Parameters
    ----------

    X1 : list of strings
        text_a for input data
    X2 : list of strings
        text_b for input data text pairs
    y : list of string or list of floats
        labels/targets for data
    model_type : string
        specifies 'classifier' or 'regressor' model
    label2id : dict map of string to int
        label map for classifer labels
    max_seq_length : int
        maximum length of input text sequence (text_a + text_b)
    tokenizer : BertTokenizer)
        word tokenizer followed by WordPiece Tokenizer
    """
    def __init__(self,
                 X1, X2, y,
                 model_type,
                 label2id,
                 max_seq_length,
                 tokenizer):

        self.X1 = X1
        self.X2 = X2
        self.y = y

        self.len = len(self.X1)
        self.model_type = model_type
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        if self.X2 is not None:
            text_a = str(self.X1[index])
            text_b = str(self.X2[index])
        else:
            text_a = str(self.X1[index])
            text_b = None

        feature = convert_text_to_features(text_a, text_b,
                                           self.max_seq_length,
                                           self.tokenizer)

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)

        if self.y is not None:

            label = self.y[index]

            if self.model_type == 'classifier':
                label_id = self.label2id[label]
                target = torch.tensor(label_id, dtype=torch.long)
            elif self.model_type == 'regressor':
                target = torch.tensor(label, dtype=torch.float32)
            return input_ids, segment_ids, input_mask, target
        else:
            return input_ids, segment_ids, input_mask

    def __len__(self):
        return self.len
