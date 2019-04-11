import os
import sys
import csv

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report

from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import load_model

DATADIR = 'tests/data'

def setup_function(function):
    print ("\n" + "="*75)


def teardown_function(function):
    print ("")


def get_sst_data(train_file=DATADIR + '/sst2/train.tsv', 
                 dev_file=DATADIR + '/sst2/dev.tsv'):

    train = pd.read_csv(train_file, sep='\t',  encoding='utf8', keep_default_na=False)
    train.columns=['text','label']

    dev = pd.read_csv(dev_file, sep='\t',  encoding='utf8', keep_default_na=False)
    dev.columns=['text', 'label']

    label_list = np.unique(train['label'])

    return train, dev, label_list   


def sst2_test_data():
    train,dev,label_list = get_sst_data()

    X_train = train['text']
    y_train = train['label']
    X_dev = dev['text']
    y_dev = dev['label']

    return X_train, y_train, X_dev, y_dev


def test_bert_sklearn_accy():
    """
    Test bert_sklearn accuracy 
    compare against  huggingface run_classifier.py 
    on 200 rows of SST-2 data.
    """
    print("Running bert-sklearn...")                            
    X_train, y_train, X_dev, y_dev = sst2_test_data()

    # define model
    model = BertClassifier()
    model.validation_fraction = 0.0
    model.learning_rate = 5e-5 
    model.gradient_accumulation_steps = 2
    model.max_seq_length = 64
    model.train_batch_size = 16
    model.eval_batch_size = 8
    model.epochs = 2

    model.fit(X_train,y_train)

    bert_sklearn_accy = model.score(X_dev,y_dev)
    bert_sklearn_accy /= 100

    # run huggingface BERT run_classifier and check we get the same accuracy
    cmd = r"python tests/run_classifier.py --task_name sst-2 \
                                --data_dir ./tests/data/sst2 \
                                --do_train  --do_eval \
                                --output_dir . \
                                --bert_model bert-base-uncased \
                                --do_lower_case \
                                --learning_rate 5e-5 \
                                --gradient_accumulation_steps 2 \
                                --max_seq_length 64 \
                                --train_batch_size 16 \
                                --eval_batch_size 8 \
                                --num_train_epochs 2"
                                
    print("\nRunning huggingface run_classifier.py...\n")                            
    os.system(cmd)
    print("...finished run_classifier.py\n")  

    # parse run_classifier.py output file and find the accy
    accy = open("eval_results.txt").read().split("\n")[0] # 'acc = 0.76'
    accy = accy.split("=")[1]
    accy = float(accy)
    print("bert_sklearn accy: %.02f, run_classifier.py accy : %0.02f"%(bert_sklearn_accy,accy))

    # clean up 
    print("\nCleaning up eval file: eval_results.txt")
    os.remove("eval_results.txt")

    assert bert_sklearn_accy == accy


def test_save_load_model():
    """Test saving/loading a fitted model to disk"""

    X_train,y_train,X_dev,y_dev = sst2_test_data()

    model = BertClassifier()
    model.max_seq_length = 64
    model.train_batch_size = 8
    model.epochs= 1

    model.fit(X_train,y_train)

    accy1 = model.score(X_dev,y_dev)

    savefile='./test_model_save.bin'
    print("\nSaving model to ",savefile)

    model.save(savefile)

    # load model from disk
    new_model = load_model(savefile)

    # predict with new model
    accy2 = new_model.score(X_dev, y_dev )

    # clean up 
    print("Cleaning up model file: test_model_save.bin ")
    os.remove(savefile)

    assert accy1 == accy2


def test_regression():
    """Test on regression data w array inputs"""

    train = pd.read_csv(DATADIR + "/stsb/train.csv")
    X_train = train[['text_a','text_b']]
    y_train = train['label']
    
    X_train = X_train.values
    y_train = y_train.values

    model = BertRegressor()
    model.validation_fraction = 0.0
    model.max_seq_length = 64
    model.train_batch_size = 16
    model.eval_batch_size = 8
    model.epochs = 1

    model.fit(X_train,y_train)

    accy = model.score(X_train,y_train)

    assert accy <= 100
    

def test_nonbinary_classify():
    """Test non-binary classification w list inputs"""

    train = pd.read_csv(DATADIR + "/mnli/train.csv")
    X_train = train[['text_a','text_b']]
    y_train = train['label']

    X_train = list(X_train.values)
    y_train = list(y_train.values)

    # define model
    model = BertClassifier()
    model.validation_fraction = 0.0
    model.max_seq_length = 64
    model.train_batch_size = 16
    model.eval_batch_size = 8
    model.epochs = 1

    model.fit(X_train,y_train)

    accy = model.score(X_train,y_train)

    assert accy <= 100


"""
# longer tests...    
def test_large_model_load():
    
    X_train,y_train, X_dev, y_dev =  sst2_test_data()

    
    model = BertClassifier(bert_model='bert-large-uncased',
                           epochs=1,
                           validation_fraction=0,
                           max_seq_length=16,
                           train_batch_size=4)

    # fit model
    model.fit(X_train, y_train)

    # score model
    accy = model.score(X_dev, y_dev)
        
    assert accy<=100

def test_all_model_loads():

    X_train,y_train, X_dev, y_dev =  sst2_test_data()

    SUPPORTED_MODELS = ('bert-base-uncased','bert-large-uncased','bert-base-cased',
                    'bert-large-cased','bert-base-multilingual-uncased',
                    'bert-base-multilingual-cased', 'bert-base-chinese')


    for bert_model in SUPPORTED_MODELS:
        print("="*60)
        model = BertClassifier(bert_model='bert-large-uncased',
                               epochs=1,
                               validation_fraction=0,
                               max_seq_length=16,
                               train_batch_size=4)

        # fit model
        model.fit(X_train, y_train)

        # score model
        accy = model.score(X_dev, y_dev)
        
    assert accy<=100   
"""    
