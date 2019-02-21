# scikit-learn wrapper for BERT
A scikit-learn model for text classification/regression based on the [huggingface pytorch](https://github.com/huggingface/pytorch-pretrained-BERT) port of Google's [BERT](https://github.com/google-research/bert)(Bidirectional Encoder Representations from Transformers) model.

* Added an MSE loss for regression tasks
* Added configurable MLP as final classifier/regressor

## basic operation

**`model.fit(X,y)`** where

* **`X`**: list, pandas dataframe, or numpy array of text or text pairs

* **`y`** : list, pandas dataframe, or numpy array of labels/targets

```python3
from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import load_model

# define model
model = BertClassifier()   # for classification 
# model = BertRegressor()  # for regression 
 
# fit model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# make probabilty predictions
y_pred = model.predict_proba(X_test)

# score model on test data
model.score(X_test, y_test)

# save model to disk
savefile='/data/mymodel.bin'
model.save(savefile)

# load model from disk
new_model = load_model(savefile)

# do stuff with new model
new_model.score(X_test, y_test)
```
See [demo](https://github.com/charles9n/bert-sklearn/blob/master/demo.ipynb) notebook.

## model options

```python3
# try different options...
model.bert_model = 'bert-large-uncased'
model.num_mlp_layers = 3
model.max_seq_length = 196
model.epochs = 4
model.learning_rate = 4e-5
model.gradient_accumulation_steps = 4

# fit model
model.fit(X_train, y_train)

# do stuff...
model.score(X_test, y_test)
```
See [options](https://github.com/charles9n/bert-sklearn/blob/master/Options.md)


## hyperparameter tuning

```python3
from sklearn.model_selection import GridSearchCV

params = {'epochs':[3, 4], 'learning_rate':[2e-5, 3e-5, 5e-5]}

# wrap classifier in GridSearchCV
clf = GridSearchCV(BertClassifier(validation_fraction=0), 
                    params,
                    scoring='accuracy',
                    verbose=True)

# fit gridsearch 
clf.fit(X_train ,y_train)
```
See [demo_tuning_hyperparameters](https://github.com/charles9n/bert-sklearn/blob/master/demo_tuning_hyperparams.ipynb) notebook.

## GLUE datasets
The train and dev data sets from the [GLUE(Generalized Language Understanding Evaluation) ](https://github.com/nyu-mll/GLUE-baselines) benchmarks were used with `bert-base-uncased` model and compared againt the reported results in the Google paper and [GLUE leaderboard](https://gluebenchmark.com/leaderboard).

|    | MNLI(m/mm)| QQP   | QNLI | SST-2| CoLA | STS-B | MRPC | RTE |
| - | - | - | - | - |- | - | - | - |
|BERT base(leaderboard) |84.6/83.4  | 89.2 | 90.1 | 93.5 | 52.1 | 87.1  | 84.8 | 66.4 | 
| bert-sklearn  |83.7/83.9| 90.2 |88.6 |92.32 |58.1| 89.7 |86.8 | 64.6 |

Individual runs can be found can be found [here](https://github.com/charles9n/bert-sklearn/tree/master/glue_examples).


## installation

requires python >= 3.5 and pytorch >= 0.4.1

```bash
# install pytorch-pretrained-bert from PyPI
pip install pytorch-pretrained-bert

# setup bert-sklearn locally
git clone -b master https://github.com/charles9n/bert-sklearn
cd bert-sklearn
pip install .
```

## references

* [Google's original tf code](https://github.com/google-research/bert)  and [paper](https://arxiv.org/abs/1810.04805)

* [huggingface pytorch port](https://github.com/huggingface/pytorch-pretrained-BERT)

