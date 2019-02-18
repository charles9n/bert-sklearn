
##  GLUE Examples


The train and dev data sets from the [GLUE(Generalized Language Understanding Evaluation) ](https://github.com/nyu-mll/GLUE-baselines) benchmarks were used with `bert-base-uncased` model and compared againt the reported results in the Google paper and [GLUE leaderboard](https://gluebenchmark.com/leaderboard). The GLUE leaderboard scores are based on the test sets in which the labels are private. 

The scores are all classification accuracy, except for `STS-B` where  the score is Pearson correlation, and 
 `CoLA` where the score is Mathews correlation.

|    | MNLI(m/mm)| QQP   | QNLI | SST-2| CoLA | STS-B | MRPC | RTE |
| - | - | - | - | - |- | - | - | - |
|BERT base(leaderboard) |84.6/83.4  | 89.2 | 90.1 | 93.5 | 52.1 | 87.1  | 84.8 | 66.4 | 
| bert-sklearn  |83.7/83.9| 90.2 |88.6 |92.32 |58.1| 89.7 |86.8 | 64.6 |




The individual runs can be found here:

* [Corpus of Linguistic Acceptability(CoLA)](https://github.com/charles9n/bert-sklearn/blob/master/glue_examples/CoLA.ipynb)

* [Microsoft Research Paraphrase Corpus(MRPC)](https://github.com/charles9n/bert-sklearn/blob/master/glue_examples/MRPC.ipynb)

* [Recognizing Textual Entailment(RTE)](https://github.com/charles9n/bert-sklearn/blob/master/glue_examples/RTE.ipynb)

* [Multi-Genre Natural Language Inference(MNLI)](https://github.com/charles9n/bert-sklearn/blob/master/glue_examples/MNLI.ipynb)

* [Quora Question Pair(QQP)](https://github.com/charles9n/bert-sklearn/blob/master/glue_examples/QQP.ipynb)

* [Question Natural Language Inference(QNLI)](https://github.com/charles9n/bert-sklearn/blob/master/glue_examples/QNLI.ipynb)

* [Stanford Sentiment Treebank (SST-2)](https://github.com/charles9n/bert-sklearn/blob/master/glue_examples/SST-2.ipynb)

* [Semantic Textual Similarity Benchmark (STS-B)](https://github.com/charles9n/bert-sklearn/blob/master/glue_examples/STS-B.ipynb)








