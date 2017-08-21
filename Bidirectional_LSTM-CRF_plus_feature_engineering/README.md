## Bidirectional_LSTM-CRF_plus_feature_engineering

This code is almost identical to the one presented in the [Bidirectional_LSTM-CRF file]. The only difference is that it gives the option to include  the hand-crafted features presented in the paper (further description below in the __Train a model__ section.


## Initial setup

To use the tagger, you need Python 2.7, with Numpy, NLTK and Theano installed.


## Tag sentences

The fastest way to use the tagger is to use one of the pretrained models:

```
./tagger.py --model models/english/ --input input.txt --output output.txt
```

The input file should contain one sentence by line, and they have to be tokenized. Otherwise, the tagger will perform poorly.


## Train a model

To train your own model, you need to use the train.py script and provide the location of the training, development and testing set:

```
./train.py --train train.txt --dev dev.txt --test test.txt
```

The training script will automatically give a name to the model and store it in ./models/
There are many parameters you can tune (CRF, dropout rate, embedding dimension, LSTM hidden layer size, etc). To see all parameters, simply run:

```
./train.py --help
```

__Note!__ As we explain in the introduction of this page, the only difference between this code and the [Bidirectional_LSTM-CRF] is the option to concatenate hand-crafted features with the word embeddings as input of the model. There are tow type of features in the code:

```
--morph_features_dim 1 --semantic_features_dim 1
```
Note that in the paper we have not make this distinction. The reason behind is that we have always used both features together. If you want to use the features, you only need to set that option to 1 (the features are already in the train, dev and test files).

Input files for the training script have to follow the same format than the CoNLL2003 sharing task: each word has to be on a separate line, and there must be an empty line after each sentence. A line must contain at least 2 columns, the first one being the word itself, the last one being the named entity. It does not matter if there are extra columns that contain tags or chunks in between. Tags have to be given in the IOB format (it can be IOB1 or IOB2).


[Bidirectional_LSTM-CRF file]: https://github.com/ijauregiCMCRC/healthNER/tree/master/Bidirectional_LSTM-CRF
