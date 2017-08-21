# CRF:

The CRF model is the baseline conventional model of the paper. We use it to have a comparison between a neural network approach and a more traditional approach. For the implementation of the CRF we have used the HCRF open-source tool.

In order to reproduce the results obtained with the CRF, we need to follow the workflow presented in the image below:

![crf_flowchart](https://user-images.githubusercontent.com/23091295/29344532-e7a7a426-827b-11e7-9cae-d6870c8fbdd5.jpg)

First, the training and test data needs to be prepared in the HCRF format with the data_preparation code. The file contains two main '.py' files for that: _data_CRF_file_creation.py_ and _using tag_CRF_file_creation.py_. 

The firs file (_data_CRF_file_creation.py_), creates a file of the features for the CRF model in the following format:

3    2    %first sequence has 2 tokens (each token(t) has 3 features(f))

0.1    0.8    %f1 of t1, f1 of t2

0.8    0.9    %f2 of t1, f2 of t2

0.4    0.6    %f3 of t1, f3 of t2

to create the input files for the CRF model in a specific format dataTrain.csv, dataTest.csv (using , labelsTrain.csv and labelsTest.csv () with the data_preparation python code. 

Second, we train and test the CRF model with those files (see the HCRF guideline).

Finally, we obtain the results.txt file from the HCRF software. We convert results.txt to a "conlleval" evaluation format (data_conll_preparation.py) and evaluate using the conlleval.pl perl file.


