# CRF:

The CRF model is the baseline conventional model of the paper. We use it to have a comparison between a neural network approach and a more traditional approach. For the implementation of the CRF we have used the HCRF open-source tool.

In order to reproduce the results obtained with the CRF, we need to follow the workflow presented in the image below:

![crf_flowchart](https://user-images.githubusercontent.com/23091295/29503550-dc66106c-867b-11e7-8d74-ed61e56e389e.jpg)

__1. data_preparation__: First, the training and test data needs to be prepared in the HCRF format with the data_preparation code. The file contains two main '.py' files for that: _data_CRF_file_creation.py_ and _using tag_CRF_file_creation.py_. 

The first file (_data_CRF_file_creation.py_), creates the files of features (dataTrain.csv and dataTest.csv) for the CRF model in the following format:

_3    2    %first sequence has 2 tokens (each token(t) has 3 features(f))_

_0.1    0.8    %f1 of t1, f1 of t2_

_0.8    0.9    %f2 of t1, f2 of t2_

_0.4    0.6    %f3 of t1, f3 of t2__

The second file (_tag_CRF_file_creation.py_), creates the files of labels (labelTrain.csv and labelTest.csv) to train and test the CRF model in the following format:

_1 5 %first sequence has 5 tokens and only 1 dimension (time)_

_4 2 1 1 1 %labels of the 5 tokens_

_1 7 %second sequence has 7 tokens and only 1 dimension (time)_

_1 1 1 4 7 9 1 %labels of the 7 tokens_

At the end of the data_preparation we must have created 4 files: _dataTrain.csv_, _labelsTrain.csv_, _dataTest.csv_ and _labelsTest.csv_.

__2. train and test the CRF model__: Second, the CRF model is trained with the HCRF toolkit. The HCRF receives the previously generated files in the correct format and with the following command, it trains and tests a CRF model:

```
./hcrfTest -t -d dataTrain.csv -l labelsTrain.csv -m model.txt -T -D dataTest.csv -L labelsTest.csv -r results.txt -c stats.txt -f features.txt -a crf -h 3 -P 4 -p 4 -i 10 -w 1
```

After the training, the model will be stored in the file model.txt, and after the test the predictions will be stored in the results.txt file. For further information about the HCRF tool read the [guideline].

__3. evaluate the model__: Finally, we need to evaluate the predictions made by the CRF model and stored in the results.txt file with the conlleval evaluation script, which is a "strict" evaluation (for more information read our paper). First, we convert results.txt to a "conlleval" evaluation format with the data_conll_preparation.py file (inside the data_preparation folder). Finally, we evaluate the converted results using the conlleval.pl perl file with the following command:

```
perl conlleval < converted_results.txt > final_results.txt
```

These are the final results of the model.

[guideline]: https://sourceforge.net/projects/hcrf/
