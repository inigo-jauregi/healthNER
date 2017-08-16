CRF:

It includes the data preparation files for the HCRF2.0b software to train a CRF model.

First, we need to create dataTrain.csv, dataTest.csv (using data_CRF_file_creation.py), labelsTrain.csv and labelsTest.csv (using tag_CRF_file_creation.py) with the data preparation python code.

Second, we train and test the CRF model with those files (see the HCRF guideline).

Finally, we obtain the results.txt file from the HCRF software. We convert results.txt to a "conlleval" evaluation format (data_conll_preparation.py) and evaluate using the conlleval.pl perl file.
