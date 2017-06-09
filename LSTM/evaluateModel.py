#!/usr/bin/env python

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader

from utils import models_path, evaluate, eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model import Model

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="data_DER/DrugBank/NER/train.txt",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="data_DER/DrugBank/NER/dev.txt",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="data_DER/DrugBank/NER/test.txt",
    help="Test set location"
)
opts = optparser.parse_args()[0]

# Parse parameters
# Initialize model
model = Model( models_path=None, model_path="models/modelDER_MedLine_LSTMchar25_LSTMword100_Dropout0.5_TagIOB_LearnSGD_Glove_mimic_600_2")
print "Model location: %s" % model.model_path
parameters = model.parameters

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()

print "Calling the prepare_dataset :--"
# Index data
#Prepare the data correctly to use it later (not in the build model!!!)
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

test_score,test_eval_lines_prob = evaluate(parameters, f_eval, test_sentences,
                                  test_data, model.id_to_tag, dico_tags)

final_test_lines=""
for line in test_eval_lines_prob:
    final_test_lines=final_test_lines+"\n"+line
print final_test_lines