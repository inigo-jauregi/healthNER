#***************README***********************************************
#
#The following code creates a suitable tags file
#for the package HCRF. The structure of the tag file is:
#
#
#Data format of labels files:  labelsTrain.csv and labelsTest.csv
#
#1 5 % first sequence has 5 frames (tokens) and only 1 dimension (time)
#4 2 1 1 1 % labels of the 5 tokens
#1 7 % second sequence has 7 tokens
#1 1 1 4 7 9 1 % labels of the 7 tokens

#Note! In order to have the same IDs for the tags, the training file is always executed first, and after the
#test file, when the mappings have been already created

import csv
import cPickle as pickle
import os

#First read the file
train=open("data/DrugBank/train_plus_features.txt","r")
dev=open("data/DrugBank/dev_plus_features.txt","r")
test=open("data/DrugBank/test_plus_features.txt","r")
#Open a '.csv' file to save the data
csvfile=open("results/DrugBank/labelsTrain.csv","w")
csv_writer=csv.writer(csvfile,delimiter=',',quotechar='"',lineterminator='\n')
csvfileTest=open("results/DrugBank/labelsTest.csv","w")
csv_writerTest=csv.writer(csvfileTest,delimiter=',',quotechar='"',lineterminator='\n')

#Create a dictionary of the tags
if (os.path.exists("/home/ijauregi/Desktop/CMCRC/CRF/data_preparation/src/data/DrugBank/mappings_tags.p")):
    print "Mappings loaded"
    dico_tags = pickle.load(open('data/DrugBank/mappings_tags.p', 'rb'))
else:
    dico_tags={} #At the beginning is empty if it is training

#Counter for the tokens in each line
token_counter=0
sentence_tags=[]

for word in train:
    #Clean data
    word=word.replace("\t"," ")
    word=word.replace("\n","")

    if (len(word.split(" "))==6):
        # Increment sentence length
        token_counter += 1
        #Save the word
        token=word.split(" ")[0]
        #Check the tag
        tag_name=word.split(" ")[-1]
        #If the tag is new
        if (tag_name not in dico_tags):
            dico_tags[tag_name]=len(dico_tags)
        tag=dico_tags[tag_name]
        sentence_tags.append(tag)
        #Check the end of the sentence
    else:
        csv_writer.writerow(['1',token_counter])
        csv_writer.writerow(sentence_tags)
        token_counter=0
        sentence_tags=[]

#When it is test delete
for word in dev:
    #Clean data
    word=word.replace("\t"," ")
    word=word.replace("\n","")

    if (len(word.split(" "))==6):
        # Increment sentence length
        token_counter += 1
        #Save the word
        token=word.split(" ")[0]
        #Check the tag
        tag_name=word.split(" ")[-1]
        #If the tag is new
        if (tag_name not in dico_tags):
            dico_tags[tag_name]=len(dico_tags)
        tag=dico_tags[tag_name]
        sentence_tags.append(tag)
        #Check the end of the sentence
    else:
        csv_writer.writerow(['1',token_counter])
        csv_writer.writerow(sentence_tags)
        token_counter=0
        sentence_tags=[]


for word in test:
    #Clean data
    word=word.replace("\t"," ")
    word=word.replace("\n","")

    if (len(word.split(" "))==6):
        # Increment sentence length
        token_counter += 1
        #Save the word
        token=word.split(" ")[0]
        #Check the tag
        tag_name=word.split(" ")[-1]
        #If the tag is new
        if (tag_name not in dico_tags):
            dico_tags[tag_name]=len(dico_tags)
        tag=dico_tags[tag_name]
        sentence_tags.append(tag)
        #Check the end of the sentence
    else:
        csv_writerTest.writerow(['1',token_counter])
        csv_writerTest.writerow(sentence_tags)
        token_counter=0
        sentence_tags=[]

if not os.path.exists("/home/ijauregi/Desktop/CMCRC/CRF/data_preparation/src/data/DrugBank/mappings_tags.p"):
    pickle.dump( dico_tags, open( "data/DrugBank/mappings_tags.p", "wb" ) )
    print "Mappings Saved"

