#Prepare the output of the HCRF module to
#obtain the conll-eval score
#
#The structure of the data we need is the following:
#   Word      Grount_truth       Prediction
#---------------------------------------------
#    x          B-Problem             O
#
#Note: There is a need to have spaces between sentences!

import cPickle as pickle
dico_tags=pickle.load(open('data/MedLine/mappings_tags.p','rb'))

id_to_tag={}
#Create inverse mappings
for tag in dico_tags:
    id_to_tag[dico_tags[tag]]=tag

#File to read
CRF_out=open('results/MedLine/only_feat/results.txt','r')
#File to write
conll_data=open('results/MedLine/only_feat/conll_results.txt','w')

#Number of tags
tag_number=len(id_to_tag)
token_number=0
tag_counter=0
token_counter=0

for line in CRF_out:
    #Obtain data from first line
    if (token_number==0):
        line=line.replace("\n","")
        line=line.split(",")
        token_number=int(line[1])
    #Ignore the probabilities
    elif (tag_counter!=tag_number):
        tag_counter+=1
    #Obtain the ground truth and the prediction for each token
    elif (token_counter!=token_number-1):
        token_counter+=1
        line = line.replace("\t", " ")
        line=line.split(" ")
        write_sent="x "+id_to_tag[int(line[0])]+" "+id_to_tag[int(line[1])]+"\n"
        conll_data.write(write_sent)
    else:
        line = line.replace("\t", " ")
        line = line.split(" ")
        write_sent = "x " + id_to_tag[int(line[0])] + " " + id_to_tag[int(line[1])]+"\n"
        conll_data.write(write_sent)
        conll_data.write("\n") #To differenciate the sentence
        token_number=0
        tag_counter=0
        token_counter=0

conll_data.close()
