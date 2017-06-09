#The following code creates a suitable data file
#for the package HCRF. The structure of the data file is:
#
#
#3	2 % first sequence has 2 tokens (each token(t) has 3 features(f))
#0.1	0.8	 % f1 of t1, f1 of t2
#0.8	0.9	% f2 of t1, f2 of t2
#0.4	0.6	% f3 of t1, f3 of t2
#3	3	% second sequence has 3 tokens (each token(t) has 3 features(f))
#0.8	0.8	0.1	% f1 of t1, f1 of t2, f1 of t3
#0.8	0.9	1.1	% f2 of t1, f2 of t2, f2 of t3
#0.34	0.34	0.02	% f3 of t1, f3 of t2, f3 of t3

#For the moment, we are going to use the feature of the word embeddings
#of dimension 300. So each token will have "300 features"

import csv
import numpy as np
import cPickle as pickle

from nltk.corpus import stopwords

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def one_hot(number,cluster_size):
    vector=['0']*cluster_size
    vector[number]='1'
    return vector

def ortographic_features(token,dico_ort, ort_dim, featVector):
    code=""
    #Capitalization
    if token.lower() == token:
        code=code+"1"
    elif token.upper() == token:
        code=code+"2"
    elif token[0].upper() == token[0]:
        code=code+"3"
    else:
        code=code+"4"

    #ends_s_feature
    if token[-1]=='s':
        code=code+"1"
    else:
        code=code+"0"

    #has digit
    if hasNumbers(token):
        code=code+"1"
    else:
        code=code+"0"

    #is_numeric_feature
    if token.isdigit():
        code=code+"1"
    else:
        code=code+"0"

    #is_alpha_feature
    if token.isalpha():
        code=code+"1"
    else:
        code=code+"0"

    #is_alphanum_feature
    if token.isalnum():
        code=code+"1"
    else:
        code=code+"0"

    #is_stopword_feature
    if token in stopwords.words('english'):
        code=code+"1"
    else:
        code=code+"0"

    if (code in dico_ort):
        valueOrt=dico_ort[code]
        featVector=featVector+valueOrt
    else:
        #Random inizialization
        drange = np.sqrt(6. / (np.sum((ort_dim,))))
        valueOrt = drange * np.random.uniform(low=-1.0, high=1.0, size=(ort_dim,))
        valueOrt=valueOrt.tolist()
        dico_ort[code]=valueOrt
        featVector=featVector+valueOrt

    return dico_ort,featVector




#First read the file
train=open("data/DrugBank/train_plus_features.txt","r")
dev=open("data/DrugBank/dev_plus_features.txt","r")
test=open("data/DrugBank/test_plus_features.txt","r")
#Open the pretrained embeddings file
pre_emb=open("embeddings/pre_emb_glove.txt")
#Read the POS tag mappings
#dico_POS_tags = pickle.load(open('Tag_Mappings/mappings_POS_tags.p', 'rb'))
#Read the clusters
#dico_clusters=pickle.load(open('clusters/300_glove_clusters.p','rb'))
#cluster_size=20
#Open a '.csv' file to save the data
csvfile=open("results/DrugBank/glove/dataTrain.csv","w")
csv_writer=csv.writer(csvfile,delimiter=',',quotechar='"',lineterminator='\n')
csvfileTest=open("results/DrugBank/glove/dataTest.csv","w")
csv_writerTest=csv.writer(csvfileTest,delimiter=',',quotechar='"',lineterminator='\n')
#Counter for the tokens in each line
token_counter=0

#Feature Vector dimension
wordEmbDim=300
POS_tag_dim=70
lemma_dim=4
metamap_dim=4
cluster_dim=40
orto_dim=28
#+POS_tag_dim+lemma_dim+metamap_dim+cluster_dim+orto_dim
featDim=wordEmbDim


#INITIALIZE DICTIONARIES
#1. EMBEDDING
#Create a dictionary with the pre_trained_embeddings
dico_pre_emb={}
for line in pre_emb:
   line=line.replace("\t"," ")
   line=line.replace("\n","")
   line=line.split(" ")
   #Check if it is in the dictionary
   if (line[0] not in dico_pre_emb):
       if (len(line)==wordEmbDim+1):
           dico_pre_emb[line[0]]=line[1:]

#2.The others are learned while running
dico_POS_tags={}
dico_lemma={}
dico_metamap={}
dico_clusters={}
dico_ort={}

#Start the loop
#Save the features of the sentence
sentence_feature_vector=[]
for word in train:
    #Clean data
    word=word.replace("\t"," ")
    word=word.replace("\n","")

    if (len(word.split(" "))==6):
        # Increment sentence length
        token_counter += 1
        #Save the word
        token=word.split(" ")[0]
        #Save the POS tag
        POS_tag=word.split(" ")[1]
        #Save the lemma
        lemma=word.split(" ")[2]
        #Save metamap
        metamap=word.split(" ")[3]
        #Save cluster
        cluster=word.split(" ")[4]
        #Extract the features to the token
        featVector=[]
        #1.Word Embedding Feature
        if (token in dico_pre_emb):
            valueEmbe=dico_pre_emb[token]
            featVector=featVector+valueEmbe
        else:
            #Random inizialization
            drange = np.sqrt(6. / (np.sum((wordEmbDim,))))
            valueEmbe = drange * np.random.uniform(low=-1.0, high=1.0, size=(wordEmbDim,))
            valueEmbe=valueEmbe.tolist()
            dico_pre_emb[token]=valueEmbe
            featVector=featVector+valueEmbe

        # #2. POS tags
        # if (POS_tag in dico_POS_tags):
        #     valuePOS=dico_POS_tags[POS_tag]
        #     featVector=featVector+valuePOS
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((POS_tag_dim,))))
        #     valuePOS = drange * np.random.uniform(low=-1.0, high=1.0, size=(POS_tag_dim,))
        #     valuePOS=valuePOS.tolist()
        #     dico_POS_tags[POS_tag]=valuePOS
        #     featVector=featVector+valuePOS
        #
        # #3. Lemma
        # if (lemma in dico_lemma):
        #     valueLemma=dico_lemma[lemma]
        #     featVector=featVector+valueLemma
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((lemma_dim,))))
        #     valueLemma = drange * np.random.uniform(low=-1.0, high=1.0, size=(lemma_dim,))
        #     valueLemma=valueLemma.tolist()
        #     dico_lemma[lemma]=valueLemma
        #     featVector=featVector+valueLemma
        #
        #
        # #4. Metamap
        # if (metamap in dico_metamap):
        #     valueMetamap=dico_metamap[metamap]
        #     featVector=featVector+valueMetamap
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((metamap_dim,))))
        #     valueMetamap = drange * np.random.uniform(low=-1.0, high=1.0, size=(metamap_dim,))
        #     valueMetamap=valueMetamap.tolist()
        #     dico_metamap[metamap]=valueMetamap
        #     featVector=featVector+valueMetamap
        #
        #
        # #5. Cluster
        # if (cluster in dico_clusters):
        #     valueCluster=dico_clusters[cluster]
        #     featVector=featVector+valueCluster
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((cluster_dim,))))
        #     valueCluster = drange * np.random.uniform(low=-1.0, high=1.0, size=(cluster_dim,))
        #     valueCluster=valueCluster.tolist()
        #     dico_clusters[cluster]=valueCluster
        #     featVector=featVector+valueCluster
        #
        #
        #
        # #6. Ortographic features
        # [dico_ort,featVector]=ortographic_features(token,dico_ort, orto_dim, featVector)

        sentence_feature_vector.append(featVector)

    else:
        #Write the sentence in the '.csv' file in the appropiate structure
        csv_writer.writerow([featDim, token_counter])
        #Create vectors of each feature for all the tokens
        for i in range(0,featDim):
            vector = []
            for j in range(0,token_counter):
                #Because we want the first feature for all the tokens
                vector.append(sentence_feature_vector[j][i])

            csv_writer.writerow(vector)


        token_counter=0
        sentence_feature_vector=[]

sentence_feature_vector=[]
for word in dev:
    #Clean data
    word=word.replace("\t"," ")
    word=word.replace("\n","")

    if (len(word.split(" "))==6):
        # Increment sentence length
        token_counter += 1
        #Save the word
        token=word.split(" ")[0]
        #Save the POS tag
        POS_tag=word.split(" ")[1]
        #Save the lemma
        lemma=word.split(" ")[2]
        #Save metamap
        metamap=word.split(" ")[3]
        #Save cluster
        cluster=word.split(" ")[4]
        #Extract the features to the token
        featVector=[]
        #1.Word Embedding Feature
        if (token in dico_pre_emb):
            valueEmbe=dico_pre_emb[token]
            featVector=featVector+valueEmbe
        else:
            #Random inizialization
            drange = np.sqrt(6. / (np.sum((wordEmbDim,))))
            valueEmbe = drange * np.random.uniform(low=-1.0, high=1.0, size=(wordEmbDim,))
            valueEmbe=valueEmbe.tolist()
            dico_pre_emb[token]=valueEmbe
            featVector=featVector+valueEmbe

        # #2. POS tags
        # if (POS_tag in dico_POS_tags):
        #     valuePOS=dico_POS_tags[POS_tag]
        #     featVector=featVector+valuePOS
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((POS_tag_dim,))))
        #     valuePOS = drange * np.random.uniform(low=-1.0, high=1.0, size=(POS_tag_dim,))
        #     valuePOS=valuePOS.tolist()
        #     dico_POS_tags[POS_tag]=valuePOS
        #     featVector=featVector+valuePOS
        #
        # #3. Lemma
        # if (lemma in dico_lemma):
        #     valueLemma=dico_lemma[lemma]
        #     featVector=featVector+valueLemma
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((lemma_dim,))))
        #     valueLemma = drange * np.random.uniform(low=-1.0, high=1.0, size=(lemma_dim,))
        #     valueLemma=valueLemma.tolist()
        #     dico_lemma[lemma]=valueLemma
        #     featVector=featVector+valueLemma
        #
        #
        # #4. Metamap
        # if (metamap in dico_metamap):
        #     valueMetamap=dico_metamap[metamap]
        #     featVector=featVector+valueMetamap
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((metamap_dim,))))
        #     valueMetamap = drange * np.random.uniform(low=-1.0, high=1.0, size=(metamap_dim,))
        #     valueMetamap=valueMetamap.tolist()
        #     dico_metamap[metamap]=valueMetamap
        #     featVector=featVector+valueMetamap
        #
        #
        # #5. Cluster
        # if (cluster in dico_clusters):
        #     valueCluster=dico_clusters[cluster]
        #     featVector=featVector+valueCluster
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((cluster_dim,))))
        #     valueCluster = drange * np.random.uniform(low=-1.0, high=1.0, size=(cluster_dim,))
        #     valueCluster=valueCluster.tolist()
        #     dico_clusters[cluster]=valueCluster
        #     featVector=featVector+valueCluster
        #
        #
        #
        # #6. Ortographic features
        # [dico_ort,featVector]=ortographic_features(token,dico_ort, orto_dim, featVector)

        sentence_feature_vector.append(featVector)

    else:
        #Write the sentence in the '.csv' file in the appropiate structure
        csv_writer.writerow([featDim, token_counter])
        #Create vectors of each feature for all the tokens
        for i in range(0,featDim):
            vector = []
            for j in range(0,token_counter):
                #Because we want the first feature for all the tokens
                vector.append(sentence_feature_vector[j][i])

            csv_writer.writerow(vector)


        token_counter=0
        sentence_feature_vector=[]

sentence_feature_vector=[]
for word in test:
    #Clean data
    word=word.replace("\t"," ")
    word=word.replace("\n","")

    if (len(word.split(" "))==6):
        # Increment sentence length
        token_counter += 1
        #Save the word
        token=word.split(" ")[0]
        #Save the POS tag
        POS_tag=word.split(" ")[1]
        #Save the lemma
        lemma=word.split(" ")[2]
        #Save metamap
        metamap=word.split(" ")[3]
        #Save cluster
        cluster=word.split(" ")[4]
        #Extract the features to the token
        featVector=[]
        #1.Word Embedding Feature
        if (token in dico_pre_emb):
            valueEmbe=dico_pre_emb[token]
            featVector=featVector+valueEmbe
        else:
            #Random inizialization
            drange = np.sqrt(6. / (np.sum((wordEmbDim,))))
            valueEmbe = drange * np.random.uniform(low=-1.0, high=1.0, size=(wordEmbDim,))
            valueEmbe=valueEmbe.tolist()
            dico_pre_emb[token]=valueEmbe
            featVector=featVector+valueEmbe

        # #2. POS tags
        # if (POS_tag in dico_POS_tags):
        #     valuePOS=dico_POS_tags[POS_tag]
        #     featVector=featVector+valuePOS
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((POS_tag_dim,))))
        #     valuePOS = drange * np.random.uniform(low=-1.0, high=1.0, size=(POS_tag_dim,))
        #     valuePOS=valuePOS.tolist()
        #     dico_POS_tags[POS_tag]=valuePOS
        #     featVector=featVector+valuePOS
        #
        # #3. Lemma
        # if (lemma in dico_lemma):
        #     valueLemma=dico_lemma[lemma]
        #     featVector=featVector+valueLemma
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((lemma_dim,))))
        #     valueLemma = drange * np.random.uniform(low=-1.0, high=1.0, size=(lemma_dim,))
        #     valueLemma=valueLemma.tolist()
        #     dico_lemma[lemma]=valueLemma
        #     featVector=featVector+valueLemma
        #
        #
        # #4. Metamap
        # if (metamap in dico_metamap):
        #     valueMetamap=dico_metamap[metamap]
        #     featVector=featVector+valueMetamap
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((metamap_dim,))))
        #     valueMetamap = drange * np.random.uniform(low=-1.0, high=1.0, size=(metamap_dim,))
        #     valueMetamap=valueMetamap.tolist()
        #     dico_metamap[metamap]=valueMetamap
        #     featVector=featVector+valueMetamap
        #
        #
        # #5. Cluster
        # if (cluster in dico_clusters):
        #     valueCluster=dico_clusters[cluster]
        #     featVector=featVector+valueCluster
        # else:
        #     #Random inizialization
        #     drange = np.sqrt(6. / (np.sum((cluster_dim,))))
        #     valueCluster = drange * np.random.uniform(low=-1.0, high=1.0, size=(cluster_dim,))
        #     valueCluster=valueCluster.tolist()
        #     dico_clusters[cluster]=valueCluster
        #     featVector=featVector+valueCluster
        #
        #
        #
        # #6. Ortographic features
        # [dico_ort,featVector]=ortographic_features(token,dico_ort, orto_dim, featVector)

        sentence_feature_vector.append(featVector)

    else:
        #Write the sentence in the '.csv' file in the appropiate structure
        csv_writerTest.writerow([featDim, token_counter])
        #Create vectors of each feature for all the tokens
        for i in range(0,featDim):
            vector = []
            for j in range(0,token_counter):
                #Because we want the first feature for all the tokens
                vector.append(sentence_feature_vector[j][i])

            csv_writerTest.writerow(vector)


        token_counter=0
        sentence_feature_vector=[]
