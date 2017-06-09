#DrugBank
DrugBank_train=open('data_DER/DrugBank/NER/train.txt','r')
DrugBank_dev=open('data_DER/DrugBank/NER/dev.txt','r')
DrugBank_test=open('data_DER/DrugBank/NER/test.txt','r')
#MedLine
MedLine_train=open('data_DER/MedLine/NER/train.txt','r')
MedLine_dev=open('data_DER/MedLine/NER/dev.txt','r')
MedLine_test=open('data_DER/MedLine/NER/test.txt','r')
#CCE
CCE_train=open('data_fixedLengthOne/conll2003/train.txt','r')
CCE_dev=open('data_fixedLengthOne/conll2003/dev.txt','r')
CCE_test=open('data_fixedLengthOne/conll2003/test.txt','r')


#Count DrugBank
dictionary_DrugBank={}
for line in DrugBank_train:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_DrugBank[line[0]]=0
for line in DrugBank_dev:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_DrugBank[line[0]]=0
for line in DrugBank_test:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_DrugBank[line[0]]=0
DrugBank_train.close()
DrugBank_dev.close()
DrugBank_test.close()

#Count MedLine
dictionary_MedLine={}
for line in MedLine_train:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_MedLine[line[0]]=0
for line in MedLine_dev:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_MedLine[line[0]]=0
for line in MedLine_test:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_MedLine[line[0]]=0

#Count CCE
dictionary_CCE={}
for line in CCE_train:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_CCE[line[0]]=0
for line in CCE_dev:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_CCE[line[0]]=0
for line in CCE_test:
    line=line.replace("\n","")
    line=line.split("\t")
    if (len(line)>=2):
        dictionary_CCE[line[0]]=0

#Print the results
print "DrugBank vocabulary: "+str(len(dictionary_DrugBank))
print "MedLine vocabulary: "+str(len(dictionary_MedLine))
print "CCE vocabulary: "+str(len(dictionary_CCE))