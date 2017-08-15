import numpy as np

embeddingGeneral=open('pre_trainedVectorEmbeddings/pre_emb_glove_fixed.txt','r')
embeddingSpecialized=open('pre_trainedVectorEmbeddings/pre_emb_mimic_300.txt','r')

dico_general={}
dico_specialized={}

dim_gen_embe=300
dim_spe_embe=300


#Learn the vocabularies
for line in embeddingGeneral:
    line=line.replace("\n","")
    line=line.split("\t")
    dico_general[line[0]]=line[1]

for line in embeddingSpecialized:
    line=line.replace("\n","")
    line=line.split(" ")
    dico_specialized[line[0]]=line[1:]

print len(dico_general)
print len(dico_specialized)

#Counters
Counter1=0
Counter2=0
Counter3=0


file_write=open("pre_trainedVectorEmbeddings/pre_emb_glove_fixed_mimic_600.txt",'w')
#Fill the new document with the combination of embeddings
for t, e in dico_general.items():
    file_write.write(t+"\t")
    Counter1+=1
    gen_embe=e
    if t in dico_specialized:
        Counter2+=1
        spe_embe=dico_specialized[t]
        spe_embe=(" ").join(spe_embe)
        total_embe=gen_embe+" "+spe_embe
    else:
        drange = np.sqrt(6. / (np.sum((dim_spe_embe,))))
        spe_embe = drange * np.random.uniform(low=-1.0, high=1.0, size=(dim_spe_embe,))
        spe_embe=spe_embe.tolist()
        spe_embe=(" ").join(map(str,spe_embe))
        total_embe=gen_embe+" "+spe_embe

    file_write.write(total_embe+"\n")


#And now iterate the other dico
for t, e in dico_specialized.items():
    if t not in dico_general:
        file_write.write(t+"\t")
        Counter3+=1
        drange = np.sqrt(6. / (np.sum((dim_gen_embe,))))
        gen_embe = drange * np.random.uniform(low=-1.0, high=1.0, size=(dim_gen_embe,))
        gen_embe=gen_embe.tolist()
        gen_embe=(" ").join(map(str,gen_embe))

        spe_embe=e
        spe_embe=(" ").join(spe_embe)

        total_embe=gen_embe+" "+spe_embe

        file_write.write(total_embe+"\n")

file_write.close()

print "Loaded "+str(Counter1)+" words from dico_general"
print "Loaded "+str(Counter2)+" words from dico_spe and dico_general"
print "Loaded "+str(Counter3)+" words from dico_spe but not dico_general"
