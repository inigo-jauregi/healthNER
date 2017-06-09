import numpy as np

embeddingGeneral=open('pre_trainedVectorEmbeddings/pre_emb_glove.txt','r')
embeddingSpecialized=open('pre_trainedVectorEmbeddings/pre_emb_glove_i2b2_300.txt','r')

dico_general={}
dico_specialized={}

dim_gen_embe=300
dim_spe_embe=300

#Learn the vocabularies
for line in embeddingGeneral:
    line=line.replace("\n","")
    line=line.replace("\t"," ")
    line=line.split(" ")
    dico_general[line[0]]=line[1:]

for line in embeddingSpecialized:
    line=line.replace("\n","")
    line=line.split(" ")
    dico_specialized[line[0]]=line[1:]

file_write=open("pre_trainedVectorEmbeddings/pre_emb_glove_average_gen_i2b2_300.txt",'w')
#Fill the new document with the combination of embeddings
for t, e in dico_general.items():
    file_write.write(t+"\t")
    gen_embe=e
    if (len(gen_embe)==dim_gen_embe):
        gen_embe=np.asarray(gen_embe)
        gen_embe=gen_embe.astype(np.float)
        if t in dico_specialized:
            spe_embe=dico_specialized[t]
            spe_embe=np.asarray(spe_embe)
            spe_embe=spe_embe.astype(np.float)
            #Obtain average
            if (len(gen_embe)==len(spe_embe)):
                total_embe=np.mean([gen_embe, spe_embe], axis=0)
            else:
                total_embe=gen_embe
            total_embe=total_embe.tolist()
            total_embe=" ".join(map(str,total_embe))
        else:
            drange = np.sqrt(6. / (np.sum((dim_spe_embe,))))
            spe_embe = drange * np.random.uniform(low=-1.0, high=1.0, size=(dim_spe_embe,))
            spe_embe=spe_embe.astype(np.float)
            #Obtain average
            if (len(gen_embe)==len(spe_embe)):
                total_embe=np.mean([gen_embe, spe_embe], axis=0)
            else:
                total_embe=gen_embe
            total_embe=total_embe.tolist()
            total_embe=" ".join(map(str,total_embe))

        file_write.write(total_embe+"\n")
    else:
        total_embe=" ".join(gen_embe)
        file_write.write(total_embe+"\n")

#And now iterate the other dico
for t, e in dico_specialized.items():
    if t not in dico_general:
        file_write.write(t+"\t")

        drange = np.sqrt(6. / (np.sum((dim_gen_embe,))))
        gen_embe = drange * np.random.uniform(low=-1.0, high=1.0, size=(dim_gen_embe,))
        gen_embe=gen_embe.astype(np.float)

        spe_embe=e
        spe_embe=np.asarray(spe_embe).astype(np.float)
        #Obtain average
        #Obtain average
        if (len(gen_embe)==len(spe_embe)):
            total_embe=np.mean([gen_embe, spe_embe], axis=0)
        else:
            total_embe=gen_embe
        total_embe=total_embe.tolist()
        total_embe=" ".join(map(str,total_embe))

        file_write.write(total_embe+"\n")

file_write.close()