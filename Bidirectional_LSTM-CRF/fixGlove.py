#Read the corrupted file
embeddingCorrupted=open('pre_trainedVectorEmbeddings/pre_emb_glove.txt','r')


#Open the new file
embeddingFixed=open('pre_trainedVectorEmbeddings/pre_emb_glove_fixed.txt','w')



for line in embeddingCorrupted:
    line_mod=line.replace("\n","")
    line_mod=line_mod.split("\t")
    word=line_mod[0]
    vector=line_mod[1].split(" ")
    new_line=word+"\t"
    if (len(vector)==300):
        embeddingFixed.write(line)
    else:
        print (len(vector))
        flag=0
        for element in vector:
            if element=='' and flag==0:
                new_line=new_line+" "
                flag=1
            elif element=='' and flag==1:
                flag=0
            else:
                new_line=new_line+str(element)
        embeddingFixed.write(new_line+"\n")

embeddingFixed.close()


embeddingCorrupted=open('pre_trainedVectorEmbeddings/pre_emb_glove_fixed.txt','r')

for line in embeddingCorrupted:
    line_mod=line.replace("\n","")
    line_mod=line_mod.split("\t")
    word=line_mod[0]
    vector=line_mod[1].split(" ")
    print (vector)
