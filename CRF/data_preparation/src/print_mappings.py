import cPickle as pickle

dico_DrugBank = pickle.load(open('data/DrugBank/mappings_tags.p', 'rb'))
dico_MedLine = pickle.load(open('data/MedLine/mappings_tags.p', 'rb'))

print "DrugBank: "
for ele in dico_DrugBank:
    print ele
print "MedLine: "
for ele in dico_MedLine:
    print ele