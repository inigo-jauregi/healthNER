import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes
import collections

from nltk.corpus import stopwords


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char

def POStag_mapping(sentences):
    """
    Create a dictionary and mapping of POS tags, sorted by frequency.
    """
    POStags=[[word[1] for word in s] for s in sentences]
    dico = create_dico(POStags)
    POStag_to_id, id_to_POStag = create_mapping(dico)
    print "Found %i POS tags" % len(dico)
    return dico, POStag_to_id, id_to_POStag

def cluster_mapping(sentences):
    """
    Create a dictionary and mapping of clusters, sorted by frequency.
    """
    clusters=[[word[4] for word in s] for s in sentences]
    dico = create_dico(clusters)
    cluster_to_id, id_to_cluster = create_mapping(dico)
    print "Found %i clusters" % len(dico)
    return dico, cluster_to_id, id_to_cluster

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def ends_s_feature(s):
    #Ends with 's' feature:
    #0 = last letter of the word is a s
    #1 = last letter of the word is not a s
    if s[-1]=='s':
        return 0
    else:
        return 1

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def has_digit_feature(s):
    #The string has a digit
    #0 = the string has a digit
    #1 = the string has not a digit
    if hasNumbers(s):
        return 0
    else:
        return 1

def is_numeric_feature(s):
    #See if it is a number
    #0 = The string is a number
    #1 = The string has non-numeric characters
    if s.isdigit():
        return 0
    else:
        return 1

def is_alpha_feature(s):
    #See if it is only alphabetic
    #0 = The string is aphabetic
    #1 = The string has non-alphabetic characters
    if s.isalpha():
        return 0
    else:
        return 1

def is_alphanum_feature(s):
    #See if it is alphanumeric
    #0 = The string is aphanumeric
    #1 = The string has non-alphanumeric characters
    if s.isalnum():
        return 0
    else:
        return 1

def is_stopword_feature(s):
    #See if it is stopword
    #0 = It is a stopword
    #1 = It is not a stopword
    if s in stopwords.words('english'):
        return 0
    else:
        return 1

def is_lemma(s):
    #See if the lemma is the same
    #0 = Lemma is the same as the original word
    #1 = Lemma is different from the original word
    if s=="1":
        return 1
    else:
        return 0

def is_metamap(s):
    #See if there is a MetaMap concept
    #0 = Concept "UNK"
    #1 = There is a concept
    if s=="1":
        return 1
    else:
        return 0


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    ends_s= [ends_s_feature(w) for w in str_words]
    digit_w= [has_digit_feature(w) for w in str_words]
    numeric= [is_numeric_feature(w) for w in str_words]
    alpha= [is_alpha_feature(w) for w in str_words]
    alphanum= [is_alphanum_feature(w) for w in str_words]
    stop_w= [is_stopword_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'ends_s': ends_s,
        'digit_w': digit_w,
        'numeric': numeric,
        'alpha': alpha,
        'alphanum': alphanum,
        'stop_w': stop_w
    }


def prepare_dataset(sentences, word_to_id, char_to_id, POStag_to_id, cluster_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        #str_word: Is the actual string of characters
        str_words = [w[0] for w in s]
        #words:  The list of IDs of the words
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        #chars: The list o IDs of the chracters
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        #caps: The list of capitalization features
        caps = [cap_feature(w) for w in str_words]
        ends_s= [ends_s_feature(w) for w in str_words]
        digit_w= [has_digit_feature(w) for w in str_words]
        numeric= [is_numeric_feature(w) for w in str_words]
        alpha= [is_alpha_feature(w) for w in str_words]
        alphanum= [is_alphanum_feature(w) for w in str_words]
        stop_w= [is_stopword_feature(w) for w in str_words]
        # print "=================="
        # for k,v in tag_to_id.iteritems():
        #     print k,v

        # print "==================\n"
        # tag_to_id.update({'I-Drug_n':16}) # added manually
        #List of IDs of the tags of each word
        #Get Lemma
        lemma = [is_lemma(w[2]) for w in s]
        metamap = [is_metamap(w[3]) for w in s]
        POS_tags = [POStag_to_id[w[1] if w[1] in POStag_to_id else 'UNK'] for w in s ]
        cluster = [cluster_to_id[w[4] if w[1] in cluster_to_id else 'UNK'] for w in s ]
        tags = [tag_to_id[w[-1]] for w in s]

        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'ends_s': ends_s,
            'digit_w': digit_w,
            'numeric': numeric,
            'alpha': alpha,
            'alphanum': alphanum,
            'stop_w': stop_w,
            'lemma': lemma,
            'metamap': metamap,
            'postags': POS_tags,
            'cluster': cluster,
            'tags': tags,
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    #We save in 'pretrained' all the words from the pretrained embeddings (Only the words!)
    #For Example: Glove have 28394 words
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
