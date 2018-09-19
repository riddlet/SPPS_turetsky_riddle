from __future__ import division
import numpy as np
import pandas as pd
import operator
import csv
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
import os
import glob
import re
import spacy
import nltk
from nltk import word_tokenize, sent_tokenize
from scipy import spatial
import textacy

def get_dataframe(path, sheet='Text'):
    docs=[]
    completed_s=pd.read_csv(os.path.join(path,'completed_sources.csv'))
    #print(completed_s)
    fileformat = "*/*-"+sheet+".csv" #Gets only text of images csvs, defaults to Text
    #print(fileformat)
    docs=glob.glob(os.path.join(path,fileformat)) #Gets a list of all those Text csvs
    #print(docs)

    df = pd.DataFrame() #initializes df as dataframe
    for doc in docs:
        if os.path.basename(os.path.dirname(doc)) in completed_s['Source'].tolist(): #name of the source extracted from folder
            #print(doc)
            #print(os.path.basename(os.path.dirname(doc)))
            try: 
                dat = pd.read_csv(doc) # gets data from doc using PANDAS, it's a dataframe!
                df = df.append(dat, ignore_index=True) #appends data from this doc to dataframe
                #print(df)
            except Exception as e: #LOL
                print (e)

    return(df)

def remove_links(doc):
    out = re.sub(r'{(.+?)}<hyperlink=.+?>', r'\1', doc)
    return(out)

def remove_tweets(doc):
    out = re.sub(r'\[tweetbegin.+?tweetend\]', '', doc)
    return(out)

def compute_pmi(w1, w2, sparse_matrix, vectorizer):
    ocurrence = sparse_matrix[:, [vectorizer.vocabulary_.get(w1), vectorizer.vocabulary_.get(w2)]].toarray()
    vect_len = ocurrence.shape[0]
    count_both = np.where((ocurrence==(1,1)).all(axis=1))[0].shape[0]
    count_r = count_both + np.where((ocurrence==(1,0)).all(axis=1))[0].shape[0]
    count_w = count_both + np.where((ocurrence==(0,1)).all(axis=1))[0].shape[0]
    try:
        pmi = np.log((count_both/vect_len)/((count_r/vect_len)*(count_w/vect_len)))
        if np.isinf(pmi):
            return(0)
        else:
            return(pmi)
    except:
        #print(w1, w2)
        return(0)

def write_dict(seeds, scoring, filename, type='sim'):
    sorted_words = sorted(scoring.items(), key=operator.itemgetter(1), reverse=True)
    if type=='sim':
        words = [x[0] for x in sorted_words[0:100]]
        words.extend([i.decode('utf8') for i in seeds])
    else:
        words = [x[0] for x in sorted_words[-100:]]
        words.extend([i.decode('utf8') for i in seeds])
    with open('dicts/'+filename+'.csv', 'wb') as resultfile:
        wr = csv.writer(resultfile, dialect='excel')
        for item in words:
            wr.writerows([[item]])

def read_dict_fromtxt(pathtofile):
	'''
	reads a dictionary (eg. liwc)
	'''
	f = open(pathtofile)
	f = f.read()
	f = nltk.word_tokenize(f)
	return(f)

def liwc_match(dictionary, doc):
    '''
    looks at a document (typically a word) and returns whether it found a match
    for that word in the dictionary
    '''
    result = 0
    prog=re.compile(unicode(dictionary), re.U)
    if prog.match(doc):
        result = 1
    return(result)

def liwc_it(art, liwc_cat, spacy_mod, bigrams=False):
    '''
    computes the number of matches between a liwc dictionary and a specific article
    '''
    hits = [0]
    for w in spacy_mod.tokenizer(art.decode('utf8')):
            hits.append(liwc_match(liwc_cat, w.lower_))
    if bigrams:
        doc = spacy_mod(art.decode('utf8'))
        extracted = textacy.extract.ngrams(doc, n=2)
        for i in extracted:
            hits.append(liwc_match(liwc_cat, ' '.join([j.lower_ for j in i])))
    return(sum(hits))

def tokenize(str):
  '''
  custom tokenizer. Tokenizes into sentences, then strips punctuation/abbr, converts to lowercase and tokenizes words
  '''
  return  [word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
      for t in sent_tokenize(str.replace("'", ""))]

def w2vsim(text, vector, spacy_mod, w2v_mod):
    '''
    Goes through all the words in a document, obtains the cosine similarity between 
    each word and the target vector, and returns the average similarity across all
    words
    '''
    sims = []
    for w in spacy_mod.tokenizer(text.decode('utf8')):
        if w.orth_ in w2v_mod.vocab:
            sims.append(1-spatial.distance.cosine(vector, w2v_mod.word_vec(w.orth_)))
    return(np.mean(sims))

def concrete(text, conc_dict, spacy_mod, w2v_mod):
    '''
    goes through all the words in a document, obtains the concreteness value from
    the loaded concreteness dictionary and returns the average concreteness across
    the document.
    '''
    sims = []
    for w in spacy_mod.tokenizer(text.decode('utf8')):
        if w.lower_ in conc_dict.keys():
            sims.append(conc_dict[w.lower_][0])
    return(np.mean(sims))

def gen_liwc_dicts(dictionary):
    '''
    reformats liwc dictionary for regex matching
    '''
    for category in dictionary.keys():
        regex_cat = []
        for n, item in enumerate(dictionary[category]):
            item = re.sub(ur'(.+)\*', r'\1.*?', item)
            item = re.sub(ur'_', r' ', item)
            item = '^'+str(item)+'$'
            regex_cat.append(item)
        dictionary[category]='|'.join(regex_cat)
    return(dictionary)


def main():
    print("Loading data and models")
    d = get_dataframe('../FergusonMedia/DATA/Sources/')
    d.ArticleText.fillna('', inplace=True)
    d['clean_text'] = d.ArticleText.apply(lambda x: remove_links(x))
    d['clean_text'] = d.clean_text.apply(lambda x: remove_tweets(x))
    model = KeyedVectors.load_word2vec_format('../../w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)
    nlp = spacy.load('en')

    vectorizer = CountVectorizer()
    cv = vectorizer.fit_transform(d.clean_text)

    r = ['black', 'white', 'race', 'ethnicity', 'diversity']
    print("Computing PMI")

    # pmi = dict.fromkeys(vectorizer.get_feature_names(), 0)
    # for w in vectorizer.get_feature_names():
    #     for s in r:
    #         pmi[w] = pmi[w] + compute_pmi(s, w, cv, vectorizer)

    # write_dict(r, pmi, 'race')

    dict_cats = {}
    dict_cats['race'] = read_dict_fromtxt('dicts/race.csv')
    dict_cats['youth'] = ['graduation', 'high school', 'teenager', 'young', 'youth', 'child']
    dict_cats['cigarillos'] = ['cigarillos', 'cigar', 'cigarette']
    dict_cats['dead_blacks'] = ['eric garner', 'travyon martin', 'ezell ford']
    dict_cats['michael_brown'] = ['michael brown', 'mike brown']
    dict_cats['protest'] = ['protest', 'protesting', 'protests', 'protesters', 'protester', 'protestor', 'protested']
    dict_cats['riot'] = ['riot', 'rioting', 'riots', 'rioters', 'rioter', 'rioted']
    dict_cats = gen_liwc_dicts(dict_cats)
    d['race'] = d.clean_text.apply(liwc_it, args=[dict_cats['race'], nlp])
    d['cigarillos'] = d.clean_text.apply(liwc_it, args=[dict_cats['cigarillos'], nlp])
    d['youth'] = d.clean_text.apply(liwc_it, args=[dict_cats['youth'], nlp])
    d['dead_blacks'] = d.clean_text.apply(lambda x: liwc_it(x, dict_cats['dead_blacks'], nlp, bigrams=True))
    d['michael_brown'] = d.clean_text.apply(lambda x: liwc_it(x, dict_cats['michael_brown'], nlp, bigrams=True))
    d['protest'] = d.clean_text.apply(liwc_it, args=[dict_cats['protest'], nlp])
    d['riot'] = d.clean_text.apply(liwc_it, args=[dict_cats['riot'], nlp])

    print("Computing w2v for race")
    w2v_sims = dict.fromkeys(vectorizer.get_feature_names(), 0)
    vect = np.mean([model.word_vec('black'), model.word_vec('white'), model.word_vec('race'), model.word_vec('ethnicity'), model.word_vec('diversity')], axis=0)
    for w in vectorizer.get_feature_names():
        if w in model.vocab:
            w2v_sims[w] = 1-spatial.distance.cosine(vect, model.word_vec(w))
        else:
            w2v_sims[w] = 0

    write_dict(r, w2v_sims, 'race_w2v')

    d['race_w2v'] = d.clean_text.apply(w2vsim, args=[vect, nlp, model])

    print("Computing w2v for race (no_diversity)")
    w2v_sims = dict.fromkeys(vectorizer.get_feature_names(), 0)
    vect = np.mean([model.word_vec('black'), model.word_vec('white'), model.word_vec('race'), model.word_vec('ethnicity')], axis=0)
    for w in vectorizer.get_feature_names():
        if w in model.vocab:
            w2v_sims[w] = 1-spatial.distance.cosine(vect, model.word_vec(w))
        else:
            w2v_sims[w] = 0

    r = ['black', 'white', 'race', 'ethnicity']
    write_dict(r, w2v_sims, 'race_w2v_no_div')

    d['race_nodiv_w2v'] = d.clean_text.apply(w2vsim, args=[vect, nlp, model])

    print("Computing w2v for cigarillos")
    w2v_sims = dict.fromkeys(vectorizer.get_feature_names(), 0)
    vect = np.mean([model.word_vec('cigar'), model.word_vec('cigarillo'), model.word_vec('cigarette')], axis=0)
    for w in vectorizer.get_feature_names():
        if w in model.vocab:
            w2v_sims[w] = 1-spatial.distance.cosine(vect, model.word_vec(w))
        else:
            w2v_sims[w] = 0

    r = ['cigar', 'cigarillo', 'cigarette']
    write_dict(r, w2v_sims, 'cigarillos_w2v')

    d['cigarillos_w2v'] = d.clean_text.apply(w2vsim, args=[vect, nlp, model])

    print("Computing w2v for youth")
    w2v_sims = dict.fromkeys(vectorizer.get_feature_names(), 0)
    vect = np.mean([model.word_vec('graduation'), model.word_vec('school'), model.word_vec('teenager'), model.word_vec('young'), model.word_vec('youth'), model.word_vec('child')], axis=0)
    for w in vectorizer.get_feature_names():
        if w in model.vocab:
            w2v_sims[w] = 1-spatial.distance.cosine(vect, model.word_vec(w))
        else:
            w2v_sims[w] = 0

    r = ['graduation', 'school', 'teenager', 'young', 'youth', 'child']
    write_dict(r, w2v_sims, 'youth')

    d['youth_w2v'] = d.clean_text.apply(w2vsim, args=[vect, nlp, model])

    print("Computing w2v for egalitarianism")

    w2v_sims = dict.fromkeys(vectorizer.get_feature_names(), 0)
    vect = np.mean([model.word_vec('egalitarianism'), model.word_vec('equal'), model.word_vec('fair')], axis=0)
    for w in vectorizer.get_feature_names():
        if w in model.vocab:
            w2v_sims[w] = 1-spatial.distance.cosine(vect, model.word_vec(w))
        else:
            w2v_sims[w] = 0

    r = ['egalitarianism', 'equal', 'fair']
    write_dict(r, w2v_sims, 'egal_w2v')

    d['egal_w2v'] = d.clean_text.apply(w2vsim, args=[vect, nlp, model])

    print("Computing w2v for egalitarianism (after review)")

    w2v_sims = dict.fromkeys(vectorizer.get_feature_names(), 0)
    vect = np.mean([model.word_vec('egalitarian'), model.word_vec('equal'), model.word_vec('fair'), model.word_vec('injustice'), model.word_vec('equity')], axis=0)
    for w in vectorizer.get_feature_names():
        if w in model.vocab:
            w2v_sims[w] = 1-spatial.distance.cosine(vect, model.word_vec(w))
        else:
            w2v_sims[w] = 0

    r = ['egalitarianism', 'equal', 'fair', 'injustice', 'equity']
    write_dict(r, w2v_sims, 'egal_postreview_w2v')

    d['egal_postreview_w2v'] = d.clean_text.apply(w2vsim, args=[vect, nlp, model])

    print("Computing w2v for individualism")

    w2v_sims = dict.fromkeys(vectorizer.get_feature_names(), 0)
    vect = np.mean([model.word_vec('individualism'), model.word_vec('earn'), model.word_vec('deserve')], axis=0)
    for w in vectorizer.get_feature_names():
        if w in model.vocab:
            w2v_sims[w] = 1-spatial.distance.cosine(vect, model.word_vec(w))
        else:
            w2v_sims[w] = 0

    r = ['individualism', 'earn', 'deserve']
    write_dict(r, w2v_sims, 'ind_w2v')

    d['ind_w2v'] = d.clean_text.apply(w2vsim, args=[vect, nlp, model])


    print("Computing w2v for individualism (after review)")

    w2v_sims = dict.fromkeys(vectorizer.get_feature_names(), 0)
    vect = np.mean([model.word_vec('earn'), model.word_vec('deserve'), model.word_vec('merit'), model.word_vec('warranted'), model.word_vec('entitled')], axis=0)
    for w in vectorizer.get_feature_names():
        if w in model.vocab:
            w2v_sims[w] = 1-spatial.distance.cosine(vect, model.word_vec(w))
        else:
            w2v_sims[w] = 0

    r = ['earn', 'deserve', 'merit', 'warranted', 'entitled']
    write_dict(r, w2v_sims, 'ind_postreview_w2v')

    d['ind_postreview_w2v'] = d.clean_text.apply(w2vsim, args=[vect, nlp, model])

    print("Computing concreteness")

    conc = pd.read_csv('dicts/concreteness.csv')
    conc_dict = conc[['Word', 'Conc.M']].set_index('Word').T.to_dict('list')
    d['concrete'] = d.clean_text.apply(concrete, args=[conc_dict, nlp, model])
    #temp = pd.read_csv('../output/dict_methods.csv')
    #d['concrete'] = temp['concrete']

    d[['ArticleID', 'Source', 'race', 'race_w2v', 'cigarillos', 'cigarillos_w2v', 
    'youth', 'youth_w2v', 'egal_w2v', 'ind_w2v', 'concrete', 'race_nodiv_w2v',
    'egal_postreview_w2v', 'ind_postreview_w2v', 'dead_blacks']].to_csv('../output/dict_methods.csv', 
        encoding='utf-8', index=False)

if __name__ == '__main__':
    main()

