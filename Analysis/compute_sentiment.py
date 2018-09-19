# This Python file uses the following encoding: utf-8
import os
import pandas as pd
import nltk
import csv
import glob
from nltk import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import numpy as np
import spacy

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
    	print doc
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

def read_dict_fromtxt(pathtofile):
	'''
	reads a dictionary (eg. liwc)
	'''
	f = open(pathtofile)
	f = f.read()
	f = nltk.word_tokenize(f)
	return(f)

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

def liwc_it(art, liwc_cat, spacy_mod):
    '''
    computes the number of matches between a liwc dictionary and a specific article
    '''
    hits = [0]
    for w in spacy_mod.tokenizer(art.decode('utf8')):
            hits.append(liwc_match(liwc_cat, w.lower_))
    return(sum(hits))

def tokenize(str):
  '''
  custom tokenizer. Tokenizes into sentences, then strips punctuation/abbr, converts to lowercase and tokenizes words
  '''
  return  [word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
      for t in sent_tokenize(str.replace("'", ""))]

def vader_compute(art, spacy_mod):
	'''
	vader is developed for short text (i.e. sentence level), so this function 
	computes sentiment for each sentence in an article and returns the mean
	across sentences
	'''
	analyzer = SentimentIntensityAnalyzer()
	comp = []
	neg = []
	neu = []
	pos = []
	art = spacy_mod(art.decode('utf8'))
	sentences = [sent.string.strip() for sent in art.sents]
	for s in sentences:
		vader_sent = analyzer.polarity_scores(s)
		comp.append(vader_sent['compound'])
		neg.append(vader_sent['neg'])
		neu.append(vader_sent['neu'])
		pos.append(vader_sent['pos'])
        
	d = {'compound' : np.mean(comp), 
	'neg' : np.mean(neg), 
	'neu' : np.mean(neu), 
	'pos' : np.mean(pos)}

	return(d)

def remove_links(doc):
	''' Removes the links annotated by RAs'''
	out = re.sub(r'{(.+)}<hyperlink=.+>', r'\1', doc)
	return(out)

def remove_tweets(doc):
	''' Removes tweets annotated by RAs'''
	out = re.sub(r'\[tweetbegin.+?tweetend\]', '', doc)
	return(out)


def main():
	'''
	main prog - gathers data from the data directory, converts it to a
	pandas dataframe, computes the sentiment metrics defined in preanalysis
	plan, then writes a new datafile to the output folder
	'''
	print('Loading Data')
	d = get_dataframe('../FergusonMedia/DATA/Sources/')
	d.to_csv('../output/combined_dat.csv', encoding='utf8', index=False)
	nlp = spacy.load('en')

	liwc_cats = {}
	liwc_cats['pos'] = read_dict_fromtxt('dicts/Posemo.csv')
	liwc_cats['neg'] = read_dict_fromtxt('dicts/Negemo.csv')
	liwc_cats = gen_liwc_dicts(liwc_cats)

	d = d.replace('', np.nan)
	d = d[pd.notnull(d.ArticleText)]
	d['cleanText'] = d.ArticleText.apply(lambda x: remove_links(x))
	d['cleanText'] = d.cleanText.apply(lambda x: remove_tweets(x))
	print('Computing Liwc Metrics')
	d['pos'] = d.cleanText.apply(liwc_it, args=[liwc_cats['pos'], nlp])
	d['neg'] = d.cleanText.apply(liwc_it, args=[liwc_cats['neg'], nlp])   
	d['words'] = d.cleanText.apply(lambda x: len(nlp.tokenizer(x.decode('utf8')))) 
	d['positivity'] = d.pos/(d.pos+d.neg)
	d['posneg_ratio'] = d.pos/d.neg
	d['posneg_diff'] = d.pos - d.neg
	d['subjectivity'] = (d.pos+d.neg)/d.words
	print('Computing Vader metrics')
	vs = d.cleanText.apply(vader_compute, args=[nlp])
	d['vader_compound'] = vs.apply(lambda x: x['compound'])
	d['vader_neg'] = vs.apply(lambda x: x['neg'])
	d['vader_neu'] = vs.apply(lambda x: x['neu'])
	d['vader_pos'] = vs.apply(lambda x: x['pos'])

	d.to_csv('../output/sentiment.csv', encoding='utf8', index=False)

if __name__ == '__main__':
    main()