# This Python file uses the following encoding: utf-8

import os
import glob
import pandas as pd
import re

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
    
def get_links(texts):
    '''
    Takes a dataframe (as constructed above) and extracts all the links from 
    the text of the articles. Returns a dictionary of links where the key is 
    the source url and the entry is the site the found link points to
    '''
    prog = re.compile(ur'<hyper.*?=("|”)?(.*?)("|”)?>')
    links = {k:[] for k in texts.Source}
    d = texts.dropna(how='all')
    for i, t in enumerate(texts['ArticleText']):
        try:
            t = t.replace('“','"').replace('”','"')
            result = prog.findall(t.decode('utf8'))
            if result:
                links[texts.Source.iloc[i]].append([r[1] for r in result])
        except:
            print (d.ArticleID.iloc[i])
    return(links)

def edge_list(linkdict):
    '''
    takes a dictionary of links (as constructed above) and creates a simple 
    edge list that consists of the source url and the domain that the found 
    links points to
    '''
    prog = re.compile(ur'https?:\/\/(www.)?(.*?)\/')
    edges = []
    for k in linkdict.keys():
        for article in linkdict[k]:
            for link in article:
                m=prog.match(link)
                if m:
                    edges.append([k, m.group(2)])
    return(edges)

def main():
    '''
    main prog writes an ouptut that is of the form:
    source - link target - num occurrnces
    '''
    print 'Grabbing Data'
    d = get_dataframe('../FergusonMedia/DATA/')
    print ''
    print 'Extracting Links'
    links = get_links(d)
    print ''
    print 'Constructing edge list'
    edges = edge_list(links)
    d=pd.DataFrame(edges)
    grouped = d.groupby([0,1])
    grouped.size().reset_index(level=1).to_csv('../output/edgelist.csv')

if __name__ == '__main__':
    main()