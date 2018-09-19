This repo contains the code and data needed to reproduce the results in [Turetsky & Riddle (2018)](http://www.travisriddle.com/docs/turetsky_riddle_inpress.pdf).

To be fully reproducible, one also needs to download google's pre-trained word2vec word embeddings. They are located [here](https://code.google.com/archive/p/word2vec/). If interested in an end-to-end run, one should run the analysis files in the following order:

- edge_processing.py
- compute_sentiment.py 
- dict_methods.py
- spps_analysis.R

Note that there was one manual step that ocurred between edge_processing.py and compute_sentiment.py. Specifically, we manually cleaned up the resulting edgelist file to remove near-duplicate entries and to make the outlinks consistent with the names of the sources we used in our analyses (e.g. abc7news.com and abcnews.go.com were both converted to ABC). 

Additionally, one should double check all file paths to ensure they are consistent with the directory structure on the users's machine.