This repo contains the code and data needed to reproduce the results in [Turetsky & Riddle (2018)](http://www.travisriddle.com/docs/turetsky_riddle_inpress.pdf).

To be fully reproducible, one also needs to download google's pre-trained word2vec word embeddings. They are located [here](https://code.google.com/archive/p/word2vec/). If interested in an end-to-end run, one should run the analysis files in the following order:

- edge_processing.py
- clean_edgelist.R
- compute_sentiment.py 
- dict_methods.py
- spps_analysis.R

One should double check all file paths to ensure they are consistent with the directory structure on the users's machine.