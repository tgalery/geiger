## Geiger

A geiger counter for online radioactivity.
This repo emerged as a playground for text classification approaches to a variety of competitions and shared texts related to online toxicity and abuse detection, including:

- [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Coling Trolling, Aggression, and Cyber-bullying](https://sites.google.com/view/trac1/home?authuser=0) 

## Setup

A makefile has been supplied for conveniently downloading resources and data.

- `make coling-english` : downloads and unpacks the english training and development data from the  Coling Trolling, Aggression, and Cyber-bullying shared task and places under `data/`.
- `make fastata`: download fastText vectors to `resources/`
- `make install`: pull submodules dependencies including Babylon's [fastText multilingual](https://github.com/Babylonpartners/fastText_multilingual) and python library dependencies.
- `make toxic`: downloads and unpacks dataset from the Toxic Classification Challenge. Note that this requires [kaggles cli](https://github.com/Kaggle/kaggle-api) to be installed and properly configured.  
- `make clean`: removes data in `resources/`
