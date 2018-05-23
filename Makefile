# default

default:

# aliases

# tasks

clean:
	rm -rf resources/wiki-news*

toxic:
	mkdir -p datasets/toxic && kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p datasets/toxic && cd datasets/toxic && unzip test.csv.zip && unzip train.csv.zip && cd -

coling-english:
	mkdir -p datasets/coling && wget -O datasets/coling/english_train.tar.gz https://www.dropbox.com/s/r0mnf6b4sktznek/english_train.tar.gz?dl=1 && cd datasets/coling && tar -xzvf english_train.tar.gz && cd -

fastata: resources/wiki-news-300d-1M-subword.vec
	wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip  -P ./resources/ && cd resources && unzip wiki-news-300d-1M-subword.vec.zip && rm wiki-news-300d-1M-subword.vec.zip && cd .. 

install:
	git pull --recurse-submodules && pip3 install -r requirements.txt

test:
	python3 -m pytest tests

.PHONY: default clean coling-english fastdata install test toxic
