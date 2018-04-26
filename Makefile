# default

default:

# aliases

# tasks

clean:
	rm -rf resources/wiki-news*

fastdata: resources/wiki-news-300d-1M-subword.vec
	wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip  -P ./resources/ && cd resources && unzip wiki-news-300d-1M-subword.vec.zip && rm wiki-news-300d-1M-subword.vec.zip && cd .. 

install: fastext
	cd fastText && pip3 install . && cd ..

test:
	python3 -m pytest tests

.PHONY: default clean fastdata install test
