#!/usr/bin/env bash

PWD=$(pwd)

UBUNTU_DIR=$PWD/ubuntu
mkdir -p $UBUNTU_DIR

# Download Ubuntu dialogue corpus
wget https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=1 -O $UBUNTU_DIR/ubuntu_data.zip
unzip -j $UBUNTU_DIR/ubuntu_data.zip -d $UBUNTU_DIR

# Download the word vocabulary file, pre-trained word embedding file, char vocabulary file, and pre-trained char embedding file. 
wget https://www.dropbox.com/s/71zysromhd8642o/word_char_word2vec.zip?dl=1 -O $UBUNTU_DIR/word_char_word2vec.zip
unzip -j $UBUNTU_DIR/word_char_word2vec.zip -d $UBUNTU_DIR
