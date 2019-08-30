#!/bin/bash
wget -O snli_data.zip https://nlp.stanford.edu/projects/snli/snli_1.0.zip 
unzip snli_data -d snli_data
rm -rf snli_data.zip
