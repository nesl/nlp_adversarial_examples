1) Download the Imdb dataset
```
./download_dataset.sh
```

2) Download the glove vector embeddings (used by the model)
```
 ./download_glove.sh 
```

3) Download the counter-fitted vectors (used by our attack)
```
./download_counterfitted_vectors.sh 
```

4) Build the vocabulary and embeddings matrix.
```
python build_embeddings.py
```

That will take like a minute, and it will tokenize the dataset and save it to a pickle file. It will also compute some auxiliary files like the matrix of the vector embeddings for words in our dictionary. All files will be saved under `aux_files` directory created by this script.

5) Train the sentiment analysis model.
```
python train_model.py
```

6)Download the Google language model.
```
./download_googlm.sh
```

7) Pre-compute the distances between embeddings of different words (required to do the attack) and save the distance matrix.

```
python compute_dist_mat.py 

```
8) Now, we are ready to try some attacks ! You can do so by running the [`IMDB_AttackDemo.ipynb`](IMDB_AttackDemo.ipynb) Jupyter notebook !


### Attacking Textual Entailment model

The model we are using for our experiment is the SNLI model of [Keras SNLI Model](https://github.com/Smerity/keras_snli) .

First, Download the dataset using 
```
bash download_snli_data.sh
```

Download the Glove and Counter-fitted Glove embedding vectors

```
bash ./download_glove.sh
bash ./download_counterfitted_vectors.sh
```

Train the NLI model
```
python sni_rnn.py
```

Pre-compute the embedding matrix 
```
python nli_compute_dist_matrix.py
```

Now, you are ready to run the attack using example code provided in [`NLI_AttackDemo.ipynb`](NLI_AttackDemo.ipynb) Jupyter notebook.
