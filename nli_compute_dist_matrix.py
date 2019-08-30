import pickle
import numpy as np
import glove_utils

with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)

nli_words_index = tokenizer.word_index

inv_word_index = {i: w for (w, i) in nli_words_index.items()}
MAX_VOCAB_SIZE = len(nli_words_index)
# Load the counterfitted-vectors (used by our attack)
glove2 = glove_utils.loadGloveModel('counter-fitted-vectors.txt')
# create embeddings matrix for our vocabulary
counter_embeddings, missed = glove_utils.create_embeddings_matrix(
    glove2, nli_words_index, None)

# save the embeddings for both words we have found, and words that we missed.
np.save(('aux_files/nli_embeddings_counter_%d.npy' %
         (MAX_VOCAB_SIZE)), counter_embeddings)
np.save(('aux_files/nli_missed_embeddings_counter_%d.npy' %
         (MAX_VOCAB_SIZE)), missed)

print('Done preparing the embedding matrix.')
print('Computing the distance matrix.. this may take a while')
c_ = -2*np.dot(counter_embeddings.T, counter_embeddings)
a = np.sum(np.square(counter_embeddings), axis=0).reshape((1, -1))
b = a.T
dist = a+b+c_
np.save(('aux_files/nli_dist_counter_%d.npy' % (MAX_VOCAB_SIZE)), dist)

# Try an example

src_word = nli_words_index['good']
neighbours, neighbours_dist = glove_utils.pick_most_similar_words(
    src_word, dist)
print('Closest words to `good` are :')
result_words = [inv_word_index[x] for x in neighbours]
print(result_words)
