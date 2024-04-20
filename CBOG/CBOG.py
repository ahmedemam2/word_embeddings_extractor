import re
import nltk
from nltk.tokenize import word_tokenize
# import emoji
from matplotlib import pyplot
from collections import defaultdict
import re
import numpy as np

from collections import Counter

def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)
    return e_z/sum_e_z

def relu(z):
    return np.maximum(0, z)


def forward_prop(x, W1, W2, b1, b2):
    h = np.dot(W1, x) + b1

    h = relu(h)

    z = np.dot(W2, h) + b2

    return z, h

def compute_cost(y, yhat, batch_size):

    logprobs = np.multiply(np.log(yhat),y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


def get_dict(data):

    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word

def compute_pca(data, n_components):

    m, n = data.shape
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]
    ### END CODE HERE ###
    return np.dot(evecs.T, data.T).T

def initialize_model(N, V, random_seed=1):
    # np.random.seed(random_seed)
    # W1 has shape (N,V)
    W1 = np.random.rand(N, V)

    # W2 has shape (V,N)
    W2 = np.random.rand(V, N)

    # b1 has shape (N,1)
    b1 = np.random.rand(N, 1)

    # b2 has shape (V,1)
    b2 = np.random.rand(V, 1)

    ### END CODE HERE ###
    return W1, W2, b1, b2

def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []

# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C4 GRADED FUNCTION: back_prop
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    z1 = np.dot(W1, x) + b1

    l1 = np.dot(W2.T, yhat - y)

    l1[z1 < 0] = 0

    grad_W1 = np.dot(l1, x.T) / batch_size

    grad_W2 = np.dot(yhat - y, h.T) / batch_size

    grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size

    grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / batch_size

    return grad_W1, grad_W2, grad_b1, grad_b2


def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.001,
                     random_seed=282, initialize_model=initialize_model,
                     get_batches=get_batches, forward_prop=forward_prop,
                     softmax=softmax, compute_cost=compute_cost,
                     back_prop=back_prop):

    W1, W2, b1, b2 = initialize_model(N, V, random_seed=random_seed)  # W1=(N,V) and W2=(V,N)

    batch_size = 128
    #    batch_size = 512
    iters = 0
    C = 2

    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        ### START CODE HERE (Replace instances of 'None' with your own code) ###
        # get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)

        yhat = softmax(z)

        cost = compute_cost(y, yhat, batch_size)
        if ((iters + 1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")

        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)

        W1 = W1 - alpha * grad_W1
        W2 = W2 - alpha * grad_W2
        b1 = b1 - alpha * grad_b1
        b2 = b2 - alpha * grad_b2

        iters += 1
        if iters == num_iters:
            break
        if iters % 5 == 0:
            alpha *= 0.11

    return W1, W2, b1, b2

def get_idx(words, word2Ind):
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx

def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed

def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i - C) : i] + data[(i + 1) : (i + C + 1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq / num_ctx_words
        yield x, y
        i += 1
        if i >= len(data):
            print("i is being set to 0")
            i = 0



def visualize(W1,W2, word2Ind):
    words = ['king', 'queen', 'lord', 'man', 'woman', 'dog', 'wolf',
             'rich', 'happy', 'sad']

    embs = (W1.T + W2) / 2.0

    idx = [word2Ind[word] for word in words]
    X = embs[idx, :]
    result = compute_pca(X, 2)
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()



def main():
    with open('shakespeare.txt') as f:
        data = f.read()  # Read in the data
    data = re.sub(r'[,!?;-]', '.', data)  # Punktuations are replaced by .
    data = nltk.word_tokenize(data)  # Tokenize string to words
    data = [ch.lower() for ch in data if ch.isalpha() or ch == '.']  # Lower case and drop non-alphabetical tokens
    print(len(data))
    word2Ind, Ind2word = get_dict(data)
    V = len(word2Ind)
    print(V)

    # In the initialization stage of the model, two matrices and two vectors are initialized. The first
    # matrix (𝑊1) has dimensions 𝑁×𝑉, where 𝑉 represents the number of words in the vocabulary and 𝑁
    # represents the dimensionality of the word vectors. The second matrix (𝑊2) has dimensions 𝑉×𝑁.
    # Additionally, two vectors, 𝑏1 and 𝑏2, are initialized. 𝑏1 is a bias vector with dimensions 𝑁×1,
    # associated with the linear layer from matrix 𝑊1, while 𝑏2 is a bias vector with dimensions 𝑉×1,
    # associated with the linear layer from matrix 𝑊2.

    #Forward, cost, gradient, backward
    C = 2
    N = 50
    word2Ind, Ind2word = get_dict(data)
    V = len(word2Ind)
    num_iters = 150
    W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)
    words = ['king', 'queen', 'lord', 'man', 'woman', 'dog', 'prince',
             'cat', 'happy', 'sad', "angry"]

    embs = (W1.T + W2) / 2

    #visualizing using pca
    idx = [word2Ind[word] for word in words]
    X = embs[idx, :]
    result = compute_pca(X, 2)
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

main()