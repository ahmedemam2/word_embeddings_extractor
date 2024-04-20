import re
import nltk
from nltk.tokenize import word_tokenize
# import emoji
import numpy as np


# nltk.download('punkt')  # download pre-trained Punkt tokenizer for English

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

def tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = nltk.word_tokenize(data)  # tokenize string to words
    data = [ ch.lower() for ch in data
             if ch.isalpha()
             or ch == '.'
             # or emojis.get_emoji_regexp().search(ch)
           ]
    return data

def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i+1):(i+C+1)]
        yield context_words, center_word
        i += 1


def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1

    return one_hot_vector


def context_words_to_vector(context_words, word2Ind, V):
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)

    return context_words_vectors

def get_training_example(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)

def relu(z):
    result = z.copy()
    result[result < 0] = 0

    return result


def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)

    return e_z / sum_e_z


def cross_entropy_loss(y_predicted, y_actual):
    loss = np.sum(-np.log(y_predicted) * y_actual)

    return loss

def main():
    corpus = 'I am happy because I am learning'
    print(f'Corpus:  {corpus}')
    words = tokenize(corpus)
    print(f'Words (tokens):  {words}')
    print()
    word2Ind, Ind2word = get_dict(words)
    V = len(word2Ind)
    for context_words, center_word in get_windows(words, 2):  # reminder: 2 is the context half-size
        print(f'Context words:  {context_words} -> {context_words_to_vector(context_words, word2Ind, V)}')
        print(f'Center word:  {center_word} -> {word_to_one_hot_vector(center_word, word2Ind, V)}')
        print()


    # The two activation functions used in the neural network.
    # softmax,ReLU

    # Forward propagation.
    N = 3 #dims of word embeddings also numbers of neurons in hidden layer.
    W1 = np.array([[0.41687358, 0.08854191, -0.23495225, 0.28320538, 0.41800106],
                   [0.32735501, 0.22795148, -0.23951958, 0.4117634, -0.23924344],
                   [0.26637602, -0.23846886, -0.37770863, -0.11399446, 0.34008124]])

    W2 = np.array([[-0.22182064, -0.43008631, 0.13310965],
                   [0.08476603, 0.08123194, 0.1772054],
                   [0.1871551, -0.06107263, -0.1790735],
                   [0.07055222, -0.02015138, 0.36107434],
                   [0.33480474, -0.39423389, -0.43959196]])

    b1 = np.array([[0.09688219],
                   [0.29239497],
                   [-0.27364426]])

    b2 = np.array([[0.0352008],
                   [-0.36393384],
                   [-0.12775555],
                   [-0.34802326],
                   [-0.07017815]])

    training_examples = get_training_example(words, 2, word2Ind, V)
    x_array, y_array = next(training_examples)
    x = x_array.copy()
    x.shape = (V, 1)
    print('x')
    print(x)
    print()

    y = y_array.copy()
    y.shape = (V, 1)
    print('y')
    print(y)
    z1 = np.dot(W1, x) + b1
    h = relu(z1)
    z2 = np.dot(W2, h) + b2
    y_hat = softmax(z2)
    print(y_hat)
    print(Ind2word[np.argmax(y_hat)])
    index_of_max = np.argmax(y_hat)
    print(y_array[index_of_max] == 1)

    # Cross-entropy loss
    print(cross_entropy_loss(y_hat, y))

    # Backpropagation.
    # ∂𝐽∂𝐖1∂𝐽∂𝐖2∂𝐽∂𝐛1∂𝐽∂𝐛2 = ReLU(𝐖⊤2(𝐲̂ −𝐲))𝐱⊤=(𝐲̂ −𝐲)
    # 𝐡⊤=ReLU(𝐖⊤2(𝐲̂ −𝐲))=𝐲̂ −𝐲
    grad_b2 = y_hat - y
    grad_W2 = np.dot(y_hat - y, h.T)
    grad_b1 = relu(np.dot(W2.T, y_hat - y))
    grad_W1 = np.dot(relu(np.dot(W2.T, y_hat - y)), x.T)

    # Gradient descent.
    alpha = 0.03
    # 𝐖2𝐛1𝐛2 := 𝐖2−𝛼∂𝐽∂𝐖2 := 𝐛1−𝛼∂𝐽∂𝐛1 := 𝐛2−𝛼∂𝐽∂𝐛2
    W1_new = W1 - alpha * grad_W1
    W2_new = W2 - alpha * grad_W2
    b1_new = b1 - alpha * grad_b1
    b2_new = b2 - alpha * grad_b2
    # Extracting the word embedding vectors from the weight matrices once the neural network has been trained.
    # loop through each word of the vocabulary
    for word in word2Ind:
        # extract the column corresponding to the index of the word in the vocabulary
        word_embedding_vector = W1[:, word2Ind[word]]

        print(f'{word}: {word_embedding_vector}')
    # loop through each word of the vocabulary
    for word in word2Ind:
        # extract the column corresponding to the index of the word in the vocabulary
        word_embedding_vector = W2.T[:, word2Ind[word]]

        print(f'{word}: {word_embedding_vector}')
    W3 = (W1 + W2.T) / 2
    # loop through each word of the vocabulary
    for word in word2Ind:
        # extract the column corresponding to the index of the word in the vocabulary
        word_embedding_vector = W3[:, word2Ind[word]]

        print(f'{word}: {word_embedding_vector}')



main()