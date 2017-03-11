#!/usr/bin/env python
"""
K nearest neighbor classifier for Amazon reviews classification
"""

import numpy as np
import scipy as sp
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import re
import copy
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from stemming.porter2 import stem

print_reviews = False
print_debug = True
mode1 = "evaluation"
mode2 = "test"
evaluation_k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

def define_files (mode, k):
    k_values = k
    if mode == mode1:
        actual_train_file = "train.dat"
        train_file = "sample_train.dat"
        test_file = "evaluation.dat"
        return actual_train_file, train_file, test_file, k_values

    elif mode == mode2:
        train_file = "train.dat"
        test_file = "test.dat"
        result_file = "format.dat"
        return train_file, test_file, result_file, k_values

    else:
        print "Mode is invalid \n\n"
        # exit


def prepare_files(actual_train_file, sample_train_file, sample_evaluation_file):
    with open(actual_train_file, "r") as fh:
        lines = fh.readlines()

    random.shuffle(lines)
    train_lines = lines[0:len(lines)/2]
    evaluation_lines = lines[len(lines)/2:]

    # Remove all rating info from evaluation file
    evaluation_labels = [int(l[:2]) for l in evaluation_lines]

    modified_evaluation_lines = []
    for t in evaluation_lines:
        t = t[2:]
        modified_evaluation_lines.append(t)

    # Create new test and train files
    train = open(sample_train_file, 'w')
    for t in train_lines:
        train.write(t)
    train.close()

    test = open(sample_evaluation_file, 'w')
    for t in modified_evaluation_lines:
        test.write(t)
    test.close()

    return evaluation_labels


def preprocess(train_file, test_file, result_file, evaluation_labels, mode, k_values):

    if print_debug is True:
        print 'Processing training file'
    # 1. Open train file
    with open(train_file, "r") as fh:
        lines = fh.readlines()

    # 2. Separate the labels
    train_labels = [int(l[:2]) for l in lines]

    # 3. Remove special characters and convert all words to lowercase
    train_docs = [re.sub(r'[^\w]', ' ',l[2:].lower()).split() for l in lines]

    # 4. Remove words with length less than 4
    train_reviews_1 = filterLen(train_docs, 4)
    train_reviews = stemDoc(train_reviews_1)
    for t in train_reviews:
        new_list = get_k_mers(t)
        t.extend(new_list)
    #print t
    num_train_samples = len(train_reviews)

    if print_debug is True:
        print 'Processing test file'
    # 5. Repeat above steps with test data
    with open(test_file, "r") as fh:
        test_lines = fh.readlines()
    test_docs = [re.sub(r'[^\w]', ' ',l.lower()).split() for l in test_lines]
    test_reviews_1 = filterLen(test_docs, 4)
    test_reviews = stemDoc(test_reviews_1)
    for t in test_reviews:
        new_list = get_k_mers(t)
        t.extend(new_list)
    num_test_samples = len(test_reviews)

    # 6. Combine train_reviews and test_reviews
    train_reviews.extend(test_reviews)

    if print_debug is True:
        print 'Building CSR matrix'
    # 7. Build csr_matrix with train and test reviews
    csr_mat = build_matrix(train_reviews)

    # 8. Decrease importance of popular words with IDF and
    # Normalize the matrix to simplify cosine similarity calculation
    mat1 = csr_idf(csr_mat, copy=True)
    mat = csr_l2normalize(mat1, copy=True)

    if print_debug is True:
        print 'Computing cosine similarity'
    # 8.1 Compute cosine similarity on sparse matrix
    similarities_sparse = cosine_similarity(mat,dense_output=False)
    # print('pairwise sparse output:\n {}\n'.format(similarities_sparse))

    if print_debug is True:
        print 'Computing KNN for test reviews'
    # 9. For each test review find the k-nearest neighbor
    all_test_labels = []
    for k in k_values:
        test_labels = []
        for test_review_index in range(num_train_samples, num_train_samples+num_test_samples):
            similarity = similarities_sparse[test_review_index, :num_train_samples].toarray().tolist()[0]
            # append with label information
            similarity_with_labels = zip(similarity, train_labels, range(len(train_labels)))

            '''
            if test_review_index % 500 == 0:
                print_reviews = True
            else:
                print_reviews = False
            '''

            # sort for k - nearest
            sorted_similarity_with_labels = sorted(similarity_with_labels, key=lambda (val, k, l): val, reverse=True)
            if print_reviews:
                print 'Computing labels through nearet neighbor for review:\n', test_reviews[test_review_index-num_train_samples], '\n\n Found following nearest reviews'
            # Choose top k values from each of the sorted list and find test label
            tmp = 0

            for j in range(k):
                if sorted_similarity_with_labels[j][0] != 0:
                    tmp += int(sorted_similarity_with_labels[j][1])
                if print_reviews:
                    print train_reviews[sorted_similarity_with_labels[j][2]], sorted_similarity_with_labels[j][1]
            if tmp == 0:
                while tmp == 0:
                    tmp = np.random.randint(-1,2)
            if tmp > 0:
                test_labels.append(1)
                tst = 1
            else:
                test_labels.append(-1)
                tst = -1

            if print_reviews:
                print 'computed label is: ',tst, '\n'

        all_test_labels.append(test_labels)

    # Check accuracy of labels with different k values
    if mode == mode1:
        temp = []
        count = []
        for idx, a in enumerate(all_test_labels):
            print "Classification report for k = ", evaluation_k[idx]
            print metrics.classification_report(evaluation_labels, a)
            n = 0
            temp = np.isclose(a, evaluation_labels)
            for t in temp:
                if t == True:
                    n = n + 1
            count.append(n)
            print "Count for k ", n
        # Return the best k value
        if print_debug is True:
            print 'Best k value ', evaluation_k[count.index(max(count))]
        return evaluation_k[count.index(max(count))]

    elif mode == mode2:
        # Write the Best k test labels to format.dat file
        target = open(result_file, 'w')
        for t in all_test_labels[0]:
            if t == 1:
                target.write("+1")
            else:
                target.write("-1")
            target.write("\n")
        target.close()

    else:
        print "Wrong mode! Inside preprocess"

    return test_labels

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents,
    each of which is a list of word/terms in the document.
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    # Remove all ratings
    for d in docs:
        #d = d[1:]
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)

    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        #d = d[1:]
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()

    return mat

#@profile
def filterLen(docs, minlen):
    """ filter out terms that are too short.
    docs is a list of lists, each inner list is a document represented as a
    list of words minlen is the minimum length of the word to keep
    """
    return [ [t for t in d if len(t) >= minlen ] for d in docs ]

def stemDoc(docs):
    """ automatically removes suffixes (and in some cases prefixes) in order to
    find the root word or stem of a given word
    """
    return [ [stem(t) for t in d ] for d in docs]

def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf.
    Returns scaling factors as dict. If copy is True,
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat

#@profile
def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm.
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum

    if copy is True:
        return mat

# Group words for different K values for K-mer implementation
def grouper(input_list, n = 2):
    for i in xrange(len(input_list) - (n - 1)):
        yield input_list[i:i+n]

def get_k_mers(input_list):
    new_list = []
    #new_list.extend(input_list)
    for first, second in grouper(input_list, 2):
        st = first + " "+second
        new_list.append(st)

    for first, second, third in grouper(input_list, 3):
        st = first + " "+second + " "+third
        new_list.append(st)

    return new_list

if __name__ == '__main__':
    # 1. For evaluation mode to find the best k value
    actual_train_file, train_file, evaluation_file, k_values = define_files(mode1, evaluation_k)
    evaluation_labels = prepare_files(actual_train_file, train_file, evaluation_file)
    k = preprocess(train_file, evaluation_file, "",evaluation_labels, mode1, k_values)

    best_k = []
    best_k.append(k)

    #2. Actual testing with original train and test files
    train_file, test_file, result_file, k_value = define_files(mode2, best_k)
    test_labels = preprocess(train_file, test_file, result_file, [], mode2, k_value)

    print 'Test label length'
    print len(test_labels)
