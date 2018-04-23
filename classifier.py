#!/usr/bin/env python3

import os
import glob
import random
from math import log10
from collections import defaultdict
from typing import Dict, List

class NBClassifier:
    """Implements a Naïve Bayes Classifier"""

    def __init__(self, documents: Dict[str, List[str]]):
        """
        Trains the classifier.

        @param documents: class label -> [path_to_doc_1, path_to_doc_2, ...]
        """
        self._classes = documents.keys()
        self._vocabulary = set()
        self._word_freqs = {c: defaultdict(int) for c in self._classes}
        self._logprior = defaultdict(float)
        self._loglikelihood = {c: defaultdict(float) for c in self._classes}
        # read word frequencies from training data
        num_docs_of_any_class = sum([len(d) for d in documents.values()])
        for current_class, docs in documents.items():
            for doc in docs:
                for word, frequency in self._get_word_freqs(doc).items():
                    self._word_freqs[current_class][word] += frequency
                    self._vocabulary.add(word)
        raise NotImplementedError("Calculate self._logprior and "
                                  "self._loglikelihood for all classes and "
                                  "words. Don't forget about smoothing.") # TODO

    def classify(self, document: str):
        """
        Returns the most likely class label for @param document.
        """
        # helper: calculate log probability by class
        def _get_log_probability(current_class, document):
            log_probability = self._logprior[current_class]
            for word in self._get_words(document):
                if word in self.vocabulary: # slow, but easy to understand
                    log_probability += self._loglikelihood[current_class][word]
            return log_probability
        log_probabilities_per_class = {c: _get_log_probability(c, document) for c in self.classes}
        return sorted(log_probabilities_per_class, key=log_probabilities_per_class.get, reverse=True)[0]

    def evaluate(self, documents: Dict[str, List[str]]):
        """
        Evaluates the classifier.

        @type documents: {str: list(str)}
        @param documents: class label -> [path_to_doc_1, path_to_doc_2, ...]

        Returns overall classification accuracy.
        """
        num_items = 0
        num_correct = 0
        tp = defaultdict(int) # true positive
        fp = defaultdict(int) # false positive
        fn = defaultdict(int) # false negative
        for true_label, docs in documents.items():
            for doc in docs:
                num_items += 1
                predicted_label = self.classify(doc)
                if true_label == predicted_label:
                    num_correct += 1
                    tp[true_label] += 1
                else:
                    fn[true_label] += 1
                    fp[predicted_label] += 1
        # overall classification accuracy
        accuracy = num_correct / num_items
        print("Classifier accuracy: {:0.2f}%.".format(accuracy*100))
        # precision, recall, f-measure per class
        for c in self.classes:
            precision = tp[c] / (tp[c] + fp[c])
            recall = tp[c] / (tp[c] + fn[c])
            f1score = 2 * ((precision * recall) / (precision + recall))
            print("Class {}:\n\t{:0.2f} precision\n\t{:0.2f} recall\n\t{:0.2f} F1-score".format(
                c, precision, recall, f1score
            ))

    @property
    def vocabulary(self):
        """
        Returns the classifier's vocabulary.
        """
        return self._vocabulary

    @property
    def classes(self):
        """
        Returns the class labels this classifier can assign.
        """
        return self._classes

    @staticmethod
    def _get_words(path_to_document: str):
        """
        Reads a document stored at @param path_to_document. Returns the words it
        contains as a list.
        """
        words = []
        with open(path_to_document, 'r') as f:
            for line in f:
                for word in line.split():
                    words.append(word)
        return words

    @staticmethod
    def _get_word_freqs(path_to_document: str):
        """
        Reads a document stored at @param path_to_document. Returns the words it
        contains, alongside their frequency, as a dictionary.
        """
        word_freqs = defaultdict(int)
        with open(path_to_document, 'r') as f:
            for line in f:
                for word in line.split():
                    word_freqs[word] += 1
        return word_freqs


if __name__ == "__main__":
    """
    Trains a Naïve Bayes classifier on 9/10 of the ham and spam documents. Uses
    the remainder for evaluation.
    """
    # read data, shuffle, and split into training and evaluation set
    docs = glob.glob('data/bare/*/*.txt')
    num_docs = len(docs)
    num_eval = int(num_docs / 10)
    num_train = num_docs - num_eval
    random.shuffle(docs)
    def format(docs):
        formatted_docs = {'ham': [], 'spam': []}
        for doc in docs:
            class_label = 'spam' if os.path.basename(doc).startswith('spmsg') else 'ham'
            formatted_docs[class_label].append(doc)
        return formatted_docs
    docs_eval = format(docs[:num_eval])
    docs_train = format(docs[num_eval:])
    print("Found {0} documents. Using {1} for training, {2} for evaluation."
          .format(num_docs, num_train, num_eval))
    print("Training...")
    # train
    classifier = NBClassifier(docs_train)
    # evaluate
    print("Evaluating...")
    classifier.evaluate(docs_eval)
