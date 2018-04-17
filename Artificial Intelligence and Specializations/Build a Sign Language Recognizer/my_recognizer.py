import warnings
from asl_data import SinglesData
from random import random


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []

    words = test_set.get_all_sequences()

    for test_id in words:
      test_dict = {}
      temp_word = None
      log_max = float('-inf')
      X, lengths = test_set.get_item_Xlengths(test_id)
      for word in models:
        try:
          test_dict[word] = models[word].score(X, lengths)
          if test_dict[word] > log_max:
            log_max = test_dict[word]
            temp_word = word
        except:
          continue

      probabilities.append(test_dict)
      guesses.append(temp_word)


    # TODO implement the recognizer
    return probabilities, guesses