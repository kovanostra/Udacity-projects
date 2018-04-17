import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        bic_min = float('inf')
        model = None
        bic = 0
        for num_states in range(self.min_n_components, self.max_n_components + 1):

            # Iterate among all possible number of components 
            try:                
                temp_model = self.base_model(num_states).fit(self.X, self.lengths)
                num_features = len(self.X[0])
                df = num_states**2 + 2*num_features*num_states - 1
                bic = -2*temp_model.score(self.X, self.lengths) + df*np.log(len(self.X))
            except:
                continue

            if bic != 0 and bic < bic_min:
                model = temp_model
                bic_min = bic

        return model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        dic_max = float('-inf')
        model = None
        dic = 0
        logL = 0
        for num_states in range(self.min_n_components, self.max_n_components + 1):

            # Iterate among all possible number of components 
            try:                
                temp_model = self.base_model(num_states).fit(self.X, self.lengths)
                logL = temp_model.score(self.X, self.lengths)
                for word in self.words:
                    if word != self.this_word:
                        temp_X, temp_lengths = self.hwords[word]
                        dic += temp_model.score(temp_X, temp_lengths)
                dic = logL - dic/(len(self.words) - 1) 
            except:
                continue

            if dic != 0 and dic > dic_max:
                model = temp_model
                dic_max = dic

        return model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        ''' Model selection using CV

            :return: GaussianHMM object
        '''
        split_method = KFold()
        if len(self.sequences) < 3:
            split_method = KFold(2)

        logL_max = float('-inf')
        model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):

            # Iterate among all possible number of components
            logL = 0
            
            start = 0 # Start index
            end = 0 # End index
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    # K-fold cross-validation loop
                    end += sum(self.lengths[cv_test_idx[0]:cv_test_idx[-1] + 1]) # End of the test index
                    
                    train_X = np.delete(self.X, range(start, end), 0) # Remove the test set from the self.X
                    train_lengths = np.delete(self.lengths, cv_test_idx, 0) # Remove the test lengths from the self.lengths

                    temp_model = self.base_model(num_states).fit(train_X, train_lengths)
                    logL += temp_model.score(self.X[start:end],
                                             self.lengths[cv_test_idx[0]:cv_test_idx[-1] + 1])
                    start = end
            except:
                continue

            mean_logL = logL/split_method.n_splits
            if logL != 0 and mean_logL > logL_max:

                # Checks whether the average logL exists and is the maximum among all components
                model = temp_model
                logL_max = mean_logL

        return model
                



