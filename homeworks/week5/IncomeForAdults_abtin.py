#Problem statement - https://abtinshahidi.github.io/files/week5.pdf
# This is an assignment to find the income of adults dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import copy
import math
import random

from statistics import mean, stdev
from collections import defaultdict


def euclidean_distance(X, Y):
    return math.sqrt(sum((x - y)**2 for x, y in zip(X, Y)))


def cross_entropy_loss(X, Y):
    n=len(X)
    return (-1.0/n)*sum(x*math.log(y) + (1-x)*math.log(1-y) for x, y in zip(X, Y))


def rms_error(X, Y):
    return math.sqrt(ms_error(X, Y))


def ms_error(X, Y):
    return mean((x - y)**2 for x, y in zip(X, Y))


def mean_error(X, Y):
    return mean(abs(x - y) for x, y in zip(X, Y))


def manhattan_distance(X, Y):
    return sum(abs(x - y) for x, y in zip(X, Y))


def mean_boolean_error(X, Y):
    return mean(int(x != y) for x, y in zip(X, Y))


def hamming_distance(X, Y):
    return sum(x != y for x, y in zip(X, Y))


def _read_data_set(data_file, skiprows=0, separator=None):
    with open(data_file, "r") as f:
        file = f.read()
        lines = file.splitlines()
        lines = lines[skiprows:]

    data_ = [[] for _ in range(len(lines))]

    for i, line in enumerate(lines):
        splitted_line = line.split(separator)
        float_line = []
        for value in splitted_line:
            try:
                value = float(value)
            except ValueError:
                if value == "":
                    continue
                else:
                    pass
            float_line.append(value)
        if float_line:
            data_[i] = float_line

    for line in data_:
        if not line:
            data_.remove(line)

    return data_


def unique(seq):
    """
    Remove any duplicate elements from any sequence,
    works on hashable elements such as int, float,
    string, and tuple.
    """
    return list(set(seq))


def remove_all(item, seq):
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]


def weighted_sample_with_replacement(n, seq, weights):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight."""
    sample = weighted_sampler(seq, weights)

    return [sample() for _ in range(n)]


def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    import bisect

    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def mode(data):
    import collections
    """Return the most common data item. If there are ties, return any one of them."""
    [(item, count)] = collections.Counter(data).most_common(1)
    return item


# argmin and argmax

identity = lambda x: x

argmin = min
argmax = max


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return argmin(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return argmax(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


def check_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)

class Data_Set:
    """
    Defining a _general_ data set class for machine learning.

    These are the following fields:

    >> data = Data_set

    data.examples:

           list of examples. Each one is a list contains of attribute values.


    data.attributes:

           list of integers to index into an example, so example[attribute]
           gives a value.


    data.attribute_names:

           list of names for corresponding attributes.


    data.target_attribute:

           The target attribute for the learning algorithm.
           (Default = last attribute)


    data.input_attributes:

           The list of attributes without the target.



    data.values:

           It is a list of lists in which each sublist is the
           set of possible values for the corresponding attribute.
           If initially None, it is computed from the known examples
           by self.setproblem. If not None, bad value raises ValueError.


    data.distance_measure:

           A measure  of  distance  function which takes two examples
           and returns a nonnegative number. It should be a symmetric
           function.
           (Defaults = mean_boolean_error) : can handle any field types.


    data.file_info:

           This should be a tuple that contains:
        (file_address, number of rows to skip, separator)



    data.name:

           This is for naming the data set.



    data.source:

            URL or explanation to the dataset main source


    data.excluded_attributes:

            List of attribute indexes to exclude from data.input_attributes.
            (indexes or names of the attributes)

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs."""

    def __init__(self, examples=None, attributes=None, attribute_names=None,
                 target_attribute=-1, input_attributes=None, values=None,
                 distance_measure=mean_boolean_error, name='', source='',
                 excluded_attributes=(), file_info=None):

        """
        Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .setproblem().

        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """

        self.file_info = file_info
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance_measure
        self.check_values_flag = bool(values)

        # Initialize examples from a list
        if examples is not None:
            self.examples = examples
        elif file_info is None:
            raise ValueError("No Examples! and No Address!")
        else:
            self.examples = _read_data_set(file_info[0], file_info[1], file_info[2])

        # Attributes are the index of examples. can be overwrite
        if self.examples is not None and attributes is None:
            attributes = list(range(len(self.examples[0])))

        self.attributes = attributes

        # Initialize attribute_names from string, list, or to default
        if isinstance(attribute_names, str):
            self.attribute_names = attribute_names.split()
        else:
            self.attribute_names = attribute_names or attributes

        # set the definitions needed for the problem
        self.set_problem(target_attribute, input_attributes=input_attributes,
                         excluded_attributes=excluded_attributes)

    def get_attribute_num(self, attribute):
        if isinstance(attribute, str):
            return self.attribute_names.index(attribute)
        else:
            return attribute

    def set_problem(self, target_attribute, input_attributes=None, excluded_attributes=()):
        """
        By doing this we set the target, inputs and excluded attributes.

        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attrname.
        Also computes the list of possible values, if that wasn't done yet."""

        self.target_attribute = self.get_attribute_num(target_attribute)

        exclude = [self.get_attribute_num(excluded) for excluded in excluded_attributes]

        if input_attributes:
            self.input_attributes = remove_all(self.target_attribute, input_attributes)
        else:
            inputs = []
            for a in self.attributes:
                if a != self.target_attribute and a not in exclude:
                    inputs.append(a)
            self.input_attributes = inputs

        if not self.values:
            self.update_values()
        self.sanity_check()

    def sanity_check(self):
        """Sanity check on the fields."""

        assert len(self.attribute_names) == len(self.attributes)
        assert self.target_attribute in self.attributes
        assert self.target_attribute not in self.input_attributes
        assert set(self.input_attributes).issubset(set(self.attributes))
        if self.check_values_flag:
            # only check if values are provided while initializing DataSet
            [self.check_example(example) for example in self.examples]

    def check_example(self, example):
        if self.values:
            for attr in self.attributes:
                if example[attr] not in self.values:
                    raise ValueError("Not recognized value of {} for attribute {} in {}"
                                     .format(example[attr], attr, example))

    def add_example(self, example):
        self.check_example(example)
        self.examples.append(example)

    def update_values(self):
        self.values = list(map(unique, zip(*self.examples)))

    def remove_examples(self, value=""):
        self.examples = [example for example in examples if value not in example]

    def sanitize(self, example):
        """Copy of the examples with non input_attributes replaced by None"""
        _list_ = []
        for i, attr_i in enumerate(example):
            if i in self.input_attributes:
                _list_.append(attr_i)
            else:
                _list_.append(None)
        return _list_

    def train_test_split(self, test_fraction=0.3, Seed=99):
        import numpy as np

        examples = self.examples
        atrrs = self.attributes
        atrrs_name = self.attribute_names
        target = self.target_attribute
        input_ = self.input_attributes
        name = self.name

        np.random.seed(Seed)
        _test_index = np.random.choice(list(range(len(examples))), int(test_fraction * len(examples)), replace=False)

        test_examples = [example for i, example in enumerate(examples) if i in _test_index]

        train_examples = [example for example in examples if example not in test_examples]

        Test_data_set = Data_Set(examples=test_examples,
                                 attributes=atrrs,
                                 attribute_names=attr_names,
                                 target_attribute=target,
                                 input_attributes=input_,
                                 name=name + " Test set", )

        Train_data_set = Data_Set(examples=train_examples,
                                  attributes=atrrs,
                                  attribute_names=attr_names,
                                  target_attribute=target,
                                  input_attributes=input_,
                                  name=name + " Train set", )

        return Train_data_set, Test_data_set

    def __repr__(self):
        return '<DataSet({}): with {} examples, and {} attributes>'.format(
            self.name, len(self.examples), len(self.attributes))



class Decision_Leaf:
    """A simple leaf class for a decision tree that hold a result."""

    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

    def display_out(self, indent=0):
        print('RESULT =', self.result)

    def __repr__(self):
        return repr(self.result)


def Decision_Tree_Learner(dataset):
    """
    Learning Algorithm for a Decision Tree
    """

    target, values = dataset.target_attribute, dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):
        if not examples:
            return plurality(parent_examples)
        elif same_class_for_all(examples):
            return Decision_Leaf(examples[0][target])
        elif not attrs:
            return plurality(examples)
        else:
            A = choose_important_attribute(attrs, examples)
            tree = Decision_Branch(A, dataset.attribute_names[A], plurality(examples))

            for (vk, exs) in split_by(A, examples):
                subtree = decision_tree_learning(
                    exs, remove_all(A, attrs), examples)
                tree.add(vk, subtree)
            return tree

    def plurality(examples):
        """Return the most occured target value for this set of examples.
        (If binary target this is the majority, otherwise plurality)"""
        most_occured = argmax_random_tie(values[target],
                                         key=lambda v: count_example_same_attr(target, v, examples))
        return Decision_Leaf(most_occured)

    def count_example_same_attr(attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def same_class_for_all(examples):
        """Are all these examples in the same target class?"""
        _class_ = examples[0][target]
        return all(example[target] == _class_ for example in examples)

    def choose_important_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs,
                                 key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""

        def _entropy_(examples):
            count = []
            for val in values[target]:
                count.append(count_example_same_attr(target, val, examples))
            return Entropy(count)

        N = len(examples)
        remainder = sum((len(examples_i) / N) * _entropy_(examples_i)
                        for (v, examples_i) in split_by(attr, examples))
        return _entropy_(examples) - remainder

    def split_by(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v])
                for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.input_attributes)


def Random_Forest(dataset, n=5, verbose=False):
    """
    An ensemble of Decision Trees trained using bagging and feature bagging.

    bagging: Bootstrap aggregating
    """

    def data_bagging(dataset, m=0):
        """Sample m examples with replacement"""
        n = len(dataset.examples)
        return weighted_sample_with_replacement(m or n, dataset.examples, [1] * n)

    def feature_bagging(dataset, p=0.7):
        """Feature bagging with probability p to retain an attribute"""
        inputs = [i for i in dataset.input_attributes if probability(p)]
        return inputs or dataset.input_attributes

    def predict(example):
        if verbose:
            print([predictor(example) for predictor in predictors])
        return mode(predictor(example) for predictor in predictors)

    predictors = [Decision_Tree_Learner(Data_Set(examples=data_bagging(dataset),
                                                 attributes=dataset.attributes,
                                                 attribute_names=dataset.attribute_names,
                                                 target_attribute=dataset.target_attribute,
                                                 input_attributes=feature_bagging(dataset))) for _ in range(n)]

    return predict


def exploreDataset(filename):
    colnames = ['age' , 'workclass' , 'finalweight','education', 'eduYrs','maritalstatus','occupation',
                'relationship','race','sex','capitalgain','captialloss','hoursperweek','nationality','incomerange']
    data = pd.read_csv(filename, sep=',', header=None)
    data.columns = colnames

    return data.drop_duplicates()




if __name__ == '__main__':
    exploreDataset("/Users/sunitakoppar/PycharmProjects/datasets/adult/adult.data")