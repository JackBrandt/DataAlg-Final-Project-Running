"""
#=====================================================================================
#Jack Brandt TODO: Go through and change names in all mysklearn to include
#               Suyash and change assignments and dates according for this project
#Course: CPSC 322
#Assignment: PA7
#Date of current version: 11/0?/2024
#Did you attempt the bonus? Yes
#Brief description of what program does:
#    Implements all of my classifiers, now including decision tree stuff
#=====================================================================================
"""
import operator
import math
import os
import time
import numpy as np
from mysklearn.myutils import compute_euclidean_distance, classifier_accuracy,\
    discretizer_classifier, compute_bootstrapped_sample, compute_random_subset
import mysklearn.myutils as mu
from mysklearn.myevaluation import stratified_kfold_split, train_test_split
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor


def get_normalizing_function(list_min, list_max):
    '''Create a function for normalizing a list of data
    Args:
        list_min (float): min
        list_max (float): max

    Returns:
        function: A function that takes a value returns it normalized
    '''
    def normalizing_function(value):
        return (value-list_min)/(list_max-list_min)
    return normalizing_function

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train,y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions=[]
        for x in X_test:
            y=self.regressor.slope * x[0] + self.regressor.intercept
            prediction=self.discretizer(y)
            predictions.append(prediction)

        return predictions

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        dists = []
        indices =[]

        for test in X_test:
            dists_and_indices = []
            for i,row in enumerate(self.X_train):
                dists_and_indices.append(
                    (compute_euclidean_distance(row,test),i))

            dists_and_indices.sort(key=operator.itemgetter(0))
            dists.append(
                [dists_and_indices[x][0] for x in range(self.n_neighbors)])
            indices.append(
                [dists_and_indices[x][1] for x in range(self.n_neighbors)])

        return dists, indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        _, kneighbors_indices = self.kneighbors(X_test)
        # For each X_test
        for i,_ in enumerate(X_test):
            classes = []
            class_counts = []
            # Count how many times the classes appear
            for neighbor_index in kneighbors_indices[i]:
                neighbor_class=self.y_train[neighbor_index]
                # If first time of class
                if neighbor_class not in classes:
                    classes.append(neighbor_class)
                    class_counts.append(1)
                # Subsequent time of class
                else:
                    class_index = classes.index(neighbor_class)
                    class_counts[class_index]+=1
            # Predict with most frequent
            prediction = classes[class_counts.index(max(class_counts))]
            #print(prediction)
            predictions.append(prediction)
        return predictions



class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
        strategy (str): Either most frequent or stratified

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self, strategy='most frequent'):
        """Initializer for DummyClassifier.

        """
        if strategy not in('most frequent', 'stratified'):
            raise ValueError('Strategy must be most frequent or stratified')
        self.most_common_label = None
        self.strategy = strategy

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        classes = []
        class_counts = X_train #This is dumb, its only purpose is to supress
        # pylint message... The better way would be to just delete X_train
        class_counts = []
        # Count how many times the classes appear
        for instance in y_train:
            # If first time of class
            if instance not in classes:
                classes.append(instance)
                class_counts.append(1)
            # Subsequent time of class
            else:
                instance_index = classes.index(instance)
                class_counts[instance_index]+=1
        if self.strategy == 'most frequent':
            # set most common label
            self.most_common_label = classes[
                class_counts.index(max(class_counts))]
        else:
            # set most common label to be a list of labels and their probability
            total = sum(class_counts)
            self.most_common_label=[(clas, class_counts[i]/total)
                                    for i, clas in enumerate(classes)]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        if self.strategy == 'most frequent':
            for _ in X_test:
                predictions.append(self.most_common_label)
        else:
            cutoffs = []
            prob_sum = 0
            for probability in self.most_common_label:
                prob_sum+=probability[1]
                cutoffs.append(prob_sum)
            for _ in X_test:
                classes = [label[0] for label in self.most_common_label]
                #print(cutoffs)
                #print(classes)
                predictions.append(
                    discretizer_classifier(cutoffs,classes,np.random.random()))
        return predictions

def classifier_results(predictions, test_set, discretizer):
    '''Prints classifier results including each test instance, the predicted
    value, and the actual value. At the end print classifier accuracy for
    the instances.

    Args:
        predictions (list of strings): The classes the classifier predicted
        test_set (list of lists): The data you tested on
        discretizer (function): The function that discretizes values into
            classes

    Returns:
        none
    '''

    # Find acutuals
    actuals = [discretizer(test_set.get_column('mpg')[x])
                for x,_ in enumerate(test_set.data)]

    # Print results
    for i, prediction in enumerate(predictions):
        print(f'Instance:{test_set.data[i]}')
        print(f'Predicted class: {prediction}    Actual class: {actuals[i]}')
    print(f'Accuracy: {classifier_accuracy(predictions,actuals)}')



class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict): The prior probabilities computed for each
            label in the training set.
        posteriors(dict of dict of dict): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # First calculate priors
        # This is the occurance of each class / total instances in y_train
        # Stored as a dict?
        self.priors = {}
        class_labels = mu.get_class_labels(y_train)
        class_frequency = [0]*len(class_labels)
        for instance in y_train: # This loop counts occurancces of each class
            index = class_labels.index(instance)
            class_frequency[index] += 1
        for i,label in enumerate(class_labels):
            self.priors[label]=class_frequency[i]/len(y_train)

        # Second calculate posteriors
        self.posteriors={}
        # Go one attribute at a time
        for attr_index,_ in enumerate(X_train[0]):
            #Create a dict for current attr
            cur_attr_label='att'+str(attr_index+1)
            self.posteriors[cur_attr_label]={}
            X_train_current_attribute = [x[attr_index] for x in X_train]
            current_value_labels = mu.get_class_labels(X_train_current_attribute)
            #Create a dict for each value
            for value in current_value_labels:
                self.posteriors[cur_attr_label][value]={}
                #Create a value for each class label
                for clas_label in class_labels:
                    self.posteriors[cur_attr_label][value][clas_label]=0
            # Now count values/classes for the attribute for all instances
            for i,instance_attr_value in enumerate([x[attr_index] for x in X_train]):
                self.posteriors[cur_attr_label][instance_attr_value][y_train[i]]=(
                    self.posteriors[cur_attr_label][instance_attr_value][y_train[i]]+1)
            # Now normalize to fraction
            for attr_value in current_value_labels:
                for class_label in class_labels:
                    self.posteriors[cur_attr_label][attr_value][class_label]=(
                        self.posteriors[cur_attr_label][attr_value][class_label]/(self.priors[class_label]*len(X_train)))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions=[]
        #Make prediction for each instance in X_test
        for instance in X_test:
            class_probs={}
            #Get probability for each class label
            for clas in self.priors:
                class_probs[clas]=1
                #Times conditional probability for each attribute
                for i,value in enumerate(instance):
                    class_probs[clas]=(
                        class_probs[clas]*self.posteriors['att'+str(i+1)][value][clas])
                #And prior
                class_probs[clas]=class_probs[clas]*self.priors[clas]
            #Make prediction based on greater probability
            max_of_probs=0
            class_of_max=[]
            for clas,value in class_probs.items():
                if value>max_of_probs:
                    max_of_probs=value
                    class_of_max=[clas]
                elif value==max_of_probs:
                    class_of_max.append(clas)
            if len(class_of_max)==1:
                #print(class_of_max)
                predictions.append(class_of_max[0])
            else:
                max_prior_value=0
                max_prior=''
                for clas in class_of_max:
                    if self.priors[clas]>max_prior_value:
                        max_prior=clas
                predictions.append(max_prior)
        return predictions

def tdidt_make_header(X_train):
    '''Makes a generic header based on length of instances, e.g. [\'att0\',\'att1\',...]
    Args:
        X_train (list of list): The data being fit on
    Returns:
        header (list of strings): Generic header
    '''
    header=[]
    for i,_ in enumerate(X_train[0]):
        header.append('att'+str(i))
    return header

def tdidt_get_class_labels(data):
    '''Takes in a list of class data, e.g., y_train, then returns
    the set of all possible class labels

    Args:
        data (list of strings): list of some combination of repeating strings
    returns:
        list of strings: The unique strings in the lists
    '''
    class_labels=[]
    for instance in data:
        if instance not in class_labels:
            class_labels.append(instance)
    return class_labels

def tdidt_get_attribute_domains(X_train):
    '''Get's domain of the attributes and returns in dictionary
    Args:
        X_train (list of list): The data being fit on
    Returns:
        attribute_domains (dictionary): The attribute domains
    '''
    header=tdidt_make_header(X_train)
    attribute_domains={}
    for i,att in enumerate(header):
        attribute_domains[att]=tdidt_get_class_labels([x[i] for x in X_train])
    return attribute_domains

def calculate_entropy_for_value_partition(partition):
    '''Calculates entropy for a value of the the value partion.
    Args:
        partition (list of lists): Something like a slice of a slice of X_train all with
            the same value for whatever attribute is being partitioned
    Returns:
        entropy (float): Entropy value, between 0 and 1
    Note:
        Entropy=-Sum_(i=1)^n p_i * log_2 (p_i)
    '''
    partition_length = len(partition)
    class_labels=[]
    class_counts = []
    for instance in partition:
        if instance[-1] not in class_labels:
            class_labels.append(instance[-1])
            class_counts.append(1)
        else:
            index = class_labels.index(instance[-1])
            class_counts[index]+=1
    class_proportions=[count/partition_length for count in class_counts]
    entropy = -sum(proportion*math.log2(proportion) for proportion in class_proportions)
    return entropy

def partition_instances(header, attribute_domains, instances, attribute):
    '''Partitions an instance based off an attribute. Basically a group by function
    Args:
        header (list): the generic list of attributes
        attribute_domains (dict): A dictionary of all possible values for each attribute
        instances (List of lists): The instances to be partitioned
        attribute (string): The attribute to parrtitioned on
    Returns:
        partitions (dictionary): The partitioned instances in the form of a
            dictionary where each possible attribute is a key, and the instances
            for that attribute are the corresponding part.
    '''
    # This is group by attribtue domain (not values of attribute in instances)
    # lets use dictionaries
    att_index = header.index(attribute)
    att_domain = attribute_domains[attribute]
    partitions = {}
    for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def select_attribute(header,attribute_domains, instances, attributes):
    '''Selects the best attribute to partition on based off entropy/info gain
    Args:
        header (list of string): The generic list of all attributes
        attribute_domains (dict): The possible values for each attribute
        instances (list of list): The set of instances
        attributes (list of string): The possible attributes to split on'''
    attribute_enews=[0]*len(attributes)
    # for each available attribute
    #     for each value in the attribute's domeain
    #          calculate the entropy for the value's partition
    #     calcculate the weighted average for the partition entropies
    # select the attribute with the smallest Enew entropy
    for i,att in enumerate(attributes):
        att_partition=partition_instances(header,attribute_domains,instances,att)
        for value in attribute_domains[att]:
            # Multiplied to set up for weighted average
            attribute_enews[i]+=calculate_entropy_for_value_partition(att_partition[value])*len(att_partition[value])
        attribute_enews[i]/=sum(len(att_partition[value]) for value in attribute_domains[att])#calculate_weighted_average_for_partition_entropies
    #print(attributes)
    #print(attribute_enews)
    att_with_smallest_enew = attributes[attribute_enews.index(min(attribute_enews))] # Get min

    # for now , select an attribute randomly
    # rand_index = np.random.randint(0, len(attributes))

    return att_with_smallest_enew

def partition_majority_vote(partition):
    '''Get most common label in the partition
    Args:
        att_partition (list): The partion
    Returns:
        label (any): The most common label'''
    #print('partition: ',partition)
    labels=[]
    label_counts=[]
    for _,item in partition.items():
        for instance in item:
            label=instance[-1]
            if label not in labels:
                labels.append(label)
                label_counts.append(1)
            else:
                label_index = labels.index(label)
                label_counts[label_index]+=1

    max_count = max(label_counts)
    majorities = []

    while True:
        try:
            majorities.append(labels.pop(label_counts.index(max_count)))
            label_counts.remove(max_count)

        except ValueError:
            break

    majorities.sort()
    #print('majorities:',majorities)
    majority=majorities[0]
    #print('majority',majority)
    return majority

def att_partition_majority_vote(att_partition):
    '''Get most common label in the aatt partition
    Args:
        att_partition (list): The active partion
    Returns:
        label (any): The most common label'''
    labels=[]
    label_counts=[]
    for instance in att_partition:
        label=instance[-1]
        if label not in labels:
            labels.append(label)
            label_counts.append(1)
        else:
            label_index = labels.index(label)
            label_counts[label_index]+=1

    max_count = max(label_counts)
    majorities = []

    while True:
        try:
            majorities.append(labels.pop(label_counts.index(max_count)))
            label_counts.remove(max_count)

        except ValueError:
            break

    majorities.sort()
    #print('majorities:',majorities)
    majority=majorities[0]
    #print('majority',majority)
    return majority

def tdidt(current_instances, available_attributes, attribute_domains,header,len_previous=0):
    '''This makes the decision tree using tdidt/recursive approach
    Args:
        current_instances (list of lists): The current lists to work with
        available_attributes (list of strings): The available attributes to split on
        attribute_domains (dict): List of possible values for each attribute
        header (list of str): Generic header based on number of attributes
        len_previous (int): Length of previous current_instances, used for case 3
    Returns:
        tree (list of lists of...): The tree in the form of nested lists
    '''
    #print('available attributes:', available_attributes)
    # basic approach (uses recursion!!):
    # select an attribute to split on
    split_attribute = select_attribute(header, attribute_domains, current_instances, available_attributes)
    #print('splitting on:', split_attribute)
    available_attributes.remove(split_attribute) # can't split on this attribute again
    # in this subtree
    tree = ['Attribute', split_attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(header,attribute_domains,current_instances, split_attribute)
    #print('partitions:',partitions)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value in sorted(partitions.keys()):# process in alphanetical order
        att_partition = partitions[att_value]
        value_subtree = ["Value",att_value]
    #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition)>0 and mu.all_same_class(att_partition):
            #print('Case 1')
            value_subtree.append(['Leaf',att_partition[0][-1],len(att_partition),sum(len(x) for _,x in partitions.items())])
            #print('value subtree: ',value_subtree)
            tree.append(value_subtree)
            #Make leaf node
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif (len(att_partition)>0 and len(available_attributes) == 0):
            #print("Case 2")
            label = att_partition_majority_vote(att_partition)
            value_subtree.append(['Leaf',label,len(att_partition),len(current_instances)])
            tree.append(value_subtree)
            #handle majority vote
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            #print("Case 3")
            #change your mind about splitting
            #overwrite tree with majority vote leaf node
            label = partition_majority_vote(partitions)
            tree=['Leaf',label,sum(len(x) for _,x in partitions.items()),len_previous]
            #print('tree: ',tree)
            return tree
        else:
            # none of base cases were tru,e recurse!!
            subtree = tdidt(att_partition, available_attributes.copy(),attribute_domains,header,len(current_instances))
            value_subtree.append(subtree)
            tree.append(value_subtree)
    return tree

def tdidt_predict(tree,instance,header):
    '''Does individual predictions for tdidt
    Args:
        tree (nested lists): The decision tree
        instance (list): an individual X_test
        header (list of str): Generic header
    Returns
        prediction (any): The prediction for the given X_test instance
    '''
    #print(header)
    #print(instance)
    # base case: we are at a leaf node and can return the class prediction
    #print(tree[0])
    infor_type = tree[0] # "Leaf" or "Attribute"
    if infor_type == "Leaf":
        return tree[1] # class label

    # If we are heere, we are at an Attribute
    # we need to match the instance's value for this attribute
    # to the appropriate subtree
    att_index = header.index(tree[1])
    #print(tree[1])
    #print(att_index)
    for i in range(2,len(tree)):
        value_list = tree[i]
        #print(value_list)
        # do we have a match with instance for this attribute?
        if value_list[1] == instance[att_index]:
            return tdidt_predict(value_list[2],instance,header)

    # If we are here, there is a error
    return None

def parse_rule_from_tree(tree, rule_string, class_name,attribute_names,dot_string=False,differentiator=''):
    '''Recursively nagivates tree to collect the rules and print them
    Args:
        tree (nested lists): the tdidt tree to gather rules from
        rule_string (str): The inprogress rule
        class_name (str): The name of the class
        attribute_names (list of str): The list of specific names for each attribute
        dot_string (bool): Whether code for a .dot should be generated (true), or just print rules (false)
        differentiator (str): Used to differentiate same attributes/values for tree viz
    Returns:
        rule_string (str): Only if dot_string == True, returns a string formattted to be inserted into a dot
            file such that it generates a tree
    '''
    match tree[0]:
        # Case 1: Attribute
        case 'Attribute':
            attribute = str(tree[1])
            if attribute_names is not None:
                attribute = tdidt_attribute_degeneralization(attribute,attribute_names)
            if dot_string:
                #print('\t'+attribute+differentiator+'[label=rule_number, shape=box];\n')
                #Add new node
                rule_string+='\t'+attribute+differentiator+f'[label={attribute}, shape=box];\n'
            else:
                rule_string+=' AND ' + attribute + ' == '
            #print(rule_string)
            for value in tree[2:]:
                if dot_string:
                    rule_string += parse_rule_from_tree(value,'',class_name,attribute_names,dot_string,attribute+differentiator)
                else:
                    parse_rule_from_tree(value,rule_string,class_name,attribute_names,dot_string,differentiator)
            # recurse/loop
            return rule_string
        # Case 2: Value
        case 'Value':
            #print(rule_string)
            if dot_string:
                #print(str(tree[1])+str(differentiator)+f'[label={str(tree[1])}, shape=box];\n')
                # recurse to get next node
                next_node = parse_rule_from_tree(tree[2],'',class_name,attribute_names,dot_string,str(tree[1])+differentiator)
                rule_string += next_node
                # add connection
                previous_node = differentiator
                next_node = str(next_node)[:next_node.index('[')]
                rule_string+='\t'+previous_node+' -- '+next_node+f'[label={str(tree[1])}];\n'
            else:
                rule_string+=str(tree[1])
                # recurse
                parse_rule_from_tree(tree[2],rule_string,class_name,attribute_names,dot_string,differentiator)
            return rule_string
        # Case 3: Leaf
        case 'Leaf':
            if dot_string:
                #print(str(tree[1])+differentiator+f'[label={str(tree[1])}, shape=oval];\n')
                rule_string+='\t'+str(tree[1])+differentiator+f'[label={str(tree[1])}, shape=oval];\n'
            else:
                rule_string+=' THEN ' + class_name + ' = ' + tree[1]
                print(rule_string)
            return rule_string
        case _:
            print("SOMETHING WENT WRONG PARSING RULES FROM TREE")

def tdidt_attribute_degeneralization(generic, header):
    '''Takes a generic attribute name, e.g. att0, and converts it into a meaningful name
    Args:
        generic (st): The generic name like att0
        header (list of str): The ordered list of specific attribute names
    Returns:
        str: The specific attribute name
    '''
    attribute_index = int(generic[3:])
    attribute = header[attribute_index]
    return attribute

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        # Make header and attribute domains
        header=tdidt_make_header(X_train)
        attribute_domains=tdidt_get_attribute_domains(X_train)
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        self.tree=tdidt(train,header.copy(), attribute_domains,header)#Two copies of header bc one will change recursively, one won't
        #print(self.tree)



    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predict=[]
        for test in X_test: # Do individual predictions
            y_predict.append(tdidt_predict(self.tree,test, tdidt_make_header(self.X_train)))
        return y_predict

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        attribute=self.tree[1]
        if attribute_names is not None:
            attribute = tdidt_attribute_degeneralization(attribute,attribute_names)
        rule_string='IF '+ attribute + ' == '
        for value in self.tree[2:]:
            parse_rule_from_tree(value, rule_string,class_name, attribute_names)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        # Generate a .dot file the looks something like the following
        #graph g {
        #   level[shape=box];
        #   phd[shape=box];
        #   // add an edge
        #   level -- phd[label="Junior"];
        #
        #   true1[label="True"];
        #   false1[label="False"];
        #   true2[label="True"];
        #   // false2[label="False"];
        #
        #   phd -- true1[label="no"];
        #   phd -- false1[label="yes"];
        #
        #   level -- true2[label="Mid"]
        #}

        # Write to .dot file
        dot_fname='tree_viz/' + dot_fname
        with open(dot_fname, 'w', encoding="utf-8") as dot_file:
            dot_file.write('graph g {\n')
            tree_rules = ''
            # Magic happens here, I imagine using decision rule function is helpful?
            rules=parse_rule_from_tree(self.tree,tree_rules,'class',attribute_names=attribute_names,dot_string=True)
            #print('the rule: ',rules)
            dot_file.writelines(rules)
            dot_file.write('}')

        # Finish with a terminal command to run the .dot file to make pdf
        os.system(f'dot -Tpdf -o tree_viz/{pdf_fname} {dot_fname}')


class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        N (int): number of trees to generate
        M (int): number of best trees chosen, i.e., generated N trees but final
                algorithm only uses the M best of them
        F (int): Size of each tree's available attributes
        trees (list of lists): A list of the form [[MyDecisionTreeClassifier, [the subattributes that formed the tree]],[MyDec..., [the subattributes that formed the tree]],...]
        TODO: idk expand this list as needed.

    Notes:
        Loosely based on prof's notes:
            https://github.com/GonzagaCPSC322/U6-Ensemble-Learning
        * Test-driven development time! *
        N, M, and F, are all parameters that need to be tuned by trial and error
        but because of random, each setting will be need to be tried multiple times
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.N = None
        self.M = None
        self.F = None
        self.trees = None

    def fit(self, X_train, y_train, N, M, F, seed=None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            N (int): number of trees to generate
            M (int): number of best trees chosen, i.e., generated N trees but final
                algorithm only uses the M best of them
            F (int): Size of each tree's available attributes

        Notes:
            Stole some functions from:
                https://github.com/GonzagaCPSC322/U6-Ensemble-Learning/blob/master/A%20Ensemble%20Learning.ipynb
        """
        # Step 0: set object variables
        self.X_train = X_train
        self.y_train = y_train
        self.N = N
        self.M = M
        self.F = F
        # Set seed
        if seed is None: # Guarentees randomness to the second
            seed=round(time.time())
        np.random.seed(seed)

        # Step 1: Generate a *random stratified* test set consisting of one
            # third of the original data set, with the remaining two thirds of the
            # instances forming the 'remainder set'.
        # In otherwords, do stratified kfold split with k=3, and simply choose one of the folds
        folds=stratified_kfold_split(X_train,y_train,3,seed,True)
        # Simply pick first fold
        fold=folds[0]
        # Convert back from indices to instances
        test_X=[X_train[x] for x in fold[1]]
        test_y=[y_train[y] for y in fold[1]]
        remainder_X=[X_train[x] for x in fold[0]]
        remainder_y=[y_train[x] for x in fold[0]]
        # ^--- Probably a good idea to verify this works like I think it does

        # Step 2: Generate N "random" decision trees using bootstrapping (giving a training and validation set)
            # over the remainder set. At each node, build your decision trees by randomly selecting F of the
            # remaining attributes as candidates to partition on. This is the standard random forest approach
            # discussed in class. Note that to build your decision trees you should still use entropy; however,
            # you are selecting from only a (randomly chosen) subset of the available attributes.
            # ^-- These should be parallel because of seeding
        for _ in range(N):
            # Step 2.1: The bootstrapping <- Also need to know this for testing
            b_seed = np.random.randint(0,1000000)
            #print(b_seed)
            sampled_x, unsampled_y = compute_bootstrapped_sample(remainder_X,b_seed)
            sampled_y, unsampled_y = compute_bootstrapped_sample(remainder_y,b_seed)
            # Step 2.2, choosing attributes and making decision trees
            # Randomly sample F attributes
            attribute_subset = compute_random_subset(list(range(len(sampled_x[0]))),F)
            attribute_subset.sort()
            print(attribute_subset)
            # ^-- As far as I can tell, getting to here is kinda what we need to know what trees to make to test against?
            # Make decision tree from sample
            TODO: NotImplementedError

        # Step 3: Select the M most accurate of the N decision trees using the corresponding validation sets.
            TODO: NotImplementedError

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        TODO: NotImplementedError
        # Write tests first!
