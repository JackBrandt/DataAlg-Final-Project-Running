"""
#=====================================================================================
#Jack Brandt
#Course: CPSC 322
#Assignment: PA6
#Date of current version: 10/15/2024
#Did you attempt the bonus? Yes
#Brief description of what program does:
#    Provides functions to assist with mysklearn.
#=====================================================================================
"""
import numpy as np # use numpy's random number generation
from mysklearn.mypytable import MyPyTable

def discretizer(val):
    """Discretizer function

    Args:
        val (float): The value you want discretized

    Returns:
        string: Either high or low
    """
    if val >= 100:
        return 'high'
    return 'low'

def discretizer_classifier(cutoff_values, classes, instance):
    """Discretizer function, turning continous value into classification.

    Args:
        cutoff_values (list of floats): The cutoff points with inclusive
                                        maxes for each class
        classes (list of strings): The parallel list of classes for each range
        instance (float): The value you want classified

    Returns:
        string: The class
    """
    for i,cutoff in enumerate(cutoff_values):
        if instance <= cutoff:
            return classes[i]
    return '-1'

def doe_discritizer(value):
    '''Makes a discretizer function for doe mpg class

    Args:
        value (float): The value to be discretized

    Returns:
        function: A discretizer from mpg to doe mpg class
    '''
    doe_mpg_cutoffs = [13,14,16,19,23,26,30,36,44,100]
    doe_mpg_classes = ['1','2','3','4','5','6','7','8','9','10']
    return discretizer_classifier(doe_mpg_cutoffs,doe_mpg_classes, value)

def compute_euclidean_distance(v1, v2):
    """Computes euclidean distance of

    Args:
        v1 (list of floats): coordinates for first point
        v2 (list of floats): coordinates for second point

    Returns:
        float: the euclidean distance"""

    # Otherwise
    square_pair_differences = [(x-y)**2 if not isinstance(x,str) else 0 if x==y else 1 for x,y in zip(v1,v2)]
    return (sum(square_pair_differences))**.5

def load_and_split(file_path, n, seed):
    """Loads data from a file, then splits it into train and test sets. Uses
     MyPyTable

     Args:
        file_path (string): The file containing your csv
        n (int): The number of instances you want in your test set
        seed (int): Random number generator seed

    Returns:
        MyPyTable: Your training set
        MyPyTable: Your test set
    """
    np.random.seed(seed)

    # Load data into table
    train_set = MyPyTable().load_from_file(file_path)
    #print(train_set.get_shape())

    # Select 5 random instances for test set
    test_set = MyPyTable(column_names=train_set.column_names)
    for _ in range(n):
        # Random index of element to move into test
        i=np.random.randint(0,train_set.get_shape()[0])
        test_set.data.append(train_set.data[i])
        # Calls drop to remove from data table
        train_set.drop_rows([i])

    return train_set, test_set

def listinator(lis):
    '''Places each element of a list inside of a list inside of a list:
    [a,b,...]=>[[a],[b],[c],...]

    Args:
    s (list): Your list

    Returns:
    list of lists'''
    listor = []
    for element in lis:
        listor.append([element])
    return listor #This is dumb

def delistininator(lis):
    '''List made of list of single elements=>list, opposite of listinator
    Args:
    lis (list of lists): your list

    Returns:
    lists
    '''
    return [x[0] for x in lis]

def classifier_accuracy(predictions, actuals):
    '''Calculates accuracy of classifier.
    Args:
        predictions (list of strings): The predicted classes
        actuals (list of strings): The actual classes

    Returns:
        float: The accuracy
    '''
    counter = 0
    for i, prediction in enumerate(predictions):
        if prediction == actuals[i]:
            counter+=1
    return counter/len(predictions)

def randomize_in_place(alist, parallel_list=None, seed=0):
    '''Shuffles the order of a list in place (i.e., no return)
    Args:
        alist (list): The main list to be shuffled
        parallel_list (list): Optional second list to be shuffled in parallel
            order.
        seed:
    '''
    np.random.seed(seed)
    for i,_ in enumerate(alist):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0,len(alist)) # rand int in [o,len(alist)]
        #do the swawp
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = (
                parallel_list[rand_index], parallel_list[i])

def pretty_step_print(step_num, step_title, method_string, knn_results,
                       dum_results):
    '''Prints the results of each step in a pretty way
    Args:
        step_num (int): The step number
        step_title (string): The step title
        method_string (string): Information about what evaluation method
        kNN_results (any): Results of this step for kNN
        dum_results (any): Results of this step for dummy classifier
    '''
    print('===========================================')
    print(f'STEP {step_num}: ' + step_title)
    print('===========================================')
    print(method_string)
    try:
        print(knn_results.__class__.__name__)
    except TypeError:
        pass
    if knn_results.__class__.__name__ == 'MyPyTable':
        print('k Nearest Neighbors Classifier: ')
        knn_results.pretty_print()
        print('Dummy Classifier: ')
        dum_results.pretty_print()

    else:
        print(f'k Nearest Neighbors Classifier: {knn_results}')
        print(f'Dummy Classifier: {dum_results}')

def get_class_labels(data):
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

def report_metrics_and_confusion(model_label, metrics, confusion,
        clas_repor):
    '''Prints the accuracy, error rate, precision, recall, F1 and confusion
    matrix for classification results

    Args:
        model_label (str): The name of the model used
        metrics (list of floats): [accuracy,precision,recall,f1]
        conf_matrix (MyPyTable): The model's confusion matrix
    '''
    accuracy,precision,recall,f1=metrics
    print('===========================================')
    print(f'{model_label} Summary Results')
    print('===========================================')
    print('1.')
    print(f'    Accuracy: {accuracy}')
    print(f'    Error Rate: {1-accuracy}')
    print('2.')
    print(f'    Precision: {precision}')
    print(f'    Recall: {recall}')
    print(f'    F1 measure: {f1}')
    print('3. Confusion Matrix:')
    confusion.pretty_print()
    print('(Bonus) Classification Report:')
    print(clas_repor)

def all_same_class(instances):
    '''Checks if all instances have the same class (where
    class is the last element in each list)
    Args:
        instances (list of lists):All instances with class as
            last element in each instance
    Returns:
        boolean: true if all same class, else false'''
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    # get here, then all same class labels
    return True

def compute_bootstrapped_sample(table, seed=None):
    '''Computes a bootstrapped sample from a table
    Args:
        table (list of lists): The table you are sampling
        seed (int): Optional parameter to seed np.random
    Returns:
        sample (list of lists): This is everything that was picked by sampling
        out_of_bag_sample (list of lists): This is everything that wasn't picked
    '''
    if seed is not None:
        np.random.seed(seed)
    n = len(table)
    # np.random.randint(low, high) returns random integers from low (inclusive) to high (exclusive)
    sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
    sample = [table[index] for index in sampled_indexes]
    out_of_bag_indexes = [index for index in list(range(n)) if index not in sampled_indexes]
    out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
    return sample, out_of_bag_sample
