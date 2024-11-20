"""
#=====================================================================================
#Jack Brandt
#Course: CPSC 322
#Assignment: PA6
#Date of current version: 11/0?/2024
#Did you attempt the bonus? Yes
#Brief description of what program does:
#    Implements evaluations for classifiers, now including some for Bayes
#=====================================================================================
"""
import numpy as np # use numpy's random number generation
from mysklearn import myutils as mu
from mysklearn import mypytable as mp

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test
          set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5
              for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator
            for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your
                code
        shuffle(bool): whether or not to randomize the order of the instances
          before splitting
            Shuffle the rows in X and y before splitting and be sure to
             maintain the parallel order of X and y!!

    Returns:
        X_train (list of list of obj): The list of training samples
        X_test (list of list of obj): The list of testing samples
        y_train (list of obj): The list of target y values for training
            (parallel to X_train)
        y_test (list of obj): The list of target y values for testing
             (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # First shuffle data if applicable
    if shuffle:
        if random_state is None:
            random_state=np.random.randint(0, 1000)
        mu.randomize_in_place(X, y, random_state)

    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]

    # If test_size is an int, grab that for test number for test, otherwise
    # convert percentage to an int
    if test_size < 1:
        test_size=test_size*len(X)

    #print(f'test size = {test_size}')
    # Then grab the appropriate number
    for i, value in enumerate(X):
        if i<=len(X)-test_size-1:
            X_train.append(value)
            y_train.append(y[i])
        else:
            X_test.append(value)
            y_test.append(y[i])

    # Return splits
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator
            for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances
          before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is
        defined as a 2-item tuple
            The first item in the tuple is the list of training set indices
              for the fold
            The second item in the tuple is the list of testing set indices
              for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is
              the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3,
              3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    #Create list of indices
    indices = [index for index,_ in enumerate(X)]

    # First shuffle data if applicable
    if shuffle:
        if random_state is None:
            random_state=np.random.randint(0,1000)
        mu.randomize_in_place(indices, seed=random_state)

    #Make n groups to put indices into
    index_groups=[[] for _ in range(n_splits)]

    # Go through data in parallel, do mod n to determine which group to put
    # data indexes into
    #print(len(indices)/n_splits)
    for i,index in enumerate(indices):
        index_groups[i//-(-len(indices)//n_splits)].append(index) #Have to do weird negation
        #division because groups have to match exactly ): Don't blame me, I wanted to use mod

    #Then create the fold tuple list thing
    folds = []
    for i,index_grp in enumerate(index_groups):
        fold_train_indexes = []
        for j,group in enumerate(index_groups):
            if j != i:
                fold_train_indexes+=group
                #print(fold_train_indexes)
        folds.append((fold_train_indexes,index_grp))
    #print(folds)
    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator
          for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances
          before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is
          defined as a 2-item tuple
            The first item in the tuple is the list of training set indices
              for the fold
            The second item in the tuple is the list of testing set indices
              for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """

    #Create list of indices
    indices = [index for index,_ in enumerate(X)]

    # First shuffle data if applicable
    if shuffle:
        if random_state is None:
            random_state=np.random.randint(0,1000)
        mu.randomize_in_place(indices, seed=random_state)

    #Make n groups to put indices into
    index_groups=[[] for _ in range(n_splits)]

    #Group by class
    indices_by_class=[[indices[0]]]
    indices=indices[1:]
    for index in indices:
        existing_class=False
        for clas in indices_by_class:
            if y[index] == y[clas[0]]:
                clas.append(index)
                existing_class=True
                break
        if not existing_class:
            indices_by_class.append([index])
    #print(indices_by_class)


    # Go through data in parallel, do mod n to determine which group to put
    # data indexes into
    #print(len(indices)/n_splits)
    counter=0
    for clas in indices_by_class:
        while len(clas)>0:
            index_groups[counter].append(clas[0])
            clas = clas[1:]
            counter+=1
            counter=counter%n_splits
    #print(index_groups)

    #Then create the fold tuple list thing
    folds = []
    for i,index_grp in enumerate(index_groups):
        fold_train_indexes = []
        for j,group in enumerate(index_groups):
            if j != i:
                fold_train_indexes+=group
                #print(fold_train_indexes)
        folds.append((fold_train_indexes,index_grp))
    #print(folds)
    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    #set n_samples as necessary
    if n_samples is None:
        n_samples=len(X)

    #set random_state as necessary
    if random_state is not None:
        np.random.seed(random_state)

    #Prepare return variabels
    X_sample=[]
    X_out_of_bag=[]

    if y is None:
        y_sample=None
        y_out_of_bag=None
    else:
        y_sample=[]
        y_out_of_bag=[]

    been_drawn = [False]*len(X) # Keeps track of bag
    #Sample
    for _ in range(n_samples):
        sample_index = np.random.randint(0,len(X)-1)
        X_sample.append(X[sample_index])
        been_drawn[sample_index]=True
        if y is not None:
            y_sample.append(y[sample_index])

    # Place out of bag
    for i,bol in enumerate(been_drawn):
        if bol is False:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix=[] #Make matrix
    for _,_ in enumerate(labels):
        matrix.append([0]*len(labels))

    #Find indexes and count
    for i,y_t in enumerate(y_true):
        y_p = y_pred[i]
        true_i=labels.index(y_t)
        pred_i=labels.index(y_p)
        matrix[true_i][pred_i]+=1
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    pred_count=0
    cor_count=0
    #count stuff
    for i,y_t in enumerate(y_true):
        pred_count+=1
        if y_t==y_pred[i]:
            cor_count+=1
    #return normalized/not
    if normalize:
        return cor_count/pred_count
    return cor_count

def random_subsample_eval(model, k, X, y):
    '''Finds the accuracy of a model over k random subsamples

    Args:
        model (mysklearn classifier object): The model to be evaluated
        k (int): Number of random subsamples
        X (list of list): Independent data
        y (list): Dependent class data
    Returns:
        string: Summarizing accuracy and error rate
    '''
    accuracy_sum=0
    for _ in range(k):
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X,y)

        # Fit model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        #accuracy
        accuracy_sum += accuracy_score(y_test, y_pred)
    return f'accuracy = {accuracy_sum/k}, error rate = {1-accuracy_sum/k}'

def cross_val_predict(model, k, X, y, stratify=False):
    '''Finds the accuracy of a model using cross validation

     Args:
        model (mysklearn classifier object): The model to be evaluated
        k (int): Number of folds
        X (list of list): Independent data
        y (list): Dependent class data
        stratify (bool): Use stratification?

    Returns:
        string: Summarizing accuracy and error rate
    '''
    # Get folds
    if stratify:
        folds = stratified_kfold_split(X,y,k)
    else:
        folds = kfold_split(X, k)

    # Get accuracy over folds
    accuracy_count=0
    pred_count=0
    for fold in folds:
        pred_count+=len(fold[1])
        model.fit([X[i] for i in fold[0]], [y[i] for i in fold[0]])
        y_pred = model.predict([X[i] for i in fold[1]])
        accuracy_count+=accuracy_score([y[i] for i in fold[1]], y_pred, False)

    return (f'accuracy = {accuracy_count/pred_count}, '+
             f'error rate = {1-accuracy_count/pred_count}')

def bootstrap_method(model, k, X, y):
    ''' Finds the accuracy of a model using bootstrap samples

    Args:
        model (mysklearn classifier object): The model to be evaluated
        k (int): Number of bootstrap_samples
        X (list of list): Independent data
        y (list): Dependent class data

    Returns:
        string: Summarizing accuracy and error rate
    '''
    weighted_accuracy_sum=0
    weight_sum=0
    # Loop k times
    for _ in range(k):
        # boot sample
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = bootstrap_sample(X,y)
        weight_sum+=len(y_out_of_bag)
        model.fit(X_sample,y_sample)
        y_pred = model.predict(X_out_of_bag)
        weighted_accuracy_sum+=len(y_out_of_bag)*accuracy_score(y_out_of_bag,y_pred)

    return (f'accuracy = {weighted_accuracy_sum/weight_sum}, '+
            f'error rate = {1-weighted_accuracy_sum/weight_sum}')

def cross_confusion_matrix_method(model, k, X, y, labels,
                                  data_label, stratify=False):
    '''Preps confusion matrix to be printed using cross validation

     Args:
        model (mysklearn classifier object): The model to be evaluated
        k (int): Number of folds
        X (list of list): Independent data
        y (list): Dependent class data
        labels (list): List of labels
        data_label (str): What data is the model being tested on
        stratify (bool): Use stratification?

    Returns:
        MyPyTable: The confusion matrix
    '''
    # Get folds
    if stratify:
        folds = stratified_kfold_split(X,y,k)
    else:
        folds = kfold_split(X, k)

    # Get accuracy over folds
    matrix=confusion_matrix([],[],labels)
    for fold in folds:
        model.fit([X[i] for i in fold[0]], [y[i] for i in fold[0]])
        y_pred=model.predict([X[i] for i in fold[1]])
        y_true=[y[i] for i in fold[1]]
        new_matrix_part=confusion_matrix(y_true,y_pred,labels)
        matrix=[[matrix[i][j] + element for j,
                 element in enumerate(row)] for i,
                 row in enumerate(new_matrix_part)]

    matrix=[[labels[i]]+row+[sum(row)]+[row[i]/sum(row)*100] if sum(row)>0 else
            [labels[i]]+row+[sum(row)]+[0] for i,row in enumerate(matrix)]
    # Store as MyPyTable for convience
    table = mp.MyPyTable([data_label]+ labels+['Total','Recognition (%)'],
                         matrix)
    return table

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio  true_positive /  (true_positive +  false_positive)
        where  true_positive is the number of true positives and  false_positive the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    #implement label/s
    if labels is None:
        labels=mu.get_class_labels(y_true)
    if pos_label is None:
        pos_label=labels[0]
    true_positive=0
    false_positive=0
    #count
    for i, pred in enumerate(y_pred):
        #We only care about positive values
        if pred == pos_label:
            #count  true_positive
            if pred==y_true[i]:
                true_positive+=1
            #count  false_positive
            else:
                false_positive+=1
    #return precision if applicable
    if  false_positive+ true_positive!=0:
        return  true_positive/ (true_positive+ false_positive)
    return 0.0

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio  true_positive /
      (true_positive + false_negative) where  true_positive is
        the number of true positives and false_negative the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    #implement label/s
    if labels is None:
        labels=mu.get_class_labels(y_true)
    if pos_label is None:
        pos_label=labels[0]
    #This is almost the same as precision, just change a few variables around
    true_positive=0
    false_negative=0
    #count
    for i, true in enumerate(y_true):
        #We only care about positive values
        if true == pos_label:
            #count  true_positive
            if true==y_pred[i]:
                true_positive+=1
            #count  false_positive
            else:
                false_negative+=1
    #return precision if applicable
    if false_negative+ true_positive!=0:
        return  true_positive/ (true_positive+false_negative)
    return 0.0

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true,y_pred,labels,pos_label)
    recall = binary_recall_score(y_true,y_pred,labels,pos_label)
    if precision + recall == 0:
        return 0
    return 2 *(precision * recall) / (precision + recall)

def get_trues_and_preds_from_folds(model, folds, X, y):
    '''Gets 2 lists from from a list of folds, the collection of actual y's
    and all predicted y's

    Args:
        model (mysklearn classifier): The model predicting stuff
        folds (list of 2-item tuples): The list of folds where each fold is
          defined as a 2-item tuple. The first item in the tuple is the list
            of training set indices for the fold. The second item in the tuple
              is the list of testing set indices for the fold
        X (list of lists): The independent data
        y (list): The dependent categoral data

    Returns:
        list: All actual y's
        list: All predicted y's, parallel to all actual y's
    '''
    all_trues = []
    all_preds = []
    for fold in folds:
        # Fit model
        X_train = [X[i] for i in fold[0]]
        y_train = [y[i] for i in fold[0]]
        model.fit(X_train,y_train)
        # Make predictions
        X_test = [X[i] for i in fold[1]]
        y_test = [y[i] for i in fold[1]]
        y_pred = model.predict(X_test)
        # Add to collection
        all_trues += y_test
        all_preds += y_pred
    return all_trues, all_preds


def get_metrics_and_conf_matrix_and_report(model,k,X,y,labels,pos_label,data_label):
    '''Uses stratified k-fold cross validation to calculate classifier
    performance metrics and confusion matrix

    Args:
        model (mysklearn classifier): The model to be evaluated
        k (int): The number of folds
        X (list of lists): The independent data
        y (list): The dependent categoral data
        labels (list of strings): The labels for each attribute
        pos_label (any): The label of a positive prediction
        data_label (string): Name of data model is being tested on


    Returns:
        list: [accuracy, precision, recall, f1]
        MyPyTable: The confusion matrix'''

    folds = stratified_kfold_split(X,y,k)
    all_trues, all_preds = get_trues_and_preds_from_folds(model,folds,X,y)
    metrics=[mu.classifier_accuracy(all_preds,all_trues),#This function bc we just want
             # only the accuracy,
             binary_precision_score(all_trues,all_preds,pos_label=pos_label),
             binary_recall_score(all_trues,all_preds,pos_label=pos_label),
             binary_f1_score(all_trues,all_preds,pos_label=pos_label)]
    return metrics, cross_confusion_matrix_method(model,k,X,y,labels,data_label,
            True), classification_report(all_trues,all_preds)

def classification_report(y_true,y_pred,labels=None,output_dict=False):
    """Build a text report and a dictionary showing the main classification metrics.

        Args:
            y_true(list of obj): The ground_truth target y values
                The shape of y is n_samples
            y_pred(list of obj): The predicted target y values (parallel to y_true)
                The shape of y is n_samples
            labels(list of obj): The list of possible class labels. If None, defaults to
                the unique values in y_true
            output_dict(bool): If True, return output as dict instead of a str

        Returns:
            report(str or dict): Text summary of the precision, recall, F1 score for each class.
                Dictionary returned if output_dict is True. Dictionary has the following structure:
                    {'label 1': {'precision':0.5,
                                'recall':1.0,
                                'f1-score':0.67,
                                'support':1},
                    'label 2': { ... },
                    ...
                    }
                The reported averages include macro average (averaging the unweighted mean per label) and
                weighted average (averaging the support-weighted mean per label).
                Micro average (averaging the total true positives, false negatives and false positives)
                multi-class with a subset of classes, because it corresponds to accuracy otherwise
                and would be the same for all metrics.

        Notes:
            Loosely based on sklearn's classification_report():
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        """
    # Get labels as necessary
    if labels is None:
        labels=mu.get_class_labels(y_true)
    possible_classes=mu.get_class_labels(y_true)
    while len(labels) > len(possible_classes):
        possible_classes.append('')

    # First step is probably to make a dict with each label's
    # precision, recall, f1, and support
    report_dict={}
    for i, clas in enumerate(possible_classes):
        report_dict[labels[i]] = {
            'precision': binary_precision_score(y_true,y_pred,labels,clas),
            'recall': binary_recall_score(y_true,y_pred,labels,clas),
            'f1-score': binary_f1_score(y_true,y_pred,labels,clas),
            'support': y_true.count(clas)
        }

    # Then, add the micro,macro, and weighted to that dict
    # Gotta keep in mind if there are two classes with at least 1 correct
    #precition,then micro gets replaced by accuracy

    # So, check for what I just said
    non_zero_cor_pred_classes=[]
    for label in labels:
        if report_dict[label]['precision'] != 0:
            non_zero_cor_pred_classes.append(label)
    # Then, if == 2 do just average # Oh man, I just realized this file is 700 lines
    if len(non_zero_cor_pred_classes) > 1:
        report_dict['accuracy']={
            'precision': '',
            'recall': '',
            'f1-score': accuracy_score(y_true,y_pred),
            'support':len(y_true)
        }
    # Else, do all the micros
    else:
        report_dict['micro avg']={
            'precision':report_dict[non_zero_cor_pred_classes[0]]['precision'],
            'recall':report_dict[non_zero_cor_pred_classes[0]]['recall'],
            'f1-score':report_dict[non_zero_cor_pred_classes[0]]['f1-score'],
            'support':len(y_true)
        }
    # Then do macros and weighted
    macro_precision = (sum(y['precision'] if x not in ['micro avg','accuracy'] else
                             0 for x,y in report_dict.items())/(len(report_dict)-1))
    macro_recall = (sum(y['recall'] if x not in ['micro avg','accuracy'] else
                             0 for x,y in report_dict.items())/(len(report_dict)-1))
    macro_f1 = (sum(y['f1-score'] if x not in ['micro avg','accuracy'] else
                             0 for x,y in report_dict.items())/(len(report_dict)-1))
    report_dict['macro avg']={
            'precision':macro_precision,
            'recall':macro_recall,
            'f1-score':macro_f1,
            'support':len(y_true)
        }
    weighted_precision = (sum(y['support']*y['precision']
                                if x not in ['micro avg','macro avg','accuracy'] else 0 for x,y in
                                  report_dict.items())/(len(y_true)))
    weighted_recall = (sum(y['support']*y['recall']
                                if x not in ['micro avg','macro avg','accuracy'] else 0 for x,y in
                                  report_dict.items())/(len(y_true)))
    weighted_f1 = (sum(y['support']*y['f1-score']
                                if x not in ['micro avg','macro avg','accuracy'] else 0 for x,y in
                                  report_dict.items())/(len(y_true)))
    report_dict['weighted avg']={
            'precision':weighted_precision,
            'recall':weighted_recall,
            'f1-score':weighted_f1,
            'support':len(y_true)
        }
    #Then either return dict or pretty string
    if output_dict:
        return report_dict
    table = mp.MyPyTable()
    table.fill_from_dict(report_dict)
    return table.pretty_string()
