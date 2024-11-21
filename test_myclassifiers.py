'''
#=====================================================================================
#Jack Brandt
#Course: CPSC 322
#Assignment: PA7
#Date of current version: 11/18/2024
#Did you attempt the bonus? Yes
#Brief description of what program does:
#    Contains the myclassifiers pytests
#=====================================================================================
'''

import numpy as np
from mysklearn.myclassifiers import MyNaiveBayesClassifier
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier, MyDummyClassifier, MyDecisionTreeClassifier,\
    MyRandomForestClassifier
from mysklearn.myutils import discretizer, compute_bootstrapped_sample,\
      get_column, get_columns
from mysklearn.myevaluation import stratified_kfold_split

# from in-class #1  (4 instances)
# This is the normalized data
X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]

# from in-class #2 (8 instances)
# I'm gonna assume normalized
X_train_class_example2 = [
    [3, 2],
    [6, 6],
    [4, 1],
    [4, 4],
    [1, 2],
    [2, 0],
    [0, 3],
    [1, 6]]

y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

# from Bramer
# I'm gonna assume this is normalized
header_bramer_example = ["Attribute 1", "Attribute 2"]
X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
"-", "-", "+", "+", "+", "-", "+"]

# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    '''Test for simple_linear_regression_classifier.fit()'''
    np.random.seed(0)
    X_train = [[val] for val in list(range(0,100))]
    y_train = [row[0] * 2 + np.random.normal(0,25) for row in X_train]

    # Create instance/fit to data
    lin_reg_classifier = MySimpleLinearRegressionClassifier(discretizer)
    lin_reg_classifier.fit(X_train, y_train)

    # Assert underlying regressor is correct
    assert np.isclose(lin_reg_classifier.regressor.slope, 1.9249174584304428)
    assert np.isclose(lin_reg_classifier.regressor.intercept, 5.211786196055158)

def test_simple_linear_regression_classifier_predict():
    '''Test for simple_linear_regression_classifier.predict()'''
    np.random.seed(0)
    # Test case 1
    X_train = [[val] for val in list(range(0,100))]
    y_train = [row[0] * 2 + np.random.normal(0,25) for row in X_train]

    # Create instance/fit to data
    lin_reg_classifier = MySimpleLinearRegressionClassifier(discretizer)
    lin_reg_classifier.fit(X_train, y_train)

    # Assert lin_reg_classifier matches desk calculation
    assert lin_reg_classifier.predict([[100],[10],[0]]) == ['high','low','low']

    # Test case 2
    X_train = [[val] for val in list(range(0,100))]
    y_train = [row[0] * 3 + np.random.normal(37,15) for row in X_train]

    # Fit to data
    lin_reg_classifier.fit(X_train, y_train)

    # Assert lin_reg_classifier matches desk calculation
    assert lin_reg_classifier.predict([[50],[10],[0]]) == ['high','low','low']

def test_kneighbors_classifier_kneighbors():
    """Test for kneighbors_classifier.kneighbors()"""
    kneigh_class = MyKNeighborsClassifier(3)

    # Assert test set 1 matches desk calculations
    kneigh_class.fit(X_train_class_example1, y_train_class_example1)
    dists, indices = kneigh_class.kneighbors([[3,7]])
    #dists match
    assert np.isclose(dists[0],[6.32455532,7.280109889,7.491922317]).all()
    #indices match
    assert indices[0] == [0,1,2]

    # Assert test set 2 matches desk calculations
    kneigh_class.fit(X_train_class_example2, y_train_class_example2)
    dists, indices = kneigh_class.kneighbors([[2,3]])
    #dists match
    assert np.isclose(dists[0],[1.414213562,1.414213562,2]).all()
    #indices match
    assert indices[0] == [0,4,6]

    # Assert test set 3 matches desk calculations
    kneigh_class.fit(X_train_bramer_example,y_train_bramer_example)
    dists, indices = kneigh_class.kneighbors([[6.9,4.2]])
    assert np.isclose(dists[0],[5.768882041,5.913543777,5.961543424]).all()
    assert indices[0] == [2,5,9]

def test_kneighbors_classifier_predict():
    """Test for kneighbors_classifier.predict()"""
    kneigh_class = MyKNeighborsClassifier(3)

    # Assert pred for set 1 matches desk calculations
    kneigh_class.fit(X_train_class_example1, y_train_class_example1)
    assert kneigh_class.predict([[3,7]])[0] == 'bad'

    # Assert prediction for set 2 matches desk calculations
    kneigh_class.fit(X_train_class_example2,y_train_class_example2)
    assert kneigh_class.predict([[2,3]])[0] == 'yes'

    # Assert prediction for set 2 matches desk calculations
    kneigh_class.fit(X_train_bramer_example,y_train_bramer_example)
    assert kneigh_class.predict([[6.9,4.2]])[0] == '+'

def test_dummy_classifier_fit():
    """Test for dummy_classifier.fit()"""
    np.random.seed(0)
    dum_class = MyDummyClassifier()

    # Test 1
    X_train=list(range(100))
    y_train=list(np.random.choice(["yes",'no'],100, replace=True, p=[0.7,0.3]))

    dum_class.fit(X_train,y_train)
    assert dum_class.most_common_label == 'yes'

    # Test 2
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True,
                                     p=[0.2, 0.6, 0.2]))

    dum_class.fit(X_train,y_train)
    assert dum_class.most_common_label == 'no'

    # Test 3
    y_train = list(np.random.choice(["bread", "soup", "mystery"], 100,
                                     replace=True,p=[0.6, 0.3, 0.1]))
    dum_class.fit(X_train,y_train)
    assert dum_class.most_common_label == 'bread' # I like sour dough (:

def test_dummy_classifier_predict():
    """Test for dummy_classifier.predict()"""
    np.random.seed(0)
    dum_class = MyDummyClassifier()

    # Test 1
    X_train=list(range(100))
    y_train=list(np.random.choice(["yes",'no'],100, replace=True, p=[0.7,0.3]))

    dum_class.fit(X_train,y_train)
    assert dum_class.predict([[101]])[0] == 'yes'

    # Test 2
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True,
                                     p=[0.2, 0.6, 0.2]))

    dum_class.fit(X_train,y_train)
    assert dum_class.predict([[101]])[0]  == 'no'

    # Test 3
    y_train = list(np.random.choice(["bread", "soup", "mystery"], 100,
                                     replace=True,p=[0.6, 0.3, 0.1]))
    dum_class.fit(X_train,y_train)
    assert dum_class.predict([[101]])[0] == 'bread' # bread (:

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Start of new tests for this assignment
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# in-class Naive Bayes example (lab task #1)
header_inclass_example = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# MA7 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain", "class"]
X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]
y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]

def test_naive_bayes_classifier_fit():
    ''' tests naive_bayes_classifier_fit() on the following test cases
    8 instance training set example traced from class, asserting against desk check of the priors and posteriors
    15 instance training set example from MA7, asserting against  desk check of the priors and posteriors
    Bramer 3.2 Figure 3.1 train dataset example, asserting against the priors and posteriors solution in Figure 3.2.
    '''
    bayes_classer = MyNaiveBayesClassifier()

    # Test on first training set
    bayes_classer.fit(X_train_inclass_example,y_train_inclass_example)
    assert bayes_classer.priors == {#Keeping the labels generic, bc I
        #think I remember Prof. telling us we didn't need labels?
        'yes' : 5/8, #class 1 represents yes
        'no' : 3/8 #class 2 represents no
    }
    assert bayes_classer.posteriors == {
        'att1' : {#att 1
            1: {
                'yes' : 4/5,
                'no' : 2/3
            },
            2 : {
                'yes' : 1/5,
                'no' : 1/3
            }
        },
        'att2' : { #att 2
            5 : {
                'yes' : 2/5,
                'no' : 2/3
            },
            6 : {
                'yes' : 3/5,
                'no' : 1/3
            }
        }
    }


    # Test on second training set
    bayes_classer.fit(X_train_iphone,y_train_iphone)
    assert bayes_classer.priors == {
        'yes' : 10/15,
        'no' : 5/15
    }
    assert bayes_classer.posteriors == {
        'att1' : {#standing
            1 : {
                'yes' : 2/10,
                'no' : 3/5
            },
            2 : {
                'yes' : 8/10,
                'no' : 2/5
            }
        },
        'att2' : {#job status
            1 : {
                'yes' : 3/10,
                'no' : 1/5
            },
            2 : {
                'yes' : 4/10,
                'no' : 2/5
            },
            3 : {
                'yes' : 3/10,
                'no' : 2/5
            }
        },
        'att3' : {#credit rating
            'fair' : {
                'yes' : 7/10,
                'no' : 2/5
            },
            'excellent' : {
                'yes' : 3/10,
                'no' : 3/5
            }
        }
    }

    #Test on third training set
    bayes_classer.fit(X_train_train,y_train_train)
    assert bayes_classer.priors == {
        'on time' : 14/20,
        'late' : 2/20,
        'very late' : 3/20,
        'cancelled' : 1/20
    }
    assert bayes_classer.posteriors == {
        'att1' : {#day
            'weekday' : {#weekday
                'on time' : 9/14,
                'late' : 1/2,
                'very late' : 3/3,
                'cancelled' : 0/1
            },
            'saturday' : {
                'on time' : 2/14,
                'late' : 1/2,
                'very late' : 0/3,
                'cancelled' : 1/1
            },
            'sunday' : {
                'on time' : 1/14,
                'late' : 0/2,
                'very late' : 0/3,
                'cancelled' : 0/1
            },
            'holiday' : {
                'on time' : 2/14,
                'late' : 0/2,
                'very late' : 0/3,
                'cancelled' : 0/1
            }
        },
        'att2' : {#season
            'spring' : {#spring
                'on time' : 4/14,
                'late' : 0/2,
                'very late' : 0/3,
                'cancelled' : 1/1
            },
            'summer' : {
                'on time' : 6/14,
                'late' : 0/2,
                'very late' : 0/3,
                'cancelled' : 0/1
            },
            'autumn' : {
                'on time' : 2/14,
                'late' : 0/2,
                'very late' : 1/3,
                'cancelled' : 0/1
            },
            'winter' : {
                'on time' : 2/14,
                'late' : 2/2,
                'very late' : 2/3,
                'cancelled' : 0/1
            }
        },
        'att3' : {#wind
            'none' : {#none
                'on time' : 5/14,
                'late' : 0/2,
                'very late' : 0/3,
                'cancelled' : 0/1
            },
            'high' : {
                'on time' : 4/14,
                'late' : 1/2,
                'very late' : 1/3,
                'cancelled' : 1/1
            },
            'normal' : {
                'on time' : 5/14,
                'late' : 1/2,
                'very late' : 2/3,
                'cancelled' : 0/1
            }
        },
        'att4' : {#rain
            'none' : {#none
                'on time' : 5/14,
                'late' : 1/2,
                'very late' : 1/3,
                'cancelled' : 0/1
            },
            'slight' : {
                'on time' : 8/14,
                'late' : 0/2,
                'very late' : 0/3,
                'cancelled' : 0/1
            },
            'heavy' : {
                'on time' : 1/14,
                'late' : 1/2,
                'very late' : 2/3,
                'cancelled' : 1/1
            }
        }
    }

def test_naive_bayes_classifier_predict():
    '''Tests naive_bayes_classifier_predict() against the following test cases
    a. 8 instance training set from class, asserting against desk check
    b. 15 instance training set from MA7, asserting against desk check
    c. Bramer's set specified unseen instances and slef-assessment exercise 1
    unseen instances asserting against solutions on pg.28-29 and appendix
    '''
    bayes_classer = MyNaiveBayesClassifier()

    #training set a
    bayes_classer.fit(X_train_inclass_example,y_train_inclass_example)
    predictions = bayes_classer.predict([[1,5]])
    assert predictions == ['yes']

    #training set b
    bayes_classer.fit(X_train_iphone,y_train_iphone)
    predictions = bayes_classer.predict([[2,2,'fair'],[1,1,'excellent']])
    assert predictions == ['yes','no']

    #training set c
    bayes_classer.fit(X_train_train,y_train_train)
    predictions = bayes_classer.predict([['weekday','winter','high','heavy'],
                                         ['weekday','summer','high','heavy'],
                                         ['sunday','summer','normal','slight']]
                                        )
    assert predictions == ['very late','on time','on time']

interview_X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

interview_y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

interview_header = ["att0", "att1", "att2", "att3"]
interview_attribute_domains = {"att0": ["Junior", "Mid", "Senior"],
        "att1": ["Java", "Python", "R"],
        "att2": ["no", "yes"],
        "att3": ["no", "yes"]}

interview_tree_solution =   ["Attribute", "att0",
                                ["Value", "Junior",
                                    ["Attribute", "att3",
                                        ["Value", "no",
                                            ["Leaf", "True", 3, 5]
                                        ],
                                        ["Value",'yes',
                                            ['Leaf','False',2,5]]
                                    ]
                                ],
                                ['Value','Mid',
                                    ['Leaf','True',4,14]
                                ],
                                ['Value','Senior',
                                    ['Attribute', 'att2',
                                        ['Value','no',
                                            ['Leaf', 'False', 3, 5]
                                        ],
                                        ['Value','yes',
                                            ['Leaf', 'True', 2, 5]
                                        ]
                                    ]
                                ]
                            ]

iphone_X_train = [
        [1,3,'fair'],
        [1,3,'excellent'],
        [2,3,'fair'],
        [2,2,'fair'],
        [2,1,'fair'],
        [2,1,'excellent'],
        [2,1,'excellent'],
        [1,2,'fair'],
        [1,1,'fair'],
        [2,2,'fair'],
        [1,2,'excellent'],
        [2,2,'excellent'],
        [2,3,'fair'],
        [2,2,'excellent'],
        [2,3,'fair']
    ]

iphone_y_train = ['no',
    'no',
    'yes',
    'yes',
    'yes',
    'no',
    'yes',
    'no',
    'yes',
    'yes',
    'yes',
    'yes',
    'yes',
    'no',
    'yes'
    ]

iphone_solution_tree =  ["Attribute", "att0",
                            ['Value',1,
                                ['Attribute', 'att1',
                                    ['Value', 1,
                                        ['Leaf','yes',1,5]
                                    ],
                                    ['Value', 2,
                                        ['Attribute', 'att2',
                                            ['Value','excellent',
                                                ['Leaf','yes',1,2]
                                            ],
                                            ['Value','fair',
                                                ['Leaf','no',1,2]
                                            ]
                                        ]
                                    ],
                                    ['Value', 3,
                                        ['Leaf','no',2,5]
                                    ]
                                ]
                            ],
                            ['Value', 2,
                                ['Attribute', 'att2',
                                    ['Value','excellent',
                                        ["Leaf",'no',4,10]
                                    ],
                                    ['Value','fair',
                                        ['Leaf','yes',6,10]
                                    ]
                                ]
                            ]
                        ]

def test_decision_tree_classifier_fit():
    '''Tests decision_tree_classifier_fit()'''
    test_tree=MyDecisionTreeClassifier()
    test_tree.fit(interview_X_train,interview_y_train)
    assert test_tree.tree == interview_tree_solution

    #print('assert1 for fit passed')

    test_tree.fit(iphone_X_train,iphone_y_train)
    assert test_tree.tree == iphone_solution_tree

instance1 = ['Junior','Java','yes','no'] # True
instance2 = ['Junior','Java','yes','yes'] # False
instance3 = ['Intern','Java','yes','yes'] # None

iphone_instance1 = [2,2,'fair']
iphone_instance2 = [1,1,'excellent']

def test_decision_tree_classifier_predict():
    '''Tests decision_tree_classifier_predict()'''
    # Create/Fit
    test_tree=MyDecisionTreeClassifier()
    test_tree.fit(interview_X_train,interview_y_train)
    # Test against 3 cases: True, False, None
    assert test_tree.predict([instance1]) == ['True']
    assert test_tree.predict([instance2]) == ['False']
    assert test_tree.predict([instance3]) == [None]
    #Other tree
    test_tree.fit(iphone_X_train,iphone_y_train)
    # Test against 2 cases:
    assert test_tree.predict([iphone_instance1]) == ['yes']
    assert test_tree.predict([iphone_instance2]) == ['yes']

# Generic data split for seed=0:
folds=stratified_kfold_split(interview_X_train,interview_y_train,3,0,True)
fold = folds[0]
remainder_X=[interview_X_train[x] for x in fold[0]]
remainder_y=[interview_y_train[x] for x in fold[0]]
interview_tree = MyDecisionTreeClassifier()
# Now get training X, for making the tree
training_X, _ = compute_bootstrapped_sample(remainder_X,0)
training_y, _ = compute_bootstrapped_sample(remainder_y,0)
interview_tree.fit(training_X, training_y) # Gotta fill in fits
forest_solution1=[[interview_tree.tree,[0,1,2,3]]]

# Make tree to test against based off bootstrap sample the .fit() uses
training_X, _ = compute_bootstrapped_sample(remainder_X,0) # The seed for these depends on what number tree that is generated is the best
training_y, _ = compute_bootstrapped_sample(remainder_y,0) # Seed y has to match the one on X
# Gotta reduce training X/y to just the selected attribute
reduced_training_X = get_column(1,training_X)
tree_1 = MyDecisionTreeClassifier()
tree_1.fit(reduced_training_X,training_y) # Gotta fill in fits

training_X, _ = compute_bootstrapped_sample(remainder_X,3)
training_y, _ = compute_bootstrapped_sample(remainder_y,3)
reduced_training_X = get_column(1, training_X)
tree_2 = MyDecisionTreeClassifier()
tree_2.fit(reduced_training_X,training_y)

training_X, _ = compute_bootstrapped_sample(remainder_X,4)
training_y, _ = compute_bootstrapped_sample(remainder_y,4)
reduced_training_X = get_column(2, training_X)
tree_3 = MyDecisionTreeClassifier()
tree_3.fit(reduced_training_X,training_y)

forest_solution2=[[tree_3.tree,[2]],
    [tree_2.tree,[1]],
    [tree_1.tree,[1]]]

# Make tree to test against based off bootstrap sample the .fit() uses
forest_solution3 = []
attributes = [[0, 1],
    [1, 3],
    [0, 1],
    [1, 3],
    [0, 2],
    [1, 2],
    [1, 2],
    [0, 1],
    [2, 3],
    [2, 3],
    [1, 2],
    [2, 3],
    [0, 2],
    [0, 1],
    [1, 2],
    [2, 3],
    [2, 3],
    [0, 3],
    [0, 2],
    [0, 2]]
for num in [4,9,10,5,6,12,15]:
    training_X, _ = compute_bootstrapped_sample(remainder_X,num)
    training_y, _ = compute_bootstrapped_sample(remainder_y,num)
    reduced_training_X = get_columns(attributes[num],training_X)
    this_tree = MyDecisionTreeClassifier()
    this_tree.fit(reduced_training_X,training_y)
    forest_solution3.append([this_tree.tree,attributes[num]])

# Apparently we'll be GRADED on using a test driven approach
# so we should do these first
def test_random_forest_classifier_fit():
    '''Test random_forest_classifier_fit()'''
    # Create
    test_forest = MyRandomForestClassifier()
    # We are encourage to use the interview dataset with
    # different parameters
    # Use seed=0 for all of these for consistency
    # Test 1: N=1,M=1,F=4 (Just 1 decision tree)
    test_forest.fit(interview_X_train,interview_y_train,1,1,4,seed=0)
    # Check to make sure parameters are properly set
        # What we really care about is test_forest.trees, but we'll check the others
        # the first time just in case
    assert test_forest.X_train == interview_X_train
    assert test_forest.y_train == interview_y_train
    assert test_forest.N == 1
    assert test_forest.M == 1
    assert test_forest.F == 4
    # The algorithm follows the following steps
    # 1. Generate stratified test set with one third for testing, and 2/3 remainder
        # This is more or less given to us, so it should be fine
        # By seeding with 0, this should be the same for all tests
    # 2. Generate N 'random' decision trees using bootstraping over the remainder set
        # The bootstraping is also given to us, and we already have a test for checking
        # decision trees
    # 3. Select the M most accurate of the N decision trees using validation sets
        # So, finally, check trees
    # This first test is against N=1, M=1, F=4, or in other words just the normal decision tree
        # Make tree to test against based off bootstrap sample the .fit() uses
    assert test_forest.trees == forest_solution1

    # Test 2: N=5, M=3, F=1 (Best 3/5 decision trees, each with 1 attribute)
    test_forest.fit(interview_X_train,interview_y_train,5,3,1,seed=0)
    #print('Forest solution 2: ', forest_solution2)
    assert test_forest.trees == forest_solution2

    # Test 3: N=20, M=7, F=2 (Best 7/20 decision trees, each with 2 attributes)
    test_forest.fit(interview_X_train,interview_y_train,20,7,2,seed=0)
    #print('Forest_solution 3:',forest_solution3)
    assert test_forest.trees == forest_solution3 # ???


def test_random_forest_classifier_predict():
    '''Tests random_forest_classifier_predict()'''
    # For this take the sets of trees for case 2 and 3 from the fit test
    # Get their predictions, take the majority vote of them
    # Compare that to what random_forest predicts
    # Should be much simpler then fit (:

    # Testing instance 1 for N=5, M=3, F=1
    test_instance_A1 = ["Mid", "R", "yes", "no"]
    test_forest=MyRandomForestClassifier()
    test_forest.fit(interview_X_train,interview_y_train,5,3,1,seed=0)
    solution_predictions = []
    solution_predictions.append(tree_1.predict([['R']]))
    #tree_1.print_decision_rules()
    solution_predictions.append(tree_2.predict([['R']]))
    #tree_3.print_decision_rules()
    solution_predictions.append(tree_3.predict([['no']]))
    majority_vote = MyDummyClassifier() # This is kinda silly, but hopefully it works
    majority_vote.fit([],solution_predictions)
    #print(solution_predictions)
    #print(majority_vote.most_common_label)
    solution_prediction_1 = majority_vote.most_common_label
    assert test_forest.predict([test_instance_A1]) == solution_prediction_1

    # Test instance 2 for N=5, M=3, F=1
    test_instance_A2 = ["Senior", "Python", "yes", "yes"]
    solution_predictions = []
    solution_predictions.append(tree_1.predict([['Python']]))
    #tree_1.print_decision_rules()
    solution_predictions.append(tree_2.predict([['Python']]))
    #tree_3.print_decision_rules()
    solution_predictions.append(tree_3.predict([['yes']]))
    majority_vote = MyDummyClassifier() # This is kinda silly, but hopefully it works
    majority_vote.fit([],solution_predictions)
    print(solution_predictions)
    print(majority_vote.most_common_label)
    solution_prediction_2 = majority_vote.most_common_label
    assert test_forest.predict([test_instance_A2]) == solution_prediction_2

    # Test instance 1 and 2 for N=20, M=7, F=2
    test_forest.fit(interview_X_train,interview_y_train,20,7,2)
    test_instance_B = [['Junior','Java','no','yes'],['Junior','Java','yes','no']]
    solution_prediction_3 = NotImplemented
    assert test_forest.predict(test_instance_B) == solution_prediction_3
