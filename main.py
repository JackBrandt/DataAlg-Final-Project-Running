'''#=====================================================================================
#Jack Brandt TODO: Go through and change names in all mysklearn to include
#               Suyash and change assignments and dates according for this project
#Course: CPSC 322
#Assignment: Project
#Date of current version: 1?/??/2024
#Brief description of what program does:
#    This file is for testing random things as neeeded, please ignore.
#====================================================================================='''
from mysklearn import myclassifiers as mc
from mysklearn.myutils import compute_bootstrapped_sample
from mysklearn.myevaluation import stratified_kfold_split, accuracy_score

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

interview_y_train = ["False", "False", "True", "True",
                      "True", "False", "True", "False",
                        "True", "True", "True", "True",
                          "True", "False"]

forest = mc.MyRandomForestClassifier()
#forest.fit(interview_X_train,interview_y_train,1,1,4,0)
forest.fit(interview_X_train,interview_y_train,5,3,1,0)

# Figuring out what are the correct answers for the trees in myrandomforest.fit

# The following are all the trees that forest fit will generate
get_column = lambda index, table: [[row[index]] for row in table]
get_columns = lambda indexes, table: [[row[index] for index in indexes] for row in table]

folds=stratified_kfold_split(interview_X_train,interview_y_train,3,0,True)
fold = folds[0]
remainder_X=[interview_X_train[x] for x in fold[0]]
remainder_y=[interview_y_train[x] for x in fold[0]]
training_X, validation_X = compute_bootstrapped_sample(remainder_X,0) # The seed for these depends on what number tree that is generated is the best
training_y, validation_y = compute_bootstrapped_sample(remainder_y,0) # Seed y has to match the one on X
# Gotta reduce training X/y to just the selected attribute
reduced_training_X = get_column(1,training_X)
tree_1 = mc.MyDecisionTreeClassifier()
tree_1.fit(reduced_training_X,training_y) # Gotta fill in fits
print(tree_1.print_decision_rules())
print(validation_X)
y_pred = tree_1.predict(get_column(1,validation_X))
print(validation_y)
print(y_pred)
tree_1_accuracy = accuracy_score(validation_y,y_pred)
print(tree_1_accuracy)

training_X, validation_X = compute_bootstrapped_sample(remainder_X,1)
training_y, validation_y = compute_bootstrapped_sample(remainder_y,1)
reduced_training_X = get_column(3,training_X)
tree_2 = mc.MyDecisionTreeClassifier()
tree_2.fit(reduced_training_X,training_y)
y_pred = tree_2.predict(get_column(3,validation_X))
print(y_pred)
tree_2_accuracy = accuracy_score(validation_y,y_pred)
print(tree_2_accuracy)

# Tree 3
training_X, validation_X = compute_bootstrapped_sample(remainder_X,2)
training_y, validation_y = compute_bootstrapped_sample(remainder_y,2)
reduced_training_X = get_column(1, training_X)
tree_3 = mc.MyDecisionTreeClassifier()
tree_3.fit(reduced_training_X,training_y)
y_pred = tree_3.predict(get_column(1,validation_X))
print(y_pred)
tree_3_accuracy = accuracy_score(validation_y,y_pred)
print(tree_3_accuracy)

# Tree 4
training_X, validation_X = compute_bootstrapped_sample(remainder_X,3)
training_y, validation_y = compute_bootstrapped_sample(remainder_y,3)
reduced_training_X = get_column(1, training_X)
tree_4 = mc.MyDecisionTreeClassifier()
tree_4.fit(reduced_training_X,training_y)
y_pred = tree_4.predict(get_column(1,validation_X))
print(y_pred)
tree_4_accuracy = accuracy_score(validation_y,y_pred)
print(tree_4_accuracy)

# Tree 5
training_X, validation_X = compute_bootstrapped_sample(remainder_X,4)
training_y, validation_y= compute_bootstrapped_sample(remainder_y,4)
reduced_training_X = get_column(2, training_X)
tree_5 = mc.MyDecisionTreeClassifier()
tree_5.fit(reduced_training_X,training_y)
y_pred = tree_5.predict(get_column(2,validation_X))
print(y_pred)
tree_5_accuracy = accuracy_score(validation_y,y_pred)
print(tree_5_accuracy)
# So the best performing trees are tree 1,4,5
# Technically 1 and 3 are a tie, but we'll pick the one that is first

# Now for figuring out the correct answers for test 3... this is gonna suck
# Cause there are 20 trees to check...
# Might as well use a loop this

forest.fit(interview_X_train,interview_y_train,20,7,2,0)
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
for i in range(20):
    training_X, validation_X = compute_bootstrapped_sample(remainder_X,i)
    training_y, validation_y = compute_bootstrapped_sample(remainder_y,i)
    reduced_training_X = get_columns(attributes[i],training_X)
    this_tree = mc.MyDecisionTreeClassifier()
    this_tree.fit(reduced_training_X,training_y)
    y_pred=this_tree.predict(get_columns(attributes[i],validation_X))
    print('The accuracy of tree ', i, ' is: ',accuracy_score(validation_y,y_pred))
# From this we conclude the 7 best trees are i=4,9,10,5,6,12,15,19
    # Which is actually 8, because again there's a tie, and I have no
    # idea how a tie is broken or how these tree's are going to be ordered
    # *shrug

# This is for figuring out which order the trees should come in
print()
forest.fit(interview_X_train,interview_y_train,5,3,1,0)
print()
print('Vs the solution is: ', [[tree_5.tree,[2]],[tree_4.tree,[1]],[tree_1.tree,[1]]])
# I think this means that if there is a tie, the algorithm picks the tree with lower index

forest.predict([["Mid", "R", "yes", "no"]])