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
forest.fit(interview_X_train,interview_y_train,1,1,4,0)
forest.fit(interview_X_train,interview_y_train,5,3,1,0)