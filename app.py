'''
#=====================================================================================
#Jack Brandt
#Course: CPSC 322
#Assignment: Final
#Date of current version: 12/??/2024
#Brief description of what program does:
#    Flask app, flask app, flask app
#=====================================================================================
'''
from flask import Flask,render_template, request
from mysklearn.myutils import load_model, stress_discretizer,heart_rate_discretizer,\
    duration_discretizer
from mysklearn.myclassifiers import MyKNeighborsClassifier

app = Flask(__name__)

# Homepage
@app.route("/", methods=["GET", "POST"])
def homepage():
    if request.method == "POST":
        stress_level = int(request.form["stress_level"])
        average_heart_rate = int(request.form["average_heart_rate"])
        duration = int(request.form["duration"])*60*1000 # Converting to miliseconds
        clas = MyKNeighborsClassifier(3)
        X_train,y_train = load_model()
        clas.fit(X_train,y_train)
        prediction = clas.predict([[stress_discretizer(stress_level),heart_rate_discretizer(average_heart_rate),duration_discretizer(duration)]])
        return str(prediction)
    return render_template('home.html')

if __name__ == '__main__':
    # header, tree = load_model()
    # print(header)
    # print(tree)
    app.run(host='0.0.0.0',port=5000, debug=True)
    # TODO: when deploy app to 'production', set debug=False
    # and check host and port values