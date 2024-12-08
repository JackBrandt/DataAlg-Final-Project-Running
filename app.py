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

app = Flask(__name__)

# Homepage
@app.route("/", methods=["GET", "POST"])
def homepage():
    if request.method == "POST":
        page=''
        return page
    return render_template('home.html')

if __name__ == '__main__':
    # header, tree = load_model()
    # print(header)
    # print(tree)
    app.run(host='0.0.0.0',port=5000, debug=True)
    # TODO: when deploy app to 'production', set debug=False
    # and check host and port values