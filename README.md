Project description:
    Brief description of project goals:
        * Classify running speed based off other metrics
        * Learn about random forest classification
        * Have fun
        * Deploy a web app with our best model (*The bonus*)
        * Report on our findings
    And we accomplished all our goals, including the bonus
How to run:
    If all you're interested in is reading the project report, just open this project with some environment with Jupyter Notebook capabilities (like VScode) and click on project_proposal.ipynb. If it's not already ran for you, just hit run all at the top. If you are interested in running a local version of the web app (unnecessary since there is a version hosted on Render), just type:
        python app.py
    into the terminal and it will start the web app, then navigate to the https:whatever ip it says in ther terminal/. If you just want to access the one hosted on Render navigate to the link include down below and wait for Render to spin up the site.
How project is organized:
    * Data for this project is stored in a number of directories.
        * unprocessed Garmin data: to see a subset of the origin data Garmin returned
        * csv_converted_data: The data converted to csv's
        * joined_nullfree_subsets: To see a partially joined and null free set of data
        * processed data: To see the fully join and then fully processed set of data (processed_data.csv)
    * The mysklearn directory contains all of the data classifier algorithms for this project, including ones like KNN and random forest
    * The static and templates directories contain css and html files respectively necessary for the web app
    * app.py contains the program for the web app itself
    * test_myclassifiers.py contains tests for the classifiers in mysklearn, including for random forest
    * running_technical_report.ipynb is our final project technical report
Link to flask app:
    https://dataalg-final-project-running.onrender.com/