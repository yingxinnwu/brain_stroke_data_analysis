# Description:
# This web app displays graphical data of the average bmi and glucose values of stroke vs non-stroke individuals
# The user is able to input their age and bmi or glucose value
# to see if their data best aligns with that of stroke or non-stroke individuals
# prediction is made using knn

from flask import Flask, redirect, render_template, request, session, url_for, Response, send_file
import os
import io
import sqlite3 as sl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split # function
from sklearn.neighbors import KNeighborsClassifier # class
import numpy as np

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
db = "brain_stroke.db"

"""
render home page that displays the two graph options 
"""
@app.route("/")
def home():
    options = {
        "bmi": "Compare average BMI of brain stroke vs non-brain stroke patients",
        "avg_blood_glucose": "Compare average blood glucose levels of brain stroke vs non-brain stroke patients"
    }
    return render_template("home.html", biosexes=db_get_biosexes(), message="Average Blood Glucose & BMI Data of Brain Stroke vs Non-Brain Stroke Patients", options=options)

# user is able to choose whether they want to see the data of female or male individuals
"""
redirect to biosex_current if appropriate data_request and biological sex option supplied by user
else redirect back to home page 
"""
@app.route("/submit_biosex", methods=["POST"])
def submit_biosex():
    session["biosex"] = request.form["biosex"]
    if 'biosex' not in session or session["biosex"] == "":
        return redirect(url_for("home"))
    if "data_request" not in request.form:
        return redirect(url_for("home"))
    session["data_request"] = request.form["data_request"]
    return redirect(url_for("biosex_current", data_request=session["data_request"], biosex=session["biosex"]))

"""
renders display to display the graphs
data_request: whether user picked to display bmi or average glucose graph
biosex: whether the user picked to display info for male or female patients 
"""
@app.route("/api/brainstroke/<data_request>/<biosex>")
def biosex_current(data_request, biosex):
    return render_template("display.html", data_request=data_request, biosex=biosex, project=False)

"""
redirect to biosex_prediction if appropriate info received for age and value 
else redirect to home
"""
@app.route("/submit_prediction", methods=["POST"])
def submit_prediction():
    if 'biosex' not in session:
        return redirect(url_for("home"))
    session["age"] = request.form["age"]
    session["value"] = request.form["value"]
    # error checking, if blank then reroute to home page
    if session["biosex"] == "" or session["data_request"] == "" or session["age"] == "" or session["value"] == "":
        return redirect(url_for("home"))
    if 'age' not in session or 'value' not in session:
        return redirect(url_for("biosex_current", data_request=session["data_request"], biosex=session["biosex"]))
    return redirect(url_for("biosex_prediction", data_request=session["data_request"], biosex=session["biosex"]))

"""
data_request: whether user picked to display bmi or average glucose graph
biosex: whether the user picked to display info for male or female patients 
render display to show graph with prediction
"""
@app.route("/api/brainstroke/<data_request>/prediction/<biosex>")
def biosex_prediction(data_request, biosex):
    return render_template("display.html", data_request=data_request, biosex=biosex, project=True, age=session["age"], value=session["value"])

"""
data_request: whether user picked to display bmi or average glucose graph
biosex: whether the user picked to display info for male or female patients 
return: image of figure
"""
@app.route("/fig/<data_request>/<biosex>")
def fig(data_request, biosex):
    fig = create_figure(data_request, biosex) #create figure
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype="image/png")

"""
create appropriate figure 
data_request: whether user picked to display bmi or average glucose graph
biosex: whether the user picked to display info for male or female patients 
return: figure 
"""
def create_figure(data_request, biosex):
    # get dataframe from function
    df = create_dataframe()
    fig = Figure()
    fig, ax = plt.subplots(1, 1)  # 1 row, 1 column

    # create new dataframe for stroke data
    df_stroke = df[df["stroke"] == 1]
    df_stroke = df_stroke[df["gender"] == biosex]  # group by biological sex
    age_list_stroke = df_stroke["age"].unique()  # get list of all ages in dataset

    # create new dataframe for non-stroke data
    df_non_stroke = df[df["stroke"] == 0]
    df_non_stroke = df_non_stroke[df["gender"] == biosex]  # group by biological sex
    age_list_non_stroke = df_non_stroke["age"].unique()  # get list of all ages in dataset

    # display graph without predicted values
    if "age" not in session:
        # graph for avg blood glucose
        if data_request == "avg_blood_glucose":
            ax.set(title="Average blood glucose of stroke vs non-stroke patients", xlabel="Age", ylabel="Average Blood Glucose")  # set titles and labels
            avg_s_glucose_list = [] # average of the average blood glucose for stroke patients
            avg_ns_glucose_list = [] # average of the average blood glucose for non-stroke patients
            df_grouped1 = df_stroke.groupby("age")  # put them all together into a big dataframe
            for age in age_list_stroke: # go through every age and calculate the average glucose value for that age
                df_group = df_grouped1.get_group(age)
                avg_val = df_group["avg_glucose_level"].mean()
                avg_s_glucose_list.append(avg_val)  # add average to avg_glucose_list for stroke patients

            df_grouped2 = df_non_stroke.groupby("age")  # put them all together into a big dataframe
            for age in age_list_non_stroke: # go through every age and calculate the average glucose value for that age
                df_group = df_grouped2.get_group(age)
                avg_val = df_group["avg_glucose_level"].mean()
                avg_ns_glucose_list.append(avg_val)  # add to average avg_glucose_list for non-stroke patients
            # make bar graph
            ax.bar(age_list_stroke, avg_s_glucose_list, width=0.8, alpha=0.2, color="purple", label="stroke")  # plot stroke values
            ax.bar(age_list_non_stroke, avg_ns_glucose_list, width=0.8, alpha=0.2, color="red", label="non-stroke")  # plot non-stroke values
            ax.grid(which='major', color='#999999', alpha=0.2)
            ax.legend()
            plt.tight_layout()
            plt.show()
        # graph for bmi
        elif data_request == "bmi":
            ax.set(title="Average BMI of stroke vs non-stroke patients", xlabel="Age", ylabel="BMI") # set title and and labels
            avg_bmi_s = [] # list of average bmi values for each age for stroke patients
            avg_bmi_ns = [] # list of average bmi values for each age for non-stroke patients
            df_grouped1 = df_stroke.groupby("age")  # put them all together into a big dataframe
            for age in age_list_stroke: # go through every age and calculate the average bmi value for that age
                df_group = df_grouped1.get_group(age)
                avg_val = df_group["bmi"].mean()
                avg_bmi_s.append(avg_val)  # add to avg_glucose_list
            df_grouped2 = df_non_stroke.groupby("age")  # put them all together into a big dataframe
            for age in age_list_non_stroke: # go through every age and calculate the average bmi value for that age
                df_group = df_grouped2.get_group(age)
                avg_val = df_group["bmi"].mean()
                avg_bmi_ns.append(avg_val)  # add to avg_glucose_list
            # make bar graph
            ax.bar(age_list_stroke, avg_bmi_s, width=0.8, alpha=0.2, color="purple", label="stroke")  # plot stroke values
            ax.bar(age_list_non_stroke, avg_bmi_ns, width=0.8, alpha=0.2, color="red", label="non-stroke")  # plot non-stroke values
            ax.grid(which='major', color='#999999', alpha=0.2)
            ax.legend()
            plt.tight_layout()
            plt.show()
        return fig
    else: # create graph with prediction
        # graph for avg blood glucose
        if data_request == "avg_blood_glucose":
            ax.set(title="Average blood glucose of stroke vs non-stroke patients", xlabel="Age", ylabel="Average Blood Glucose")  # set titles and labels
            avg_s_glucose_list = []  # average of the average blood glucose for stroke patients for each age
            avg_ns_glucose_list = []  # average of the average blood glucose for non-stroke patients for each age
            df_grouped1 = df_stroke.groupby("age")  # put them all together into a big dataframe
            for age in age_list_stroke:
                df_group = df_grouped1.get_group(age)
                avg_val = df_group["avg_glucose_level"].mean() # get average of the glucose values for this age
                avg_s_glucose_list.append(avg_val)  # add to avg_glucose_list for stroke patients

            df_grouped2 = df_non_stroke.groupby("age")  # put them all together into a big dataframe
            for age in age_list_non_stroke:
                df_group = df_grouped2.get_group(age)
                avg_val = df_group["avg_glucose_level"].mean() # get average of the glucose values for this age
                avg_ns_glucose_list.append(avg_val)  # add to avg_glucose_list for non-stroke patients
            # make bar graph
            ax.bar(age_list_stroke, avg_s_glucose_list, width=0.8, alpha=0.2, color="purple",
                   label="stroke")  # plot stroke values
            ax.bar(age_list_non_stroke, avg_ns_glucose_list, width=0.8, alpha=0.2, color="red",
                   label="non-stroke")  # plot non-stroke values
            ax.grid(which='major', color='#999999', alpha=0.2)

            all_avg_glucose = avg_s_glucose_list + avg_ns_glucose_list
            all_ages = list(age_list_stroke) + list(age_list_non_stroke)
            stroke_vals = []

            for i in range(len(avg_s_glucose_list)):
                stroke_vals.append(1)
            for i in range(len(avg_ns_glucose_list)):
                stroke_vals.append(0)

            complete_data = [[all_avg_glucose[i], all_ages[i]] for i in
                             range(len(all_avg_glucose))]  # use list comprehension to get 2d array with all average glucose and age data
            stroke_np = np.array(stroke_vals)
            # use scikit learn to make prediction
            X_train, X_test, y_train, y_test = train_test_split(complete_data, stroke_np, test_size=0.25, random_state=0)
            knn = KNeighborsClassifier(n_neighbors=1)  # KNeighborsClassifier is a class
            knn.fit(X_train, y_train)
            curr_age = int(session["age"]) # get age for prediction from user
            curr_value = float(session["value"]) # get value for prediction from user
            X_new = np.array([[curr_age, curr_value]]) # create a numpy array for prediction
            prediction = knn.predict(X_new)
            prediction = int(prediction) # get prediction
            # if the prediction is 0, it means that the user's entered value for the age they specified is closest to an individual with no stroke
            if prediction == 0:
                ax.bar(curr_age, curr_value, width=0.8, alpha=0.2, color="green",
                       label="your glucose value best aligns the non-stroke category")  # add the user's age and value to the graph
                # specify what their predicted result is in the legend
            # if the prediction is 0, it means that the user's entered value for the age they specified is closest to an individual with brain stroke
            else:
                # add the user's age and value to the graph, specify predicted result in the legend
                ax.bar(curr_age, curr_value, width=0.8, alpha=0.2, color="orange",
                       label="caution: your glucose value best aligns the stroke category")
            ax.legend()
            plt.tight_layout()
            plt.show()
        # graph for bmi values
        elif data_request == "bmi":
            ax.set(title="Average BMI of stroke vs non-stroke patients", xlabel="Age", ylabel="BMI")
            avg_bmi_s = []  # average bmi values for each age for stroke patients
            avg_bmi_ns = []  # average bmi values for each age for non-stroke patients
            df_grouped1 = df_stroke.groupby("age")  # put them all together into a big dataframe
            for age in age_list_stroke: # get average bmi values for each age for stroke patients
                df_group = df_grouped1.get_group(age)
                avg_val = df_group["bmi"].mean() # get average bmi value for each age
                avg_bmi_s.append(avg_val)  # add average to list
            df_grouped2 = df_non_stroke.groupby("age")  # put them all together into a big dataframe
            for age in age_list_non_stroke: # get average bmi values for each age for non-stroke patients
                df_group = df_grouped2.get_group(age)
                avg_val = df_group["bmi"].mean() # get average bmi value for each age
                avg_bmi_ns.append(avg_val)  # add average to list
            # make bar graph
            ax.bar(age_list_stroke, avg_bmi_s, width=0.8, alpha=0.2, color="purple",
                   label="stroke")  # plot stroke values
            ax.bar(age_list_non_stroke, avg_bmi_ns, width=0.8, alpha=0.2, color="red",
                   label="non-stroke")  # plot non-stroke values
            ax.grid(which='major', color='#999999', alpha=0.2)

            all_avg_bmi = avg_bmi_s + avg_bmi_ns
            all_ages = list(age_list_stroke) + list(age_list_non_stroke)
            stroke_vals = []

            for i in range(len(avg_bmi_s)):
                stroke_vals.append(1)
            for i in range(len(avg_bmi_ns)):
                stroke_vals.append(0)

            complete_data = [[all_avg_bmi[i], all_ages[i]] for i in
                             range(len(all_avg_bmi))]  # use list comprehension to get 2d array
            stroke_np = np.array(stroke_vals)

            # use scikit learn to make a prediction on whether the user's age and value best aligns with that of stroke or non-stroke data
            X_train, X_test, y_train, y_test = train_test_split(complete_data, stroke_np, test_size=0.25, random_state=0)
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train, y_train)
            curr_age = int(session["age"])
            curr_value = float(session["value"])
            X_new = np.array([[curr_age, curr_value]])
            prediction = knn.predict(X_new)
            prediction = int(prediction) # get prediction

            if prediction == 0: # user's inputted data best corresponds to non-stroke data
                ax.bar(curr_age, curr_value, width=0.8, alpha=0.2, color="green",
                       label="your bmi value best aligns the non-stroke category")  # plot stroke values
            else: # user's inputted data best corresponds to stroke data
                ax.bar(curr_age, curr_value, width=0.8, alpha=0.2, color="orange",
                       label="caution: your bmi value best aligns the stroke category")
            ax.legend()
            plt.tight_layout()
            plt.show()
        return fig

"""
input: none
return: dataframe created from the brain_stroke database
"""
def create_dataframe():
    conn = sl.connect(db)
    curs = conn.cursor()
    # create dataframe
    sql_query = pd.read_sql_query('''SELECT * FROM brain_stroke''', conn) # select all columns from the brain_stroke table
    df = pd.DataFrame(sql_query, columns=['gender', 'age', 'avg_glucose_level', 'bmi', 'stroke']) # create dataframe
    conn.close()
    return df

"""
input: none
return: list of biological sexes (male & female)
"""
def db_get_biosexes():
    conn = sl.connect(db)
    curs = conn.cursor()
    stmt = "SELECT `gender` FROM brain_stroke"
    data = curs.execute(stmt)
    biosexes = sorted({result[0] for result in data})
    conn.close()
    return biosexes

@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)