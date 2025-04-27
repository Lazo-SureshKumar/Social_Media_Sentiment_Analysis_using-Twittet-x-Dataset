from flask import Flask,render_template,request,flash,redirect,url_for,session,Response
import sqlite3
import os
from datetime import datetime
import main 
import re
import csv
from functools import wraps
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score 

app = Flask(__name__)
matplotlib.use('Agg')
UPLOAD_FOLDERS = 'static/files'
FILE_EXTENSIONS = {'csv'}
app.secret_key = 'secret123'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
app.config['UPLOAD_FOLDER'] =UPLOAD_FOLDER
#create user databse to store login cridencials
# con = sqlite3.connect("user.db")
# cur = con.cursor()
# cur.execute("CREATE TABLE  IF NOT EXISTS Users(username TEXT PRIMARY KEY,password INTEGER NOT NULL)")
# username = "user"
# password = 1234
# cur.execute("INSERT INTO Users(username,password) VALUES(?,?)",(username,password))
# con.commit()
# con.close()

#check file extension
def allowed_extensions(file_name):
    return '.'in file_name and file_name.rsplit('.',1)[1].lower() in FILE_EXTENSIONS


# -------------------Login Section---------------

#check if user logged in
def is_user_logged_in(f):
    @wraps(f)
    def wrap(*args,**kwargs):
        if 'user_logged_in' in session:
            return f(*args,**kwargs)
        else:
            flash('Unauthorized,Please login',danger)
            return redirect(url_for('login'))
    return wrap

#user Login

@app.route('/',methods=['POST','GET'])
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect("user.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Users WHERE username = ? AND password = ?",(username,password))
        user = cursor.fetchone()
        
        if user:
            session['user_logged_in'] = True
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid Login. Try Again','danger')
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template("upload_dataset.html")
    else:
        return redirect(url_for('home'))

#uplod Dataset
@app.route('/upload_dataset',methods=['GET','POST'])
@is_user_logged_in
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.csv'):
            new_filename = 'dataset.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],new_filename)
            file.save(filepath)              
            flash("File uploaded suceessfuly")
            return redirect(url_for('view'  ))
        else:
            flash('Please upload a .csv file')
    return render_template('upload_dataset.html')

def cleaner(tweet):
    import re
    return ''.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    
@app.route('/view')
@is_user_logged_in
def view():
    data = []
    user = ''
    if os.path.exists(UPLOAD_FOLDER+'/dataset.csv'):
        with open(UPLOAD_FOLDER+'/dataset.csv',encoding="latin 1") as csvfile:
            reader= csv.reader(csvfile,delimiter=",")   
            i=1
            print(reader)
            for row in reader:
                if i>0:
                    data.append(row)
                    i+=1
    return render_template('view.html',data=data)
    
@app.route('/clean')
@is_user_logged_in
def clean():
    records=[]
    user=""
    if os.path.exists(UPLOAD_FOLDER+'/dataset.csv'):
        with open(UPLOAD_FOLDER+'/dataset.csv',encoding="latin 1") as csvfile:
            reader= csv.DictReader(csvfile,delimiter=',')  
            i=1
            for row in reader:
                if i>0:
                    txt=cleaner(row['text'])
                    if(len(txt)>10):
                        records.append([txt])
                        i+=1
            with open(UPLOAD_FOLDER+'/clean.csv','w',newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["tweet"])
                csvwriter.writerows(records)
                            # res = main.get_tweet_sentiment()
                return render_template('clean.html',data=records)
    return render_template('clean.html')
    
@app.route('/tweetAnalysis',methods=['POST','GET'])
@is_user_logged_in
def analysis():
    tweets = main.get_tweet_sentiment()
    return render_template("tweetAnalysis.html",data=tweets)
    
@app.route('/sentiment')
@is_user_logged_in
def result():
    res=main.get_sentiment()
    tweets=main.get_tweet_sentiment()
    result=[0,0,0]
    negative=0
    nutral=0
    for row in tweets:
        if row['sentiment']=='Positive':
            result[0]+=1
        elif row['sentiment']=='Negative':
            result[1]+=1
        else:
            result[2]+=1
    no_of_tweet = len(tweets)
    return render_template('sentiment.html',res=res,result=result,no_of_tweet = no_of_tweet)

def sklearn_pg():
    vecorizer = CountVectorizer()
    df = pd.read_csv(UPLOAD_FOLDER+'/cleaned.csv')
    X = vecorizer.fit_transform(df['tweet'])
    label = df['sentiment']
    X_train,X_test,y_train,y_test = train_test_split(X,label,test_size=0.2,random_state=42)
    # print(X_train,X_test,y_train,y_test)
    model = MultinomialNB()
    model.fit(X_train,y_train)
    # print(model)
    y_pred = model.predict(X_test)
    # print(y_pred)
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))


@app.route('/plot',methods=['POST','GET'])
@is_user_logged_in
def analysis_plot():
    import random
    # plot_name = str(random.randint(1,100000))+'.jpg'
    sklearn_pg()
    plot_name = 'analysis'+'.jpg'
    res=main.get_tweet_sentiment()
    tweets = main.get_tweet_sentiment()
    result=[0,0,0]
    negative=0
    nutral=0
    for row in tweets:
        if row['sentiment']=='Positive':
            result[0]+=1
        elif row['sentiment']=='Negative':
            result[1]+=1
        else:
            result[2]+=1
    data = {'Positive':result[0],'Negative':result[1],'Neutral':result[2]}
    X= list(data.keys())
    Y = list(data.values())
    fig = plt.figure(figsize=(10,5))
    plt.bar(X,Y,color=['green','red','maroon'],width=0.4)
    plt.xlabel("Sentiment")
    plt.ylabel("no.of Tweets")
    plt.title("Sentiment Analysis of Twitter Data")
    plt.savefig('static/files/'+plot_name)
    return render_template("plot.html",res=res,result=result,plot_name=plot_name)


    
@app.route('/logout')
@is_user_logged_in
def logout():
    session.clear()
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)