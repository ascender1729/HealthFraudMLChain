from block import write_block, check_integrity
from flask import Flask, render_template, request, redirect, session
import pandas as pd
import csv
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('block.csv')

df1 = df['member_name']
df2 = df['patient_suffix']

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Save signup data to the CSV file
        with open('users.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([username, password])

        return redirect('/login')

    return render_template('signup.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the username and password match the saved signup data
        with open('users.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username and row[1] == password:
                    # Set session variable to indicate successful login
                    session['logged_in'] = True
                    session['username'] = username
                    return redirect('/index')

        # Invalid credentials
        return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

# Index page
@app.route('/index', methods=['GET', 'POST'])
def index():
    # Check if the user is logged in
    if 'logged_in' in session and session['logged_in']:
        if request.method == 'POST':
            policy = request.form.get('level')
            policy1 = request.form.get('level1')
            policy2 = request.form.get('level2')
            policy3 = request.form.get('level3')
            policy4 = request.form.get('level4')
            policy5 = request.form.get('level5')
            policy6 = request.form.get('level6')
            policy7 = request.form.get('level7')
            policy8 = request.form.get('level8')
            policy9 = request.form.get('level9')
            policy10 = request.form.get('level10')
            policy11 = request.form.get('level11')
            policy12 = request.form.get('level12')
            policy13 = request.form.get('level13')
            policy14 = request.form.get('level14')
            write_block(policy=policy, policy1=policy1, policy2=policy2, policy3=policy3, policy4=policy4, policy5=policy5, policy6=policy6, policy7=policy7, policy8=policy8, policy9=policy9, policy10=policy10,
                        policy11=policy11, policy12=policy12, policy13=policy13, policy14=policy14)

        return render_template('index.html')
    else:
        return redirect('/login')

# Search page
@app.route("/search", methods=['POST'])
def search():
    dg = float(request.form.get("search"))
    for i in range(len(df1)):
        if df1[i] == dg:
            j = i
            tex = "Policy is Valid"
            df2[j] = df2[j] + 500
            df[j] = df2[j]
            df.to_csv('update.csv')
            return render_template("index.html", available=tex)
    text = "Policy not found"
    return render_template("index.html", notavail=text)

# Checking page
@app.route('/checking')
def check():
    results = check_integrity()
    return render_template('index.html', checking_results=results)

@app.route('/')
def home():
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)
