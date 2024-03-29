import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect
import mysql.connector
import time
import datetime
import pickle
from passlib.hash import sha256_crypt
from dotenv import dotenv_values

config = dotenv_values(".env")

usr = 'Admin'  # Default Admin Username
# Default Admin Password
pwd = sha256_crypt.hash(config['default_admin_password'])
usrnm = ""

app = Flask(__name__,
            static_folder='static')
app.debug = True
app.secret_key = "super secret key"

conn = mysql.connector.connect(host=config['host'],
                               user=config['user'],
                               password=config['password'],
                               database=config['database']
                               )
cur = conn.cursor()

if(cur):
    print("Connected to db")


model = pickle.load(open('messmodel.pkl', 'rb'))


def getattendees():
    query = ("SELECT count(*) FROM feedback WHERE attendance=1")
    cur.execute(query)
    q = cur.fetchone()
    ans = q[0]
    return ans


def getmeal():
    now = datetime.datetime.now()
    actual_time = datetime.time(now.hour, now.minute, now.second)
    t1 = datetime.time(12, 0, 0)
    t2 = datetime.time(20, 0, 0)
    if(actual_time < t1):
        meal = 'breakfast'
    elif(actual_time > t1 and actual_time < t2):
        meal = 'lunch'
    else:
        meal = 'dinner'
    return 'lunch'


# meal,username

def getmenuitemsbymeal(meal, username):
    # global usrnm
    timestamp = datetime.date(2022, 8, 25)
    query = ("select i.item_id,i.item_name from student_admin s,mess,contains c,menu_items i where s.reg_no=%s and s.mess_id=mess.mess_id and mess.date=%s and mess.meal=%s and mess.menu_id=c.menu_id and c.item_id=i.item_id")
    # ts = time.time()
    # timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    vals = (username, timestamp, meal)
    cur.execute(query, vals)
    result = cur.fetchall()
    # print(usrnm)
    return result


def getmenuid(meal, username):
    # global usrnm
    query = ("select m.menu_id,m.total_cal from student_admin s,mess,menu m where s.reg_no=%s and s.mess_id=mess.mess_id and mess.date=%s and mess.meal=%s and mess.menu_id=m.menu_id")
    timestamp = datetime.date(2022, 8, 25)
    vals = (username, timestamp, meal)
    cur.execute(query, vals)
    result = cur.fetchall()
    print(result)
    return result


def getspecialmenuitems(fest_name):
    query = ("select s.spl_food,s.name from special_foodrequest s,selects c where c.spl_food=s.spl_food and c.festival_name=%s;")
    val = (fest_name,)
    cur.execute(query, val)
    result = cur.fetchall()
    print(result)
    return result


@app.route('/', methods=["GET", "POST"])
# home
@app.route('/home')
def home():
    return render_template('index.html')

# student signup


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')

# Choose


@app.route('/choose')
def choose():
    return render_template('choose.html')

# student login


@app.route('/slogin', methods=['GET', 'POST'])
def studentlogin():
    return render_template('studentlogin.html')

# Admin Login


@app.route('/adminlogin', methods=['GET', 'POST'])
def adminlogin():
    return render_template('adminlogin.html')

# admin


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    uname = request.form.get('username')
    pw = request.form.get('password')
    global usr
    global pwd
    attendees = getattendees()
    if(uname == usr and sha256_crypt.verify(pw, pwd)):
        return render_template('admin.html', new_attendance='The number of attendees are {} \n'.format(attendees))
    else:
        return redirect('/adminlogin')

# analyse


@app.route('/analyse')
def analyse():
    return render_template('analyze.html')

# STUDENT


@app.route('/student', methods=['GET', 'POST'])
def student():
    global usrnm
    usrnm = request.form.get('username')
    pswd = request.form.get('password')
    print(usrnm, pswd)
    query = ("SELECT password FROM student_admin WHERE reg_no = %s ")
    credentials = (usrnm,)
    cur.execute(query, credentials)
    ans = cur.fetchone()
    ans = ans[0]
    if(sha256_crypt.verify(pswd, ans)):
        print(getmenuitemsbymeal(getmeal(), usrnm))
        print(getmenuid(getmeal(), usrnm))
        return render_template('menu.html', items=getmenuitemsbymeal(getmeal(), usrnm), ID=getmenuid(getmeal(), usrnm))
    else:
        return redirect('/slogin')

# attendance page


@app.route('/attendance', methods=['POST', 'GET'])
def attendance():
    return render_template('student.html')

# submit Feedback


@app.route('/submitfeedback', methods=['GET', 'POST'])
def submitfeedback():
    global usrnm
    feedback = request.form.get('feedback')
    query = ('INSERT INTO feedback (feedback) VALUES (%s) WHERE regno=%s ;')
    vals = (feedback, usrnm)
    cur.execute(query, vals)
    conn.commit()
    return redirect('/attendance')


# submit
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        global usrnm
        regno = usrnm
        namequery = ("SELECT fname FROM student_admin WHERE reg_no = %s")
        regno = (regno,)
        cur.execute(namequery, regno)
        regno = regno[0]
        name = cur.fetchone()
        name = name[0]
        attendance = request.form['attendance']
        date = datetime.datetime.now()
        if attendance == "yes":
            attendance = 1
        elif attendance == "no":
            attendance = 0
        add_entry = "REPLACE INTO feedback " + \
            "(regno,name,attendance,date_of_attendance) "+"VALUES(%s,%s,%s,%s)"
        entry = (regno, name, attendance, date)
        print(entry)
        try:
            cur.execute(add_entry, entry)
            conn.commit()
            return render_template('success.html')
        except mysql.connector.IntegrityError as err:
            flash("Error: Duplicate entry encountered", 'error')
            return render_template("student.html")

        # else:
        #     return render_template('student.html')
        # return "ok"


@app.route('/submitdetails', methods=['GET', 'POST'])
def submitdetails():
    regno = request.form['regno']
    fname = request.form['fname']
    mname = request.form['mname']
    lname = request.form['lname']
    email = request.form.get('email')
    pwd = sha256_crypt.encrypt(request.form.get('password'))
    messid = request.form['messid']
    phno = request.form['phno']
    state = request.form['State']
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(
        ts).strftime('%Y-%m-%d %H:%M:%S')
    regtime = timestamp

    sdetails = ("INSERT INTO student_admin (reg_no, fname,mname,lname,email_id,phone_no,reg_time,mess_id,state,password) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")
    vals = (regno, fname, mname, lname, email,
            phno, regtime, messid, state, pwd)
    cur.execute(sdetails, vals)
    conn.commit()
    return render_template('allsuccess.html')


# success
app.route('/success', methods=['GET', 'POST'])


def success():
    return render_template('success.html')

# addfestival


@app.route('/addfestival', methods=['GET', 'POST'])
def addfestival():
    festival = request.form.get('festival')
    print(festival)
    fest_add = "REPLACE INTO "+" food_festival(festival_name) "+" VALUES (%s)"
    fest_name = (festival,)
    cur.execute(fest_add, fest_name)
    conn.commit()
    return render_template('allsuccess.html')

# special-food request


@app.route('/foodrequest', methods=['GET', 'POST'])
def foodrequest():
    food = request.form.get('food')
    festival_name = request.form.get('festival')
    amount = request.form.get('amount')
    print(food)
    food_add = "INSERT INTO " + \
        " special_foodrequest(name, festival, quantity) " + \
        " VALUES (%s, %s, %s)"
    food_name = (food, festival_name, amount)
    cur.execute(food_add, food_name)
    conn.commit()
    return render_template('allsuccess.html')


# menu
@app.route('/menu', methods=['GET', 'POST'])
def menu():
    return render_template('menu.html', items=getmenuitemsbymeal(getmeal()), ID=getmenuid(getmeal()))


# special-request
@app.route('/specialrequest')
def specialrequest():
    global usrnm
    query = ("select distinct f.festival_name from student_admin s, food_festival f where s.reg_no=%s and s.state=f.state or f.state='IN' order by length(f.festival_name)")
    values = (usrnm,)
    cur.execute(query, values)
    result = cur.fetchall()
    return render_template('specialrequest.html', result=result)


# special-food
@app.route('/specialfood/<fest_name>', methods=['GET', 'POST'])
def specialfood(fest_name):
    print(fest_name)
    query = (
        "SELECT spl_food, name, quantity from special_foodrequest where festival=%s;")
    fest = (fest_name,)
    cur.execute(query, fest)
    festlist = cur.fetchall()
    return render_template('specialfood.html', fest_name=fest_name, festlist=festlist)

# last page(success)


@app.route('/allsuccess')
def allsuccess():
    return render_template('allsuccess.html')

# feedback


@app.route('/feedback', methods=['POST', 'GET'])
def feedback():
    fb = request.form.get('comments')
    print(fb)
    add_entry = "INSERT INTO "+" comments "+" VALUES (%s)"
    fb_1 = (fb,)
    cur.execute(add_entry, fb_1)
    conn.commit()
    return render_template('student.html')


# prediction0
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    day = request.form.get("weekday")
    day = day.lower()

    def refactoring(day):
        word_dict = {'monday': 0, 'tuesday': 1, 'wednesday': 2,
                     'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
        rating_dict = {0: 7, 1: 8.5, 2: 9.1, 3: 8.9, 4: 8.6, 5: 7, 6: 7.9}
        wastage_dict = {0: 153.334, 1: 143, 2: 107.233,
                        3: 102.223, 4: 112.344, 5: 349.456, 6: 330.233}
        weekend = 1 if day in ['Saturday', 'Sunday'] else 0
        wastage = 0
        return list([word_dict[day], weekend, rating_dict[word_dict[day]], wastage, wastage_dict[word_dict[day]]])

    input_list = refactoring(day)
    inputt_list = input_list[:len(input_list)-1]
    prediction = model.predict([inputt_list])
    print([refactoring(day)])
    output = round(prediction[0], 3)
    attendees = getattendees()

    return render_template('admin.html', new_attendance='The number of attendees are {} \n'.format(attendees), prediction_text1='Menu rating for Today is : {} \n'.format(input_list[2]), prediction_text2='Average wastage on this day is: {} Kgs \n '.format(input_list[4]), prediction_text3='To avoid this wastage this is the predicted amount to be cooked :\n{} Kgs'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(port=8000, debug=True)
