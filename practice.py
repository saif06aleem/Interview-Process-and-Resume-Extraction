from flask import Flask, render_template, request, session, flash, redirect, url_for, send_file
import sqlite3 as sql
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import cv2
from keras.models import load_model
import numpy as np
import time
import tkinter as tk
import threading
import sounddevice as sd
import soundfile as sf
import os
import speech_recognition as sr
import csv
from difflib import SequenceMatcher
from PIL import Image, ImageTk
from subprocess import Popen
#importing libraries
from extract_txt import read_files
from txt_processing import preprocess
from txt_to_features import txt_features, feats_reduce
from extract_entities import get_number, get_email, rm_email, rm_number, get_name, get_skills
from model import simil
import json
import uuid

#used directories for data, downloading and uploading files
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files\\resumes\\')
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files\\outputs\\')
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data/')

# Make directory if UPLOAD_FOLDER does not exist
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# Make directory if DOWNLOAD_FOLDER does not exist
if not os.path.isdir(DOWNLOAD_FOLDER):
    os.mkdir(DOWNLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['SECRET_KEY'] = 'nani?!'

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'doc', 'docx'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


'''@app.route('/', methods=['GET'])
def main_page():
    return _show_page()'''


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    app.logger.info(request.files)
    upload_files = request.files.getlist('file')
    app.logger.info(upload_files)
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if not upload_files:
        flash('No selected file')
        return redirect(request.url)
    for file in upload_files:
        original_filename = file.filename
        if allowed_file(original_filename):
            extension = original_filename.rsplit('.', 1)[1].lower()
            filename = str(uuid.uuid1()) + '.' + extension
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_list = os.path.join(UPLOAD_FOLDER, 'files.json')
            files = _get_files()
            files[filename] = original_filename
            with open(file_list, 'w') as fh:
                json.dump(files, fh)

    flash('Upload succeeded')
    return redirect(url_for('upload_file'))


@app.route('/download/<code>', methods=['GET'])
def download(code):
    files = _get_files()
    if code in files:
        path = os.path.join(UPLOAD_FOLDER, code)
        if os.path.exists(path):
            return send_file(path)
    os.abort(404)


def _show_page():
    files = _get_files()
    return render_template('resume.html', files=files)


def _get_files():
    file_list = os.path.join(UPLOAD_FOLDER, 'files.json')
    if os.path.exists(file_list):
        with open(file_list) as fh:
            return json.load(fh)
    return {}


@app.route('/process', methods=["POST"])
def process():
    if request.method == 'POST':

        rawtext = request.form['rawtext']
        jdtxt = [rawtext]
        resumetxt = read_files(UPLOAD_FOLDER)
        p_resumetxt = preprocess(resumetxt)
        p_jdtxt = preprocess(jdtxt)

        feats = txt_features(p_resumetxt, p_jdtxt)
        feats_red = feats_reduce(feats)

        df = simil(feats_red, p_resumetxt, p_jdtxt)

        t = pd.DataFrame({'Original Resume': resumetxt})
        dt = pd.concat([df, t], axis=1)

        dt['Phone No.'] = dt['Original Resume'].apply(lambda x: get_number(x))

        dt['E-Mail ID'] = dt['Original Resume'].apply(lambda x: get_email(x))

        dt['Original'] = dt['Original Resume'].apply(lambda x: rm_number(x))
        dt['Original'] = dt['Original'].apply(lambda x: rm_email(x))
        dt['Candidate\'s Name'] = dt['Original'].apply(lambda x: get_name(x))

        skills = pd.read_csv(DATA_FOLDER + 'skill_red.csv')
        skills = skills.values.flatten().tolist()
        skill = []
        for z in skills:
            r = z.lower()
            skill.append(r)

        dt['Skills'] = dt['Original'].apply(lambda x: get_skills(x, skill))
        dt = dt.drop(columns=['Original', 'Original Resume'])
        sorted_dt = dt.sort_values(by=['JD 1'], ascending=False)

        out_path = DOWNLOAD_FOLDER + "Candidates.csv"
        sorted_dt.to_csv(out_path, index=False)

        return send_file(out_path, as_attachment=True)
class QuizApp:
    def __init__(self, root):
        self.root = root
        self.max_emotion_label = None
        self.max_emotion_count = None
        self.root.title("Quiz App with Emotion Detection")
        self.root.geometry("800x600")
        # Load the pre-trained face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load the pre-trained emotion recognition model
        self.emotion_model = load_model("C:/Users/USER/Desktop/INTERVIEW_PROJECT/my_model.zip")
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.start_button = tk.Button(root, text="Start", command=self.start_quiz, bg="#aaffaa")
        self.start_button.pack(pady=20)
        self.question_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.question_label.pack()
        self.countdown_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.countdown_label.pack()
        self.speak_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.speak_label.pack()
        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()
        self.cap = None
        self.emotion_thread = None
        self.emotion_start_times = {label: None for label in self.emotion_labels}
        self.latest_answers = []

    def start_quiz(self):
        self.start_button.config(state=tk.DISABLED)
        self.questions = ["Define Python ?", "What is pandas in python ?", "What is numpy in python ?"]
        self.current_question_index = 0
        self.emotion_thread = threading.Thread(target=self.detect_emotions)
        self.emotion_thread.start()
        self.ask_question_after_delay()

    def ask_question_after_delay(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=question)
            self.start_countdown(5)
            self.current_question_index += 1
        else:
            self.show_completion_message()

    def start_countdown(self, seconds):
        self.countdown_label.config(text=f"Time remaining: {seconds} seconds")
        if seconds > 0:
            self.root.after(1000, self.start_countdown, seconds - 1)
        else:
            self.countdown_label.config(text="")
            self.record_audio()

    def record_audio(self):
        duration = 5
        self.speak_label.config(text="Speak now")
        threading.Thread(target=self._record_audio, args=(duration,)).start()

    def _record_audio(self, duration):
        fs = 44100
        audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype='int16')
        sd.wait()
        save_directory = "C:/Users/USER/Desktop/INTERVIEW_PROJECT"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        filename_wav = os.path.join(save_directory, f"question_{self.current_question_index}.wav")
        sf.write(filename_wav, audio_data, fs)
        text = self.convert_audio_to_text(filename_wav)
        filename_text = os.path.join(save_directory, f"question_{self.current_question_index}.txt")
        with open(filename_text, 'w') as text_file:
            text_file.write(text)
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def convert_audio_to_text(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            filename_answer = os.path.join(r'C:/Users/USER/Desktop/INTERVIEW_PROJECT',
                                           f"Ans{self.current_question_index}.txt")
            with open(filename_answer, 'r') as answer_file:
                answer_text = answer_file.read()
                similarity = SequenceMatcher(None, text.lower(), answer_text.lower()).ratio()
            result_text = f"Question {self.current_question_index} - Similarity with Answer: {similarity * 100:.2f}%"
            self.save_result_to_csv(result_text)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        return text

    def save_result_to_csv(self, result_text):
        csv_filename = os.path.join("C:/Users/USER/Desktop/INTERVIEW_PROJECT", "results.csv")
        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Question Number", "Similarity Result"])
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.current_question_index, result_text])

    def clear_question(self):
        self.question_label.config(text="")
        self.speak_label.config(text="")
        self.ask_question_after_delay()

    def save_answer_similarity(self, similarity):
        self.latest_answers.append(similarity)
        if len(self.latest_answers) > 3:  # Keep track of the latest 5 answers
            self.latest_answers.pop(0)  # Remove the oldest answer

    def get_average_similarity(self):
        if self.latest_answers:
            return sum(self.latest_answers) / len(self.latest_answers)
        return 0

    def show_completion_message(self):
        self.question_label.config(text="Congratulations! You have completed the quiz.")
        self.start_button.config(state=tk.NORMAL)
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
        if self.emotion_thread is not None and self.emotion_thread.is_alive():
            self.emotion_thread.join()
        max_emotion_label = max(self.emotion_counters, key=self.emotion_counters.get)
        max_emotion_count = self.emotion_counters[max_emotion_label]
        self.max_emotion_label = max_emotion_label
        self.max_emotion_count = max_emotion_count
        average_similarity = self.get_average_similarity()
        return average_similarity

    def detect_emotions(self):
        # Dictionary to store the emotion counters
        self.emotion_counters = {label: 0 for label in self.emotion_labels}

        # Open the system camera (0 represents the default camera, you can change it if you have multiple cameras)
        self.cap = cv2.VideoCapture(0)

        while True:
            # Read a frame from the camera
            ret, frame = self.cap.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform face detection
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Iterate through detected faces
            for (x, y, w, h) in faces:
                # Extract the face region
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                # Normalize the pixel values
                roi_gray = roi_gray / 255.0

                # Reshape for the model input
                roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

                # Perform emotion prediction
                emotion_prediction = self.emotion_model.predict(roi_gray)

                # Get the dominant emotion
                dominant_emotion = self.emotion_labels[np.argmax(emotion_prediction)]

                # Display the emotion label on the frame
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                            cv2.LINE_AA)

                # Draw rectangles around the detected faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Check if the emotion is the same as the previous frame
                if self.emotion_start_times[dominant_emotion] is None:
                    self.emotion_start_times[dominant_emotion] = time.time()
                elif time.time() - self.emotion_start_times[dominant_emotion] > 2:
                    # If the emotion has been detected for more than 2 seconds, increment the counter
                    self.emotion_counters[dominant_emotion] += 1
                    print(f"{dominant_emotion} count: {self.emotion_counters[dominant_emotion]}")

            # Display the resulting frame
            self.display_frame(frame)

            # Break the loop if 'Esc' key is pressed
            if cv2.waitKey(1) == 27:  # 27 is the ASCII code for 'Esc'
                break

        # Release the camera and close the window
        self.cap.release()
        cv2.destroyAllWindows()

    def display_frame(self, frame):
        # Convert the frame to RGB format for displaying in tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/gohome')
def homepage():
    return render_template('index.html')


@app.route('/service')
def servicepage():
    return render_template('services.html')


@app.route('/coconut')
def coconutpage():
    return render_template('Coconut.html')


@app.route('/cocoa')
def cocoapage():
    return render_template('cocoa.html')


@app.route('/arecanut')
def arecanutpage():
    return render_template('arecanut.html')


@app.route('/paddy')
def paddypage():
    return render_template('paddy.html')


@app.route('/about')
def aboutpage():
    return render_template('about.html')


@app.route('/enternew')
def new_user():
    return render_template('signup.html')


@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO agriuser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",
                            (nm, phonno, email, unm, passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()


@app.route('/userlogin')
def user_login():
    return render_template("login.html")


@app.route('/logindetails', methods=['POST', 'GET'])
def logindetails():
    if request.method == 'POST':
        usrname = request.form['username']
        passwd = request.form['password']

        with sql.connect("agricultureuser.db") as con:
            cur = con.cursor()
            cur.execute("SELECT username,password FROM agriuser where username=? ", (usrname,))
            account = cur.fetchall()

            for row in account:
                database_user = row[0]
                database_password = row[1]
                if database_user == usrname and database_password == passwd:
                    session['logged_in'] = True
                    return render_template('home1.html')
                else:
                    flash("Invalid user credentials")
                    return render_template('login.html')


@app.route('/info')
def predictin():
    return render_template('info.html')


@app.route('/info1')
def predictin1():
    return render_template('info1.html')






@app.route('/info', methods=['POST', 'GET'])
def predcrop():
    global app
    if request.method == 'POST':
        if __name__ == "__main__":
            root = tk.Tk()
            app = QuizApp(root)
            root.mainloop()

        average_similarity = app.show_completion_message()
        return render_template('result123.html', max_emotion_label=app.max_emotion_label,
                               max_emotion_count=app.max_emotion_count,
                               average_similarity=average_similarity)


@app.route('/info1', methods=['POST', 'GET'])
def predcrop1():
    if request.method == 'POST':
        # Replace 'INTERVIEW_PROJECT' with the actual directory path where app.py resides
        proc = Popen(['python', 'INTERVIEW_PROJECT/app.py'])
        proc.wait()  # Wait for the subprocess to finish
        return render_template('resume.html')


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
