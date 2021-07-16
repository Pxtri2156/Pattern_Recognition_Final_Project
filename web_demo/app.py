
from flask import Flask, request, send_file,jsonify
from datetime import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
from flask_ngrok import run_with_ngrok
import os
import sys
sys.path.append('../')
from utils.test_demo import print_name

# from flask_cors import CORS
def save_audio(file):
    q_id = datetime.timestamp(datetime.now())
    q_id = str(int(q_id))
    extension = file.filename.split(".")[-1]
    file_path = os.path.join(os.getcwd(), "audio", f"{q_id}.{extension}")
    file.save(file_path)
    return file_path


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#


app = Flask(__name__)


@app.route('/', methods=['GET'])
def html():
    f = open('web/index.html')
    data = f.read()
    return data


@app.route('/script.js', methods=['GET'])
def script():
    f = open('web/script.js')
    data = f.read()
    return data

@app.route('/spectrum/<string:img_name>', methods=['GET'])
def spectrum(img_name):
    return send_file(os.path.join(os.getcwd(), f"spectrum/{img_name}"), as_attachment=True), 200

@app.route('/api', methods=['POST', 'GET'])
def index():
    t = {"message": "Welcome to Anime Finder!"}
    return t, 200

def generate_spectogram_img(audio_path):
    print("audio path: ", audio_path)
    data, sample_rate = librosa.load(audio_path,duration=2.5, offset=0.6 )
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='log')
    save_path = "spectrum/tmp.png"
    plt.savefig(save_path)
    print("Saved at {}".format(save_path))
    return save_path

@app.route('/api/recognize',methods=['POST'])
def recognize():
    t = {"message": "recognize"}
    if request.files:
        audio_file = request.files['audio']
        # Gọi hàm lưu ảnh quang phổ vào spectrum 
        audio_path = save_audio(audio_file)
        img_path = generate_spectogram_img(audio_path)
        print_name()
        print('request form' , request.form)
        print("type: ", request.form['type'])
        metrics = {
            "classifier":[40,40,40,40,40,40],
            "spectrum" : img_path
        }
        #
        # predict function, return type metrics
        #
        return jsonify(metrics),200
    else:
        return t,200


if __name__ == '__main__':
    os.system('rm -rf audio')
    # os.system('rm -rf spectrum')
    os.system('mkdir -p audio')
    # os.system('mkdir -p spectrum')
    ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
    # CORS(app)
    # run_with_ngrok(app)
    app.run()