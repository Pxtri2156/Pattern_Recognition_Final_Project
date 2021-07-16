########################################
#               sonlhcsuit             #
#               14/07/2021             #
########################################
from flask import Flask, request, send_file,jsonify
from datetime import datetime
# from flask_cors import CORS
from flask_ngrok import run_with_ngrok
import os


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

def save_img():
    pass 

@app.route('/api/recognize',methods=['POST'])
def recognize():
    t = {"message": "recognize"}
    if request.files:
        audio_file = request.files['audio']
        # Gọi hàm lưu ảnh quang phổ vào spectrum 
        audio_path = save_audio(audio_file)
        print('request form' , request.form)
        metrics = {
            "classifier":[40,40,40,40,40,40],
            "spectrum" : "spectrum/example.png"
        }
        #
        # predict function, return type metrics
        #
        return jsonify(metrics),200
    else:
        return t,200

# @app.route('/api/query', methods=['post'])
# def api_query():
#     if request.files:
#         similarity = request.form.get('similarity', 'cosine')
#         method = request.form.get('method', 'COLOR')
#         lsh = request.form.get('lsh', '0')
#         # print(similarity, method, lsh)
#         file = request.files['query_image']
#         if allowed_file(file.filename) and file:
#             q_id = save_image(file)
#             print(q_id)
#             out = query(q_id, method=method, similarity=similarity, lsh=lsh)
#             out = tojson(out)
#             out = list(map(lambda x: x.split('/')[-1], out))
#             data = generate_links(out)
#             print(data)
#         # return {"img": [], "links": [], "q_id": q_id}, 200

#         return {"data":data, "q_id": q_id}, 200
#     else:
#         return {"message": "No file"}, 200


# @app.route('/api/image/q/<string:q_id>', methods=['GET'])
# def get_q_image(q_id):
#     fp = os.path.join( app.config['qp'], q_id)
#     if os.path.exists(fp):
#         img = os.listdir(fp)[0]
#         fp = os.path.join(fp, img)
#         encoded = get_encoded_img(fp)
#         return {"img": encoded}, 200
#     else:
#         return {'error': 'Query image not found'}, 404


# @app.route('/api/image/<string:img_name>', methods=['GET'])
# def get_image(img_name):
#     base = '/content/database'
#     fp = os.path.join(base, img_name)
#     if os.path.exists(fp):
#         encoded = get_encoded_img(fp)
#         return {"img": encoded}, 200
#     else:
#         return {'error': 'Image not found'}, 404


if __name__ == '__main__':
    os.system('rm -rf audio')
    # os.system('rm -rf spectrum')
    os.system('mkdir -p audio')
    # os.system('mkdir -p spectrum')
    ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
    # CORS(app)
    # run_with_ngrok(app)
    app.run()