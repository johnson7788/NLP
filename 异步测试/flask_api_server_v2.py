import os
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
import hashlib
from functools import wraps
import time
from flask_executor import Executor
from PIL import Image
import pytesseract
from flask_redis import FlaskRedis

app = Flask(__name__)

#redis用于存放用户的token和异步时临时存储用户的索引
redis_client = FlaskRedis(app, charset='utf-8', decode_responses=True)

#启动异步操作多线程
executor = Executor(app)
app.config['EXECUTOR_TYPE'] = 'thread'
app.config['EXECUTOR_MAX_WORKERS'] = 5

#存放图片位置
UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = set(['jpg', 'png'])


@app.route("/echo", methods=['GET'])
def echo():
    """
    测试服务器运行正常
    :return:
    """
    return jsonify({'result': True})

def verify(token:str, timestamp:str ,sign:str)-> bool:
    """
    验证token，timestamp，sign
    :param token: token
    :param timestamp: 时间戳
    :param sign: 签名
    :return: bool
    """
    res = redis_client.hgetall(token)
    #如果token不存在，返回False
    if not res:
        return False
    #如果sign不正确，返回FALSE，
    res_vale = res.values()
    if timestamp in res_vale and sign in res_vale:
        return True
    return False

def authorize(f):
    @wraps(f)
    def decorated_function(*args, **kws):
            #如果header不存在token等关键字，直接返回401
            if not 'token' in request.headers or not 'timestamp' in request.headers or not 'sign' in request.headers:
                abort(401)
            token = request.headers['token']
            timestamp = request.headers['timestamp']
            sign = request.headers['sign']
            #如果token，签名验证不通过，返回401
            if not verify(token,timestamp,sign):
                abort(401)
            return f(*args, **kws)
    return decorated_function

def cal_md5(content) -> str:
    """
    给定content，计算md5
    :param content:
    :return:
    """
    md5 = hashlib.md5()
    content=str(content)
    md5.update(content.encode('UTF-8'))
    result = md5.hexdigest()
    return result

@app.route("/gentoken", methods=['GET', 'POST'])
def generate_token():
    """
    生成用户token, 把token放入redis，以后可以把用户信息存入DB，临时的token放入redis
    :return:
    """
    rand = os.urandom(32)
    token = cal_md5(rand)
    timestamp = str(time.time())
    sign = cal_md5(str(token)+timestamp)
    TK = {'timestamp':timestamp, 'sign':sign}
    redis_client.hmset(token, TK)
    return jsonify({'token': token, 'timestamp': timestamp, 'sign': sign})

def allowed_file(filename: str)-> bool:
    """
    校验上传的图片格式, 如果格式正确返回True
    :param filename:
    :return:
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def recognise(image: str) -> str:
    """
    使用tesseract路径识别图片
    :param image: 图片名字
    :return: 图片识别后的结果
    """
    #tesseract路径
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
    #图片识别成文字
    res = pytesseract.image_to_string(Image.open(UPLOAD_FOLDER+'/'+image))
    return res

#同步任务
@app.route("/upload_sync", methods=['POST'])
@authorize
def upload_sync():
    """
    上传图片文件,并立即处理，返回成功结果
    :return:
    """
    #存储图片
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        #开始识别图片
        res = recognise(filename)
        return jsonify({'code':0,'result': res})


@app.route("/upload_async", methods=['POST'])
@authorize
def upload_async():
    """
    上传图片，不能立即处理完成，先返回成功处理的页面，等待用户调取
    :return:
    """
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
    token = request.headers['token']
    #参数说明，第一个参数是标识操作的用户的token，后面是funtion和它的参数, 识别图片的并发
    executor.submit_stored(token, recognise, filename)
    return jsonify({'code':0, 'result':'upload sucess, Please get result from API upload_async_result'})

@app.route('/upload_async_result', methods=['GET'])
@authorize
def get_result():
    token = request.headers['token']
    #如果图片识别没有完成，那么返回图片正在识别中的状态，等待用户再次请求此接口
    if not executor.futures.done(token):
        return jsonify({'code':1, 'status': executor.futures._state(token), 'result': "Task is not complete, Please wait a second"})
    #用户图片识别完成，获取识别结果并返回给用户
    future = executor.futures.pop(token)
    return jsonify({'code':0, 'status': 'done', 'result': future.result()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
