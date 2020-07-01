import os
from quart import Quart, request, jsonify, abort
from werkzeug.utils import secure_filename
import hashlib
from functools import wraps
import time

Tokens = [
    {"sign": "c8499156ae8acc3f2d6a1453b9315eb1", "timestamp": "1592805495.355849", "token": "ff1c1eef10cad322ddbcd842a952b46c"}
]

UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = set(['jpg', 'png'])

app = Quart(__name__)

def verify(token:str, timestamp:str ,sign:str)-> bool:
    """
    验证token，timestamp，sign
    :param token: token
    :param timestamp: 时间戳
    :param sign: 签名
    :return: bool
    """
    for tk in Tokens:
        tkv = tk.values()
        if token in tkv and timestamp in tkv and sign in tkv:
            return True
    return False

def authorize(f):
    @wraps(f)
    def decorated_function(*args, **kws):
            if not 'token' in request.headers or not 'timestamp' in request.headers or not 'sign' in request.headers:
                abort(401)

            token = request.headers['token']
            timestamp = request.headers['timestamp']
            sign = request.headers['sign']
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

@app.route("/token", methods=['GET', 'POST'])
async def generate_token():
    """
    生成用户token
    :return:
    """
    rand = os.urandom(32)
    token = cal_md5(rand)
    timestamp = str(time.time())
    sign = cal_md5(str(token)+timestamp)
    TK = {'token': token, 'timestamp':timestamp, 'sign':sign}
    Tokens.append(TK)
    return jsonify({'token': token, 'timestamp': timestamp, 'sign': sign})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#做一些异步任务,测试
async def blocking_io():
    #文件操作，IO类型任务，例如日志等，使用线程或异步asyncio
    with open("/dev/urandom", "rb") as f:
        return f.read(100)
async def cpu_bound():
    # CPU-Bound， 消耗CPU的操作，使用多进程完成
    return sum(i * i for i in range(10 ** 6))

#模拟同步任务，
@app.route("/upload_sync", methods=['POST'])
async def upload_sync():
    """
    上传图片文件,并立即处理，返回成功结果
    :return:
    """
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return jsonify({'result': 'success upload'})

#模拟异步任务
@app.route("/upload_async", methods=['POST'])
async def upload_async():
    """
    上传图片，不能立即处理完成，先返回成功处理的页面，等待用户调取
    :return:
    """
    with concurrent.futures.ProcessPoolExecutor() as pool:
        future = pool.submit(cpu_bound)
        for fut in concurrent.futures.as_completed([future]):
            return jsonify({'result': fut.done()})

@app.route("/echo", methods=['GET'])
async def echo():
    """
    测试服务器运行正常
    :return:
    """
    return jsonify({'result': True})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
