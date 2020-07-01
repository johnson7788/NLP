from aiohttp import web
from typing import List
import os


Tokens = [
    {"sign": "c8499156ae8acc3f2d6a1453b9315eb1", "timestamp": "1592805495.355849", "token": "ff1c1eef10cad322ddbcd842a952b46c"}
]
UPLOAD_FOLDER = 'images/'

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

async def authorize(request: web.Request):
        if not 'token' in request.headers or not 'timestamp' in request.headers or not 'sign' in request.headers:
            return web.HTTPForbidden()

        token = request.headers['token']
        timestamp = request.headers['timestamp']
        sign = request.headers['sign']
        if not verify(token,timestamp,sign):
            return web.HTTPForbidden()
        request.transport.write(b"HTTP/1.1 100 Continue\r\n\r\n")

async def handle(request):
    name = request.match_info.get('name', "Anonymous")
    text = "Hello, " + name
    return web.Response(text=text)

async def echo(request):
    return web.Response(text='echo')

async def upload(request):
    post = await request.post()
    image = post.get("image")
    with open(UPLOAD_FOLDER + image.filename, 'wb') as file:
        file.write(image.file.read())
    return web.json_response({'result': 'success upload'})

app = web.Application()
app.add_routes([web.get('/', handle),
                web.post('/upload', upload, expect_handler=authorize),
                web.get('/echo', echo)])

if __name__ == '__main__':
    web.run_app(app, port=5000, host='0.0.0.0' )