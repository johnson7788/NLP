import aiohttp
import asyncio
import unittest
import requests

url = 'http://127.0.0.1:5000'
#确保Flask server已经启动

def get_header():
    headers = {
        'token': 'ff1c1eef10cad322ddbcd842a952b46c',
        'timestamp': '1592805495.355849',
        'sign': 'c8499156ae8acc3f2d6a1453b9315eb1'
    }
    return headers

class NamesTestCase(unittest.TestCase):
    def test_get_echo(self):
        """测试服务器正常启动"""
        r = requests.get(url +'/echo')
        self.assertEqual(r.status_code,  200)

    def test_get_token(self):
        """测试获取token"""
        r = requests.get(url+'/token')
        self.assertEqual(r.status_code,  200)
        self.assertIn('sign', r.json())
        self.assertIn('timestamp', r.json())
        self.assertIn('token', r.json())

    def test_try_process(self):
        """测试多进程"""
        headers = get_header()
        r = requests.get(url + '/process',  headers=headers)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()['result'])

    def test_try_upload(self):
        """测试用获取的token上传图片"""
        with open("test.png", 'rb') as img:
            files = {
                'image': img
            }
            headers = get_header()
            r = requests.post(url + '/upload', headers=headers, files=files)
            self.assertEqual(r.status_code, 200)

if __name__ == '__main__':
    ##确保Flask server已经启动
    unittest.main()
