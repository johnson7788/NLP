import unittest
import requests
import time

url = 'http://127.0.0.1:5000'
#确保Flask server已经启动

r = requests.get(url + '/gentoken')
res = r.json()
headers = {
    'token': res['token'],
    'timestamp': res['timestamp'],
    'sign': res['sign']
}

class FlaskTestCase(unittest.TestCase):
    def test_get_echo(self):
        """测试服务器正常启动"""
        r = requests.get(url +'/echo')
        self.assertEqual(r.status_code,  200)

    def test_get_token(self):
        """测试获取token"""
        r = requests.get(url+'/gentoken')
        self.assertEqual(r.status_code,  200)
        self.assertIn('sign', r.json())
        self.assertIn('timestamp', r.json())
        self.assertIn('token', r.json())

    def test_try_upload_sync(self):
        """上传图片文件,并立即处理，返回成功结果"""
        with open("test.png", 'rb') as img:
            files = {
                'image': img
            }
            r = requests.post(url + '/upload_sync', headers=headers, files=files)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()['code'], 0)

    def test_try_upload_async(self):
        """测试上传图片，不能立即处理完成，先返回成功处理的页面，等待用户调取"""
        with open("test.png", 'rb') as img:
            files = {
                'image': img
            }
            r = requests.post(url + '/upload_async', headers=headers, files=files)
            self.assertEqual(r.status_code, 200)
        time.sleep(3)
        r = requests.get(url+'/upload_async_result', headers=headers)
        self.assertEqual(r.status_code,  200)
        self.assertEqual(r.json()['code'], 0)

if __name__ == '__main__':
    ##确保Flask server已经启动
    unittest.main()
