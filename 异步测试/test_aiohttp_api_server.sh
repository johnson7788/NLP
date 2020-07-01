#!/bin/bash
#测试上传文件接口
echo "Test upload image"
curl -X POST -H "token:ff1c1eef10cad322ddbcd842a952b46c" -H "timestamp:1592805495.355849" -H "sign:c8499156ae8acc3f2d6a1453b9315eb1" -F "image=@test.png" http://127.0.0.1:5000/upload

