#!/bin/bash
#每秒发送请求次数
RATE=2000/s
#测试时间，测试,总测试请求RATE*DURATION
DURATION=3s

#如果使用多线程，线程个数
THREAD=10
#如果使用多进程，进程个数
WORKER=4

#单核单线程
flaskbench_singlethread(){
  title=bench_singlethread
  echo "启动单核单线程server"
  python flask_api_server.py & > /dev/null 2>&1
#  ps aux | grep flask | grep -v grep  > bench_singlethread.html
  echo "开始压测并生成结果"
  $1 $title
  pkill python
  echo "关闭server"
}

#单核10线程
flaskbench_multithread(){
  title=bench_multithread
  echo "启动单核多线程server"
  bash gunicorn_flask_api_server.sh &
#  ps aux | grep flask | grep -v grep  > bench_multithread.html
  #等待gunicorn完全启动
  sleep 1
  echo "开始压测并生成结果"
  $1 $title
  pkill python
  sleep 1
  echo "关闭server"
}

#多核单线程
flaskbench_multiprocess(){
  title=bench_multiprocess
  #禁用线程FLAG，改用多核
  USERTHREAD=1
  echo "启动多核单线程 gunicorn_flask_api_server server"
  bash gunicorn_flask_api_server.sh &
#  ps aux | grep flask | grep -v grep  > bench_multiprocess.html
  #等待gunicorn完全启动
  sleep 1
  echo "开始压测并生成结果"
  $1 $title
  pkill python
  sleep 1
  echo "关闭server"
}

#ayncio异步server测试
flaskbench_aiohttp(){
  title=bench_aiohttp
  #禁用线程FLAG，改用多核
  echo "启动aiohttp_api_server server"
  python aiohttp_api_server.py & > /dev/null 2>&1
#  ps aux | grep flask | grep -v grep  > bench_multiprocess.html
  #等待gunicorn完全启动
  sleep 1
  echo "开始压测并生成结果"
  $1 $title
  pkill python
  sleep 1
  echo "关闭server"
}

#无负载压力测试
noload(){
jq -ncM '{method: "GET", url: "http://127.0.0.1:5000/echo", body: "Hello!" | @base64 }' | vegeta attack -format=json -rate=$RATE -duration=$DURATION > results.bin
cat results.bin  | vegeta report > $1.txt
cat results.bin | vegeta plot -title $1 > $1.html
rm results.bin
}

#加上传图片负载时的测试结果, 图片的上传方法暂时无文档，todo
uploadimage(){
jq -ncM '{method: "POST", url: "http://127.0.0.1:5000/upload", file:"image=@test.png", header: {"token": "ff1c1eef10cad322ddbcd842a952b46c", "timestamp":"1592805495.355849", "sign":"c8499156ae8acc3f2d6a1453b9315eb1"} }' | vegeta attack -format=json -rate=$RATE -duration=$DURATION > results.bin
cat results.bin  | vegeta report > $1.txt
cat results.bin | vegeta plot -title $1 > $1.html
rm results.bin
}

#flaskbench_singlethread noload
flaskbench_multithread noload
#flaskbench_multiprocess noload
flaskbench_aiohttp noload