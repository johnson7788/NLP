#!/bin/bash
PORT=5000
SCRIPT=flask_api_server
#进程数量
WORKER=${WORKER:-2}
#默认多线程个数10
THREAD=${THREAD:-10}

#是否使用多线程模式, 默认多线程
USERTHREAD=${USERTHREAD:-0}

if [ $USERTHREAD -eq 0 ]; then
  echo "启用多线程,线程个数 $THREAD"
  gunicorn -b localhost:$PORT -w $WORKER --threads $THREAD $SCRIPT:app
else
  echo "启用多进程, 进程个数 $WORKER"
  gunicorn -b localhost:$PORT -w $WORKER $SCRIPT:app
fi