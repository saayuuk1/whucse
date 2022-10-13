import requests
import json
import time

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
timer = time.perf_counter

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#填写百度控制台中相关开通了“音频文件转写”接口的应用的的API_KEY及SECRET_KEY
API_KEY = 'j6UhdcMdwQecpGCkGA9BFFNh'
SECRET_KEY = 'IQTjC93A78HIERZ8uy2YAh7yCy1q4b6p'

"""  获取请求TOKEN start 通过开通音频文件转写接口的百度应用的API_KEY及SECRET_KEY获取请求token"""

class DemoError(Exception):
    pass

TOKEN_URL = 'https://openapi.baidu.com/oauth/2.0/token'
# SCOPE = 'brain_bicc'  # 有此scope表示有asr能力，没有请在网页里勾选 bicc
SCOPE = 'brain_asr_async'  # 有此scope表示有asr能力，没有请在网页里勾选
# SCOPE = 'brain_enhanced_asr'  # 有此scope表示有asr能力，没有请在网页里勾选

def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    post_data = post_data.encode( 'utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req)
        result_str = f.read()
    except URLError as err:
        print('token http response http code : ' + str(err.code))
        result_str = err.read()
    result_str =  result_str.decode()

    result = json.loads(result_str)

    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not SCOPE in result['scope'].split(' '):
            raise DemoError('scope is not correct')
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')

"""  获取鉴权结束，TOKEN end """

def creat_task():
    """  发送识别请求 """

    #待进行语音识别的音频文件url地址，需要可公开访问。建议使用百度云对象存储（https://cloud.baidu.com/product/bos.html）
    speech_url_list = [
        "https://gitee.com/tsuyaki/image/raw/master/image/export_ofoct.com.mp3",
        ]   

    task_id_list = []

    for speech_url in speech_url_list:


        url = 'https://aip.baidubce.com/rpc/2.0/aasr/v1/create'  #创建音频转写任务请求地址

        body = {
            "speech_url": speech_url,
            "format": "mp3",        #音频格式，支持pcm,wav,mp3，音频格式转化可通过开源ffmpeg工具（https://ai.baidu.com/ai-doc/SPEECH/7k38lxpwf）或音频处理软件
            "pid": 1537,        #模型pid，1537为普通话输入法模型，1737为英语模型
            "rate": 16000       #音频采样率，支持16000采样率，音频格式转化可通过开源ffmpeg工具（https://ai.baidu.com/ai-doc/SPEECH/7k38lxpwf）或音频处理软件
        }

        # token = {"access_token":"24.19fd462ac988cb2d1cdef56fcb4b568a.2592000.1579244003.282335-11778379"}

        token = {"access_token":fetch_token()}

        headers = {'content-type': "application/json"}

        response = requests.post(url,params=token,data = json.dumps(body), headers = headers)

        # 返回请求结果信息，获得task_id，通过识别结果查询接口，获取识别结果
        print(response.text)
        task_id_list.append(response.json()['task_id'])
        
    return task_id_list

def query_result(task_id_list=None):
    """  发送查询结果请求 """

    #转写任务id列表，task_id是通过创建音频转写任务时获取到的，每个音频任务对应的值
    if not task_id_list:
        task_id_list = [
            "626b8cfc72dec6800729ea0e",
            ]   


    for task_id in task_id_list:


        url = 'https://aip.baidubce.com/rpc/2.0/aasr/v1/query'  #查询音频任务转写结果请求地址

        body = {
            "task_ids": [task_id],
        }

        token = {"access_token":fetch_token()}

        headers = {'content-type': "application/json"}

        response = requests.post(url,params=token,data = json.dumps(body), headers = headers)


        print(json.dumps(response.json(), ensure_ascii=False))

if __name__ == '__main__':
    #task_id_list = creat_task()
    query_result()