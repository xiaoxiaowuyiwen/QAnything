import concurrent.futures
import hashlib
import json
import time
import traceback
import urllib.request
import uuid
from typing import (
    List,
)

import requests

from qanything_kernel.configs import model_config

'''
添加鉴权相关参数 -
    appKey : 应用ID
    salt : 随机值
    curtime : 当前时间戳(秒)
    signType : 签名版本
    sign : 请求签名

    @param appKey    您的应用ID
    @param appSecret 您的应用密钥
    @param paramsMap 请求参数表
'''


def addAuthParams(appKey, appSecret, params):
    q = params.get('q')
    if q is None:
        q = params.get('img')
    q = "".join(q)
    salt = str(uuid.uuid1())
    curtime = str(int(time.time()))
    sign = calculateSign(appKey, appSecret, q, salt, curtime)
    params['appKey'] = appKey
    params['salt'] = salt
    params['curtime'] = curtime
    params['signType'] = 'v3'
    params['sign'] = sign


def returnAuthMap(appKey, appSecret, q):
    salt = str(uuid.uuid1())
    curtime = str(int(time.time()))
    sign = calculateSign(appKey, appSecret, q, salt, curtime)
    params = {'appKey': appKey,
              'salt': salt,
              'curtime': curtime,
              'signType': 'v3',
              'sign': sign}
    return params


'''
    计算鉴权签名 -
    计算方式 : sign = sha256(appKey + input(q) + salt + curtime + appSecret)
    @param appKey    您的应用ID
    @param appSecret 您的应用密钥
    @param q         请求内容
    @param salt      随机值
    @param curtime   当前时间戳(秒)
    @return 鉴权签名sign
'''


def calculateSign(appKey, appSecret, q, salt, curtime):
    strSrc = appKey + getInput(q) + salt + curtime + appSecret
    return encrypt(strSrc)


def encrypt(strSrc):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(strSrc.encode('utf-8'))
    return hash_algorithm.hexdigest()


def getInput(input):
    if input is None:
        return input
    inputLen = len(input)
    return input if inputLen <= 20 else input[0:10] + str(inputLen) + input[inputLen - 10:inputLen]


# https://ai.youdao.com/DOCSIRMA/html/aigc/api/embedding/index.html
# https://github.com/youdao-zhiyun/aicloud-demo-python/blob/master/apidemo/TextEmbeddingDemo.py

class YouDaoEmbeddings:
    model_name: str = "text-embedding-youdao-001"
    deployment: str = model_name  # to support Azure OpenAI Service custom deployment names
    embedding_ctx_length: int = 416
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    base_url: str = model_config.ONLINE_EMBED_SERVICE_URL

    def __init__(self):
        pass

    def _get_embedding(self, queries):
        # queries = '123'   # 如果传入的是一个字符串，可以正常工作
        curtime = int(time.time())
        salt = str(curtime)

        '''
        sign=sha256(应用ID+input+salt+curtime+应用密钥)；
        其中，input的计算方式为：input=q前10个字符 + q长度 + q后10个字符（当q长度大于20）或 input=q字符串（当q长度小于等于20）；
        传多个q时,需要拼接为一个字符串参与签名计算。例：第一个q=文本1,第二个q=文本2,则参与计算签名q=文本1文本2
        '''

        # 拼接多个q字符串
        combined_q = ''.join(queries)

        print(f'_get_embedding, queries length: {len(queries)}, combined_q length: {len(combined_q)}', flush=True)

        # 计算input
        if len(combined_q) > 20:
            input_str = combined_q[:10] + str(len(combined_q)) + combined_q[-10:]
        else:
            input_str = combined_q

        # 获取当前时间戳
        cur_time = str(curtime)

        # 拼接字符串并计算SHA256签名
        sign_str = model_config.ONLINE_EMBED_APP_ID + input_str + salt + cur_time + model_config.ONLINE_EMBED_APP_KEY
        sign = hashlib.sha256(sign_str.encode('utf-8')).hexdigest()
        print(
            f'app id: {model_config.ONLINE_EMBED_APP_ID}, app key: {model_config.ONLINE_EMBED_APP_KEY}, salt: {salt}, curtime: {cur_time}, sign: {sign}',
            flush=True)

        '''
        data = {
            'appKey': model_config.ONLINE_EMBED_APP_ID,
            'curtime': curtime,
            'q': queries,
            'queries': queries,
            'salt': salt,
            'sign': sign,
            'signType': 'v3',
        }
        '''
        # print(f'data: {data}', flush=True)
        print('embedding data length:', sum(len(s) for s in queries), flush=True)
        # headers = {"content-type": "application/json"}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        url = self.base_url
        # 下面这样是不行的，如果queries是一个简单的字符串，可以正常工作，但是如果是一个列表，就会报错
        # todo：现在由2种方法解决这个问题，第一是继续使用网易的embedding，其次是使用其他厂商的embedding，例如阿里云的
        # todo: 如果使用阿里云的，那么维度是1536，而网易的是768，这个在设置zilliz的milvus的collection时需要注意适配
        # url = f'{url}?appKey={model_config.ONLINE_EMBED_APP_ID}&curtime={curtime}&salt={salt}&sign={sign}&signType=v3&q={queries}'
        try:
            data = {'q': queries}
            addAuthParams(model_config.ONLINE_EMBED_APP_ID, model_config.ONLINE_EMBED_APP_KEY, data)
            res = requests.post(url, data, headers)
            print(f'!!!!!!!!!!embedding response!!!!!!!!!!!!: {res.json()}', flush=True)
            return res.json()
        except Exception as e:
            print(f'embedding error: {traceback.format_exc()}, e: {str(e)}')
            return None

    def getModelVersion(self):
        data = ''
        headers = {"content-type": "application/json"}

        url = self.base_url + "/getModelVersion"
        req = urllib.request.Request(
            url=url,
            headers=headers,
            data=json.dumps(data).encode("utf-8")
        )

        f = urllib.request.urlopen(
            req
        )
        js = json.loads(f.read().decode())

        return js

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本的embedding，输入是文本列表，输出是embedding列表
        :param texts:
        :return:
        """
        all_embeddings = []
        batch_size = 16
        total_texts = sum(len(s) for s in texts)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(self._get_embedding, batch)
                futures.append(future)
            for future in futures:
                embd_js = future.result()
                if embd_js:
                    # embeddings = embd_js["embeddings"]
                    embeddings = embd_js["result"]["embeddingList"]
                    # model_version = embd_js["model_version"]
                    model_version = embd_js["result"]["modelVersion"]
                    print(model_version)
                    all_embeddings += embeddings
                else:
                    raise Exception("embedding error, data length: %d" % total_texts)
        return all_embeddings

    @property
    def embed_version(self):
        return self.getModelVersion()['model_version']
