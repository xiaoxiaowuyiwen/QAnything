import concurrent.futures
import hashlib
import json
import time
import traceback
import urllib.request
from typing import (
    List,
)

from qanything_kernel.configs import model_config


# https://ai.youdao.com/DOCSIRMA/html/aigc/api/embedding/index.html

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

        data = {
            'appKey': model_config.ONLINE_EMBED_APP_ID,
            'curtime': curtime,
            'q': queries,
            'queries': queries,
            'salt': salt,
            'sign': sign,
            'signType': 'v3',
        }
        print(f'data: {data}', flush=True)
        print('embedding data length:', sum(len(s) for s in queries), flush=True)
        headers = {"content-type": "application/json"}
        url = self.base_url
        url = f'{url}?appKey={model_config.ONLINE_EMBED_APP_ID}&curtime={curtime}&salt={salt}&sign={sign}&signType=v3'
        req = urllib.request.Request(url=url, headers=headers, data=json.dumps(data).encode("utf-8"))
        try:
            f = urllib.request.urlopen(req)
            js = json.loads(f.read().decode())
            print(f'!!!!!!!!!!embedding response!!!!!!!!!!!!: {js}', flush=True)
            return js
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
                    embeddings = embd_js["embeddings"]
                    model_version = embd_js["model_version"]
                    print(model_version)
                    all_embeddings += embeddings
                else:
                    raise Exception("embedding error, data length: %d" % total_texts)
        return all_embeddings

    @property
    def embed_version(self):
        return self.getModelVersion()['model_version']
