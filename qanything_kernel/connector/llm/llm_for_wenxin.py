import json
from abc import ABC
from typing import List

import requests
import tiktoken
from dotenv import load_dotenv

from qanything_kernel.configs import model_config
from qanything_kernel.connector.llm.base import (BaseAnswer, AnswerResult)

load_dotenv()


# 直接调用wenxin的API，不需要自己部署服务
class WenxinLLM(BaseAnswer, ABC):
    model: str = "gpt-3.5-turbo"
    token_window: int = 4096
    max_token: int = 512
    offcut_token: int = 50
    truncate_len: int = 50
    temperature: float = 0
    history: List[List[str]] = []
    history_len: int = 2

    def __init__(self):
        super().__init__()
        self.wenxin_access_token = self._get_access_token()
        print(f'wenxin_access_token: {self.wenxin_access_token}')
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _get_access_token(self):
        client_id = model_config.WENXIN_CLIENT_ID
        client_secret = model_config.WENXIN_CLIENT_SECRET
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}"
        payload = ""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)
        json_data = response.json()
        error = json_data.get('error')
        if error:
            error_description = json_data.get('error_description')
            print('error:', error, 'error_description:', error_description)
            return None
        return json_data['access_token']

    def _request_wenxin(self, messages):
        # todo：如果access token过期，重新获取access token，并再次调用接口
        # 文心错误码：https://cloud.baidu.com/doc/WENXINWORKSHOP/s/tlmyncueh
        call_succ = False
        while not call_succ:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + self.wenxin_access_token
            payload = json.dumps({"messages": messages})
            headers = {'Content-Type': 'application/json'}
            response = requests.request("POST", url, headers=headers, data=payload)
            json_data = response.json()
            usage = json_data.get('usage')
            if usage:
                total_input_tokens = usage.get('prompt_tokens', 0)
                total_output_tokens = usage.get('completion_tokens', 0)
                print('total_input_tokens:', total_input_tokens, 'total_output_tokens:', total_output_tokens)
            result = json_data.get('result', '')
            if result:
                return result
            elif json_data.get('error_code') == 110:
                self.wenxin_access_token = self._get_access_token()
                continue
            else:
                print(f"Error calling Wenxin API: {json_data}")
                return ""

    @property
    def _llm_type(self) -> str:
        return "OpenAILLM"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def num_tokens_from_messages(self, message_texts):
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = 0
        for message in message_texts:
            num_tokens += len(encoding.encode(message, disallowed_special=()))
        return num_tokens

    def num_tokens_from_docs(self, docs):
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = 0
        for doc in docs:
            num_tokens += len(encoding.encode(doc.page_content, disallowed_special=()))
        return num_tokens

    def _call(self, prompt: str, history: List[List[str]]) -> str:
        messages = []
        for pair in history:
            question, answer = pair
            messages.append({"role": "user", "text": question})
            messages.append({"role": "assistant", "text": answer})
        messages.append({"role": "user", "text": prompt})
        print(messages)
        try:
            response = self._request_wenxin(messages)
            return response
        except Exception as e:
            print(f"Error calling Wenxin API: {e}")
            return ""

    def generatorAnswer(self, prompt: str, history: List[List[str]] = [], streaming: bool = False) -> AnswerResult:
        response_text = self._call(prompt, history)
        answer_result = AnswerResult()
        answer_result.llm_output = {"answer": response_text}
        answer_result.prompt = prompt
        yield answer_result
