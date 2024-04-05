import os
from typing import List, Union, Callable

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from sanic.request import File

from qanything_kernel.configs.model_config import UPLOAD_ROOT_PATH, SENTENCE_SIZE, ZH_TITLE_ENHANCE
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import *
from qanything_kernel.utils.loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from qanything_kernel.utils.loader.csv_loader import CSVLoader
from qanything_kernel.utils.loader.my_recursive_url_loader import MyRecursiveUrlLoader
from qanything_kernel.utils.splitter import ChineseTextSplitter
from qanything_kernel.utils.splitter import zh_title_enhance

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n", ".", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " "],
    chunk_size=400,
    length_function=num_tokens,
)


class LocalFile:
    def __init__(self, user_id, kb_id, file: Union[File, str], file_id, file_name, embedder, is_url=False,
                 in_milvus=False):
        self.user_id = user_id  # 用户id
        self.kb_id = kb_id  # kb：knowledge base，也就是知识库
        self.file_id = file_id  # 文件id
        self.docs: List[Document] = []  # Document是langchain_community.docstore.document.Document
        self.embs = []  # embedding
        self.emb_infer = embedder  # embedding是YouDaoLocalEmbeddings?
        self.url = None
        self.in_milvus = in_milvus  # 是否在milvus中
        self.file_name = file_name  # 文件名
        if is_url:  # 如果是url
            self.url = file  # url链接
            self.file_path = "URL"
            self.file_content = b''
        else:
            if isinstance(file, str):  # 什么情况下file是str？看起来是文件路径
                self.file_path = file
                with open(file, 'rb') as f:
                    self.file_content = f.read()
            else:  # 这种情况下file应该就是File类型，也就是上传的文件
                upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
                file_dir = os.path.join(upload_path, self.file_id)
                os.makedirs(file_dir, exist_ok=True)
                self.file_path = os.path.join(file_dir, self.file_name)  # 本地的文件路径
                self.file_content = file.body

            with open(self.file_path, "wb+") as f:  # 将文件内容写入本地文件
                f.write(self.file_content)

        debug_logger.info(f'success init localfile {self.file_name}')

    def split_file_to_docs(self, ocr_engine: Callable, sentence_size=SENTENCE_SIZE,
                           using_zh_title_enhance=ZH_TITLE_ENHANCE):
        if self.url:
            # url的处理，自己写的MyRecursiveUrlLoader
            debug_logger.info("load url: {}".format(self.url))
            loader = MyRecursiveUrlLoader(url=self.url)
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(text_splitter=textsplitter)
        elif self.file_path.lower().endswith(".md"):
            # md文件的处理,直接用的是langchain_community.document_loaders.UnstructuredFileLoader
            loader = UnstructuredFileLoader(self.file_path, mode="elements")
            docs = loader.load()
        elif self.file_path.lower().endswith(".txt"):
            # txt文件的处理，直接用的是langchain_community.document_loaders.TextLoader
            loader = TextLoader(self.file_path, autodetect_encoding=True)
            texts_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(texts_splitter)
        elif self.file_path.lower().endswith(".pdf"):
            # pdf文件的处理，自己写的UnstructuredPaddlePDFLoader
            loader = UnstructuredPaddlePDFLoader(self.file_path, ocr_engine)
            texts_splitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
            docs = loader.load_and_split(texts_splitter)
        elif self.file_path.lower().endswith(".jpg") or self.file_path.lower().endswith(
                ".png") or self.file_path.lower().endswith(".jpeg"):
            # 图片文件的处理，自己写的UnstructuredPaddleImageLoader
            loader = UnstructuredPaddleImageLoader(self.file_path, ocr_engine, mode="elements")
            texts_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(text_splitter=texts_splitter)
        elif self.file_path.lower().endswith(".docx"):
            # docx文件的处理，直接用的是langchain_community.document_loaders.UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(self.file_path, mode="elements")
            texts_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(texts_splitter)
        elif self.file_path.lower().endswith(".xlsx"):
            # xlsx文件的处理
            # loader = UnstructuredExcelLoader(self.file_path, mode="elements")
            csv_file_path = self.file_path[:-5] + '.csv'
            xlsx = pd.read_excel(self.file_path, engine='openpyxl')
            xlsx.to_csv(csv_file_path, index=False)
            loader = CSVLoader(csv_file_path, csv_args={"delimiter": ",", "quotechar": '"'})  # 自己写的CSVLoader
            docs = loader.load()
        elif self.file_path.lower().endswith(".pptx"):
            # pptx文件的处理，直接用的是langchain_community.document_loaders.UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(self.file_path, mode="elements")
            docs = loader.load()
        elif self.file_path.lower().endswith(".eml"):
            # eml文件的处理，直接用的是langchain_community.document_loaders.UnstructuredEmailLoader
            loader = UnstructuredEmailLoader(self.file_path, mode="elements")
            docs = loader.load()
        elif self.file_path.lower().endswith(".csv"):
            # csv文件的处理
            loader = CSVLoader(self.file_path, csv_args={"delimiter": ",", "quotechar": '"'})
            docs = loader.load()
        else:
            raise TypeError("文件类型不支持，目前仅支持：[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]")

        if using_zh_title_enhance:
            debug_logger.info("using_zh_title_enhance %s", using_zh_title_enhance)
            docs = zh_title_enhance(docs)

        # 重构docs，如果doc的文本长度大于800tokens，则利用text_splitter将其拆分成多个doc
        # text_splitter: RecursiveCharacterTextSplitter
        debug_logger.info(f"before 2nd split doc lens: {len(docs)}")
        docs = text_splitter.split_documents(docs)
        debug_logger.info(f"after 2nd split doc lens: {len(docs)}")

        # 这里给每个docs片段的metadata里注入file_id
        for doc in docs:
            doc.metadata["file_id"] = self.file_id
            doc.metadata["file_name"] = self.url if self.url else os.path.split(self.file_path)[-1]

        write_check_file(self.file_path, docs)

        if docs:
            debug_logger.info('langchain analysis content head: %s', docs[0].page_content[:100])
        else:
            debug_logger.info('langchain analysis docs is empty!')

        self.docs = docs

    def create_embedding(self):
        """
        创建embedding，遍历docs，将每个doc的page_content传入emb_infer._get_len_safe_embeddings方法，得到embedding
        在此之前，需要先调用split_file_to_docs方法，将文件拆分成多个doc，就是所谓的文档分块的过程
        :return:
        """
        self.embs = self.emb_infer._get_len_safe_embeddings([doc.page_content for doc in self.docs])
