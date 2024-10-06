import chainlit as cl
import pathlib
import logging
import json
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, JSONLoader
from langchain.schema import Document

class DocumentLoaderException(Exception):
    pass

class DocumentLoader:
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".json": JSONLoader,
    }

    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        ext = pathlib.Path(file_path).suffix
        loader = DocumentLoader.supported_extensions.get(ext)
        if not loader:
            raise DocumentLoaderException(
                f'Invalid Extension Type {ext}, cannot load this type of file'
            )
        if ext == ".json":
            loader = JSONLoader(file_path, jq_schema='.', text_content=False)
        else:
            loader = loader(file_path)
        
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents")
        return docs

def configure_retriever(docs: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  
        chunk_overlap=20  
    )
    if isinstance(docs[0].page_content, dict):
        for doc in docs:
            doc.page_content = json.dumps(doc.page_content)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 4})
