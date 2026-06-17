import os
import shutil
import gradio as gr

from sentence_transformers import CrossEncoder

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PDF_DIR = "day08/pdfs"
PERSIST_DIR = "day08/chroma_pdf"
def answer (question, history):
    return f"you asked {question}"


gr.ChatInterface(
    fn=answer,
    title="Day 8 — PDF RAG"
).launch(share=True)