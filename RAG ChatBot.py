!pip install \
gradio==4.44.0 \
ibm-watsonx-ai==1.1.2  \
langchain==0.2.11 \
langchain-community==0.2.10 \
langchain-ibm==0.1.11 \
chromadb==0.4.24 \
pypdf==4.3.1 \
pydantic==2.9.1



from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr
import os
PERSIST_DIRECTORY = "chroma_db"

# Suppress Warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# Loading pdf document
def pdf_loader(file):
    loader = PyPDFLoader(file.name)
    doc = loader.load()
    return doc

# Splitting into chunks
def splitter(doc):
    split = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len
    )
    chunks = split.split_documents(doc)
    return chunks

# Define Embedding Model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

# Vector Database (Chroma)    
def vector_database(chunks):
    embedder = watsonx_embedding()

    # If persistent DB exists, load it
    if os.path.exists(PERSIST_DIRECTORY):
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedder
        )
    else:
        vectordb = Chroma.from_documents(
            chunks,
            embedder,
            persist_directory=PERSIST_DIRECTORY
        )
        vectordb.persist()  # Save to disk

    return vectordb



# Define Retriever
def retriever(vectordb):
    retrv = vectordb.as_retriever()
    return retrv


# LLM
def llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm

# Question-Answering Chain
def retriever_qa(retrv,query):
    qa_retriever = RetrievalQA.from_chain_type(
        llm=llm(),
        chain_type="stuff",
        retriever=retrv,
        return_source_documents=False
    )
    response = qa_retriever.invoke(query)
    return response['result']

# Execute
def exec(file,query):
    doc = pdf_loader(file)
    chunks = splitter(doc)
    vectordb = vector_database(chunks)
    retrv = retriever(vectordb)
    response = retriever_qa(retrv,query)
    return response

# Sample usage (without gradio)
# with open("A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf", "rb") as f:
#    print(exec(f, "What this paper is talking about?"))

# Gradio
rag_chatbot = gr.Interface(
    fn = exec,
    allow_flagging="never",
    inputs = [
        gr.File(label="Upload your PDF file", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Enter your query", placeholder="Ask a question about the content of your file...")
    ],
    outputs = gr.Textbox(label="Output"),
    title= "RAG Chatbot",
    description="Upload your PDF document and ask any question about its content. This chatbot uses RAG to retrieve relevant information from the document and tries to generate accurate answers using a LLM model."
)


# Launch app (Gradio)
rag_chatbot.launch(share=True)


