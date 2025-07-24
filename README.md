# RAG QA Chatbot - Ask Questions from any PDF

This is an AI-powered **RAG (Retrieval-Augmented Generation) Chatbot** that allows users to upload a PDF and ask natural language questions about its content. It uses **IBM Watsonx LLM**, **Watsonx Embeddings**, **LangChain**, and **ChromaDB**, all wrapped in a simple **Gradio web interface**.

---

## Features

- Upload any PDF document
- Ask questions about its content
- Uses embeddings + vector similarity (ChromaDB)
- Powered by Watsonx Foundation Models and LangChain

---

## How It Works

1. **Load PDF**  
   The PDF file is parsed and converted into raw text using `PyPDFLoader`.

2. **Split Text**  
   The text is split into manageable chunks using `RecursiveCharacterTextSplitter`.

3. **Embed Chunks**  
   Each chunk is embedded using IBM’s `WatsonxEmbeddings` (`slate-125m` retriever model).

4. **Store Vectors (ChromaDB)**  
   The embeddings are stored in a local persistent vector database (ChromaDB).

5. **Retrieve Relevant Chunks**  
   When you ask a question, it is embedded and compared to the stored vectors to retrieve the most relevant chunks.

6. **Generate Answer**  
   The retrieved context is passed to IBM’s `mixtral-8x7b-instruct-v01` LLM to generate a concise answer.

7. **Respond in Gradio UI**  
   The generated response is displayed in a simple chat-like interface using Gradio.
   
---

## Tech Stack

- **LangChain** — RetrievalQA pipeline
- **IBM Watsonx** — LLM and embedding models via API
- **Chroma** — Vector database for semantic search
- **Gradio** — Frontend interface

