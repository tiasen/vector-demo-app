from typing import List

from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main(question: str) -> str:
    loader = TextLoader("./blogs/vector-db.txt")

    doc = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    chunks = splitter.split_documents(doc)
    
    print(len(chunks))
    
    embedding = OpenAIEmbeddings()
    
    chroma = Chroma(persist_directory="./vector-db")
    
    db = chroma.from_documents(chunks, embedding=embedding)
    
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())
    
    result = qa.run(question)
    return result
 

if __name__ == "__main__":
    print("Hello langchain!")

    query = "How to use Vector DB?"
    result = main(query)  
    print(result)
