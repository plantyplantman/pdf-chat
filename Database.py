import os
import glob
import hashlib
import pickle
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
from PDFProcessor import PDFProcessor

os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


class Database:
    def __init__(self, folder_path: str = "dbs",
                 index_name="manual_db",
                 embeddings=OpenAIEmbeddings(),
                 llm=OpenAI(temperature=0, max_tokens=512)):
        self.embeddings = embeddings
        self.db = FAISS.load_local(
            folder_path=folder_path,
            index_name=index_name,
            embeddings=embeddings)
        compressor = LLMChainExtractor.from_llm(llm=llm)
        retriever = self.db.as_retriever()
        self.retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=compressor
        )

    def similarity_search(self, query: str):
        return self.retriever.get_relevant_documents(query)

    @classmethod
    def from_texts(cls,
                   texts: List[str],
                   folder_path: str = "dbs",
                   index_name="manual_db",
                   embeddings=OpenAIEmbeddings(),
                   llm=OpenAI(temperature=0,
                              max_tokens=512)):
        instance = cls(folder_path, index_name, embeddings, llm)
        instance.db = FAISS.from_texts(texts, embeddings)
        return instance

    @classmethod
    def from_pdf(cls, pdf_path: str, embeddings=OpenAIEmbeddings(),
                 llm=OpenAI(temperature=0, max_tokens=512)):
        # Process the PDF file and obtain a list of Document objects
        pdf_processor = PDFProcessor(pdf_path)
        documents = pdf_processor.process()
        db = FAISS.from_documents(documents, embeddings)
        # Initialize a new Database instance with the created FAISS instance
        new_instance = cls(folder_path="temp", index_name="temp",
                           embeddings=embeddings, llm=llm)
        new_instance.db = db
        new_instance.retriever = db.as_retriever()
        new_instance.retriever = ContextualCompressionRetriever(
            base_retriever=new_instance.retriever,
            base_compressor=LLMChainExtractor.from_llm(llm=llm))

        return new_instance

    def save(self, filename: str):
        self.db.save_local(filename)

    @classmethod
    def load(cls, filename: str,
             folder_path: str = "dbs",
             index_name="manual_db",
             embeddings=OpenAIEmbeddings(),
             llm=OpenAI(temperature=0, max_tokens=512)):
        instance = cls(folder_path, index_name, embeddings, llm)
        instance.db = FAISS.load_local(
            folder_path=folder_path,
            index_name=index_name,
            embeddings=embeddings)
        return instance

    def merge_from(self, other):
        self.db.merge_from(other.db)

    def update_database(self,
                        pdf_dir: str = "pdfs",
                        processed_files_pickle: str = "processed_files.pickle"):
        if not os.path.exists(processed_files_pickle):
            processed_files = set()
        else:
            with open(processed_files_pickle, "rb") as f:
                processed_files = pickle.load(f)

        pdf_files = glob.glob(f"{pdf_dir}/*.pdf")
        new_files = []

        for pdf_file in pdf_files:
            with open(pdf_file, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash not in processed_files:
                new_files.append(pdf_file)
                processed_files.add(file_hash)

        for pdf_file in new_files:
            pdf_processor = PDFProcessor(pdf_file)
            documents = pdf_processor.process()
            texts = [doc.page_content for doc in documents]
            new_db = Database.from_texts(
                texts,
                self.folder_path,
                self.index_name,
                self.embeddings,
                self.llm)
            self.merge_from(new_db)

        self.save("faiss_index")

        with open(processed_files_pickle, "wb") as f:
            pickle.dump(processed_files, f)
