import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from Database import Database
from PDFProcessor import PDFProcessor

os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def pretty_print_docs(docs):
    """Helper function for printing docs"""
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" +
                d.page_content for i, d in enumerate(docs)]
        )
    )
    return


def qa(query: str) -> str:
    docs = db.similarity_search(query=query)
    pretty_print_docs(docs)
    summary = summmary_chain.run({"query": query, "context": docs})
    return summary


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OpenAI(temperature=0, model="text-davinci-003", max_tokens=2000)
summary_prompt = PromptTemplate(
    template="Context:\n{context}\n\n"
    "Instruction: Using the context above, write a concise summary of the text as it relates to the query below."
    "Make sure to include the source of the information. The source is indicated by [Page: X from Y.pdf]."
    "Answer step-by-step.\nQuery:\n{query}\n\nSummary: ",
    input_variables=["context", "query"],
)

summmary_chain = LLMChain(
    llm=OpenAI(temperature=0, max_tokens=1048), prompt=summary_prompt
)


if __name__ == "__main__":
    db = Database.load('manual_db')
    while True:
        query = input("Enter query: ")
        if query == "exit":
            break
        if query == "update":
            db.update_database()
        if query == "new":
            pdf_path = input("Enter PDF path: ")
            db = Database.from_pdf(pdf_path, embeddings=embeddings, llm=llm)
        else:
            print(qa(query))
