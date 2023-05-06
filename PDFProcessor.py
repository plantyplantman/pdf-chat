import os
import re
import urllib.request
from typing import List
import fitz
import glob
from langchain.schema import Document

os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


class PDFProcessor:
    def __init__(self, path: str):
        self.path = path

    def download_pdf(self, url: str, output_path: str) -> None:
        urllib.request.urlretrieve(url, output_path)

    def preprocess(self, data: List[Document] | str) -> List[Document] | str:
        if not data:
            return data
        if isinstance(data, str):
            data = re.sub("\s+", " ", data)
            data = data.replace("\n", " ")
            return data
        if isinstance(data[0], str):
            data = re.sub("\s+", " ", data)
            data = data.replace("\n", " ")
            return data
        else:
            for d in data:
                d.page_content = re.sub("\s+", " ", d.page_content)
                d.page_content = d.page_content.replace("\n", " ")
            return data

    def pdf_to_text(self, start_page: int = 1, end_page: int = 0) -> List[str]:
        try:
            doc = fitz.open(self.path)
        except OSError:
            print(f"Error: could not open file {self.path}")
            return []
        total_pages: int = doc.page_count

        if end_page == 0:
            end_page = total_pages

        text_list = []

        for i in range(start_page - 1, end_page):
            try:
                text = doc.load_page(i).get_text("text")
                text = self.preprocess(text)
                text_list.append(text)
            except Exception:
                print(
                    "Error: could not extract text from page"
                    f"{i+1} in file {self.path}")

        doc.close()
        return text_list

    def text_to_chunks(
        self, texts: List[str], word_length: int = 300, start_page: int = 1
    ) -> List[str]:
        text_toks = [t.split(" ") for t in texts]
        chunks = []

        for idx, words in enumerate(text_toks):
            for i in range(0, len(words), word_length):
                chunk = words[i: i + word_length]
                if (
                    (i + word_length) > len(words)
                    and (len(chunk) < word_length)
                    and (len(text_toks) != (idx + 1))
                ):
                    text_toks[idx + 1] = chunk + text_toks[idx + 1]
                    continue
                chunk = " ".join(chunk).strip()
                chunk = f"[Page: {idx+start_page} from {self.path}]" + \
                    " " + '"' + chunk + '"'
                chunks.append(chunk)
        return chunks

    def process(self) -> List[Document]:
        documents: List[Document] = []
        texts = self.pdf_to_text()
        data = self.text_to_chunks(texts)
        for text in data:
            documents.append(Document(page_content=text))
        print(f"Processed {len(documents)} documents from {self.path}")
        return documents

    @staticmethod
    def flatten_array(arr):
        result = []
        for i in arr:
            if isinstance(i, list):
                result.extend(PDFProcessor.flatten_array(i))
            else:
                result.append(i)
        return result

    @staticmethod
    def embed_directory(path: str) -> List[Document]:
        documents: List[Document] = []
        pdf_files = glob.glob(f"{path}/*.pdf")

        for pdf_file in pdf_files:
            processor = PDFProcessor(pdf_file)
            documents.append(processor.process())

        documents = PDFProcessor.flatten_array(documents)

        print(f"Processed {len(documents)} documents from {path}")
        return documents
