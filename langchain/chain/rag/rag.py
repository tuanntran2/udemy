from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from abc import abstractmethod
from dotenv import load_dotenv


load_dotenv()


class Rag:
    template = """
        Use the following pieces of context to answer the question at the end.
        {context}
        Question: {question}
        Answer:
    """

    def __init__(
        self,
        path: str,
        embedding_model=OpenAIEmbeddings(),
        chunk_size: int = 1000,
        chunk_overlap: int = 20,
    ):
        self._embedding_model = embedding_model
        self._documents = self.load_document(path)
        self._chunks = self.create_chunks(
            self._documents,
            chunk_size,
            chunk_overlap
        )
        self._vector_store = self.create_vector_store()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def create_vector_store(self) -> VectorStore:
        return None

    def load_document(self, path: str):
        if path.endswith(".txt") or path.endswith(".md"):
            loader = TextLoader(path)
        elif path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".csv"):
            loader = CSVLoader(path)
        elif '.' not in path:
            loader = DirectoryLoader(path)
        else:
            raise ValueError("Unsupported file type")
        return loader.load()

    def create_chunks(
        self,
        documents: list[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 20
    ):
        """
        Split documents into smaller chunks for easier processing.

        Args:
            documents (List[Document]):
                List of documents to be split.
            chunk_size (int, optional):
                Maximum size of each chunk
                Defaults: 1000.
            chunk_overlap (int, optional):
                Number of overlapping characters between chunks
                Defaults: 20.

        Returns:
            List[Document]: List of document chunks.

        Remark:
            Enhance the logic by chunking based on sentences.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        return text_splitter.split_documents(documents)

    def create_embed_documents(self, chunks: list[Document]):
        return [
            self._embedding_model.embed_documents(chunk)
            for chunk in chunks
        ]

    def create_embed_query(self, query: str):
        return self._embedding_model.embed_query(query)

    def search(self, query: str, llm=ChatOpenAI()):
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_template(self.template)
        chain = (
            {
                "context": self._vector_store.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            } | prompt | llm | StrOutputParser()
        )

        return chain.invoke(query)

    def search_with_score(self, query: str):
        return self._vector_store.similarity_search_with_score(query)

    def search_by_vector(self, vector: list[float]):
        return self._vector_store.similarity_search_by_vector(vector)

    def search_by_vector_with_score(self, vector: list[float]):
        return self._vector_store. \
            similarity_search_by_vector_with_score(vector)

    def retrieve(self, query: str, k: int = 4):
        return self.retriever.invoke(query)


class RagChroma(Rag):
    def create_vector_store(self):
        return Chroma.from_documents(
            documents=self._chunks,
            embedding=self._embedding_model,
        )


class RagFaiss(Rag):
    def create_vector_store(self):
        return FAISS.from_documents(
            documents=self._chunks,
            embedding=self._embedding_model,
        )


def test_rag():

    text_rag = RagChroma(
        "docs\\vacation.md",
        chunk_size=100,
        chunk_overlap=10
    )
    result = text_rag.search("Create summary for my vacation")
    print(result)


if __name__ == "__main__":
    test_rag()

    # ------------------------------
    # text_rag = RagChroma(
    #     "docs\\vacation.md",
    #     chunk_size=100,
    #     chunk_overlap=10
    # )
    # # result = text_rag.search("what date is the flight to Beijing")
    # result = text_rag.retrieve("what date is the flight to Beijing")
    # print(result)

    # # ------------------------------
    # text_rag = RagFaiss(
    #     "docs\\vacation.md",
    #     chunk_size=100,
    #     chunk_overlap=10
    # )
    # embed_vector = text_rag.create_embed_query(
    #     "what date is the flight to Beijing"
    # )
    # result = text_rag.search_by_vector(embed_vector)
    # print(result)

    # ------------------------------
    # pdf_rag = RagFaiss("docs\\Excel+Course+Document.pdf")
    # print(pdf_rag._documents)
    # chunks = pdf_rag.create_chunks(pdf_rag._documents)
    # print(chunks)

    # ------------------------------
    # csv_rag = Rag("docs\\SampMovie_collection_dataset.csv")
    # print(csv_rag._documents)
    # chunks = csv_rag.create_chunks(csv_rag._documents)
    # print(chunks)

    # ------------------------------
    # dir_rag = Rag("docs\\text_docs")
    # print(dir_rag._documents)
