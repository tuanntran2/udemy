from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    DirectoryLoader,
    WebBaseLoader
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from dotenv import load_dotenv
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


class Rag:
    def __init__(
        self,
        embedding_model: Embeddings = OpenAIEmbeddings(),
        vector_store_cls=Chroma,
    ):
        """
        Initialize the RAG system.
        Args:
            embedding_model (Embeddings):
                The embedding model to use for vectorization.
                There are many embedding models available
                * OpenAIEmbeddings
                * HuggingFaceInstructEmbeddings
                * SentenceTransformersEmbeddings
                * VertexAIEmbeddings
                * CohereEmbeddings
                * BedrockEmbeddings
                * OllamaEmbeddings
                * JinaEmbeddings
                * VoyagerEmbeddings
                ...
            vector_store_cls:
                The vector store class to use for storing embeddings.
                There are many vector store classes available
                * Chroma
                * FAISS
                * Pinecone
                * Weaviate
                * Qdrant
                * Milvus
                ...
        """
        load_dotenv()
        self._embedding_model = embedding_model
        self._vector_store = None
        self._vector_store_cls = vector_store_cls

    def load(
        self,
        sources: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 20,
    ):
        if sources is None or len(sources) == 0:
            raise ValueError("No data sources provided")

        # Load Documents
        documents: List[Document] = []
        for source in sources:
            documents.extend(self._load_document(source))

        # Create Chunks
        chunks = self._create_chunks(
            documents,
            chunk_size,
            chunk_overlap
        )

        # Create Vector Store
        self._vector_store = self._vector_store_cls.from_documents(
            documents=chunks,
            embedding=self._embedding_model,
        )

    def _load_document(self, source: str) -> List[Document]:
        if Path(source).is_file():
            if source.lower().endswith((".txt", ".md")):
                loader = TextLoader(source)
            elif source.lower().endswith(".pdf"):
                loader = PyPDFLoader(source)
            elif source.lower().endswith(".csv"):
                loader = CSVLoader(source)
            elif source.lower().endswith((".json", ".jsonl")):
                loader = JSONLoader(source)
        elif Path(source).is_dir():
            loader = DirectoryLoader(
                source,
                glob="**/*.{txt,md,pdf,csv,json,jsonl}")
        elif source.startswith("http"):
            loader = WebBaseLoader(source)
        else:
            raise ValueError("Unsupported file type")

        return loader.load()

    def _create_chunks(
        self,
        documents: list,
        chunk_size: int = 1000,
        chunk_overlap: int = 20
    ):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)

    @property
    def retriever(self):
        if self._vector_store is None:
            raise ValueError(
                "Vector store is not initialized. Call load() first."
            )
        return self._vector_store.as_retriever()

    def search(self, query: str, model=ChatOpenAI()):
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        template = """
            Use the following pieces of context to answer the question
            at the end.
            {context}
            Question: {question}
            Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            } | prompt | model | StrOutputParser()
        )

        return chain.invoke(query)


def test_rag_with_default():
    rag = Rag()
    rag.load(
        sources=[
            "https://www.espn.com/nba/scoreboard",
        ],
        chunk_size=500,
        chunk_overlap=50,
    )

    query = "which teams won last night NBA games?"
    answer = rag.search(query)

    print("\n--- RAG ANSWER ---")
    print(answer)


def test_rag_with_huggingface():
    from langchain_community.embeddings import HuggingFaceInstructEmbeddings

    # requires: langchain-huggingface package
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
    )
    rag = Rag(embedding_model=embedding_model)
    rag.load(
        sources=[
            "https://www.espn.com/nba/scoreboard",
        ],
        chunk_size=500,
        chunk_overlap=50,
    )

    query = "which teams won last night NBA games?"
    answer = rag.search(query)

    print("\n--- RAG ANSWER WITH HUGGINGFACE EMBEDDINGS---")
    print(answer)


if __name__ == "__main__":
    # test_rag_with_default()
    test_rag_with_huggingface()
