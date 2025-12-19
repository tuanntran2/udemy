from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from dotenv import load_dotenv


load_dotenv()


def load_document():
    loader = TextLoader("docs\\LangchainRetrieval.txt")
    doc = loader.load()
    print(doc)

    loader = PyPDFLoader("docs\\Excel+Course+Document.pdf")
    docs = loader.load_and_split()
    print(len(docs))

    loader = CSVLoader(
        "docs\\Movie_collection_dataset.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["Collection", "Marketin_expense", "Budget"]
        }
    )
    data = loader.load()
    print(data[0])

    loader = DirectoryLoader("docs", glob="**/*.txt", show_progress=True)
    docs = loader.load()
    print(len(docs))


def create_vectorstore(doc_chunks, embeddings_model):
    """
    Create a Chroma vectorstore from document chunks and an embeddings model.

    Args:
        doc_chunks (List[Document]): List of document chunks to be embedded.
        embeddings_model (BaseEmbeddings): Embeddings model to convert text to vectors.

    Returns:
        Chroma: A Chroma vectorstore containing the embedded documents.

    Examples:
        Using HuggingFaceInstructEmbeddings
        >>> from langchain_community.embeddings import HuggingFaceInstructEmbeddings
        >>> from langchain_chroma import Chroma
        >>> embeddings_model = HuggingFaceInstructEmbeddings()
        >>> doc_chunks = [...]  # List of Document objects
        >>> vectorstore = create_vectorstore(doc_chunks, embeddings_model)

        Using OpenAIEmbeddings
        >>> from langchain_openai import OpenAIEmbeddings
        >>> from langchain_chroma import Chroma
        >>> embeddings_model = OpenAIEmbeddings()
        >>> doc_chunks = [...]  # List of Document objects
        >>> vectorstore = create_vectorstore(doc_chunks, embeddings_model)
    """
    return Chroma.from_documents(doc_chunks, embeddings_model)


def search(vectorstore, query, embeddings_model):
    """
    Perform a similarity search on a vectorstore using a query and an embeddings model.

    Args:
        vectorstore (Chroma): The Chroma vectorstore to search.
        query (str): The query string to search for.
        embeddings_model (BaseEmbeddings): The embeddings model to convert the query to a vector.

    Returns:
        List[Document]: A list of documents that are most similar to the query.
    """
    query_vector = embeddings_model.embed_query(query)
    return vectorstore.similarity_search_by_vector(query_vector)


if __name__ == "__main__":
    loader = TextLoader("docs\\vacation.md")
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        length_function=len,
    )

    doc_chunks = text_splitter.split_documents(doc)

    text_chunks = [doc_chunk.page_content for doc_chunk in doc_chunks]

    embeddings_model = OpenAIEmbeddings()
    vectors = embeddings_model.embed_documents(text_chunks)

    embeddings_model = HuggingFaceInstructEmbeddings()
    # vectors2 = embeddings_model2.embed_documents(text_chunks)

    # embedded_query = embeddings_model.embed_query(
    #     "what date is the flight to Beijing?"
    # )

    # db = Chroma.from_documents(doc_chunks, embeddings_model)
    db = FAISS.from_documents(doc_chunks, embeddings_model)

    result = db.similarity_search("what date is the flight to Beijing?")
    print(result)

    # embedding_vector = embeddings_model.embed_query(
    #     "what date is the flight to Beijing?"
    # )
    # result2 = db.similarity_search_by_vector(embedding_vector)
    # print(result2)
