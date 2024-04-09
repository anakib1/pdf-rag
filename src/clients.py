from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pathlib
import gdown
from .chains import PdfAndGoogleChain


def embed_pdf(folder: str = 'data', name: str = 'book.pdf'):
    pathlib.Path(folder).mkdir(exist_ok=True)
    path = pathlib.Path(folder).joinpath(name)
    if not path.exists():
        print('Downloading book PDF.')
        gdown.download('https://drive.google.com/file/d/1CwhFM4gInp9xV4G4sdnYE_rN0StmqQ2z/view?usp=drive_link',
                       str(path), fuzzy=True)
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100)
    return splitter.split_documents(
        documents
    )


class AcademicClient:

    def create_vectordb(self):
        if pathlib.Path('db').exists():
            self.vectordb = Chroma(persist_directory='db', embedding_function=OpenAIEmbeddings())
        elif pathlib.Path('src/db').exists():
            self.vectordb = Chroma(persist_directory='src/db', embedding_function=OpenAIEmbeddings())
        else:
            print('Not found cached DB. Rebuilding DB state, could use money from OPENAI!!!!')
            texts = embed_pdf()
            self.vectordb = Chroma.from_documents(
                documents=texts,
                embedding=OpenAIEmbeddings(),
                persist_directory="db"
            )
            self.vectordb.persist()

    def __init__(self):
        self.create_vectordb()
        self.chain = PdfAndGoogleChain(
            self.vectordb.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 30, "k": 6}))

    def answer(self, query, options):
        return self.chain.answer(query, options)
