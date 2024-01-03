from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter)

# load it into Chroma
db = Chroma.from_documents(
    docs, 
    embedding=embeddings,
    persist_directory="emb"
)


print(docs)

