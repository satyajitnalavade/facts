from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter)

print(docs)

