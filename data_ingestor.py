from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
from pdf_loader import File
CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """
You're an expert in document analysis. Your task is to provide brief, relevant context for a chunk of text from the given documents.

Here is the document:
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
1. Identify the main topic or concept discussed in the chunk.
2. Mention any relevant information or comparisons from the broader document content.
3. If applicable, note how this information relates to the overall theme or purpose of the document.
4. Include any key figures, dates, or percentages that provide important context.
5. Do not use phrases like "This chunk discusses" or "This section provided". Instead directly state the discussion.

Please give a short succint context to situate this chunk within the overall document for the purpose of improving search retrieval of the chunk.

Context:
""".strip()
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = Config.Preprocessing.CHUNK_SIZE,
    chunk_overlap = Config.Preprocessing.CHUNK_OVERLAP
)

def create_llm() -> ChatOllama:
    return ChatOllama(model=Config.Preprocessing.LLM, temperature=0, keep_alive=1)

def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model = Config.Preprocessing.RERANKER, top_n = Config.Chatbot.N_CONTEXT_RESULTS)

def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Preprocessing.EMBEDDING_MODEL)

def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
    messages = CONTEXT_PROMPT.format_messages(document=document, chunk=chunk)
    response = llm.invoke(messages)
    return response.content

def _create_chunks(document: Document) -> List[Document]:
    chunks = text_splitter.split_documents([document])
    if not Config.Preprocessing.CONTEXTUALIZE_CHUNKS:
        return chunks
    llm = create_llm()
    contextual_chunks = []
    for c in chunks:
        context = _generate_context(llm, document.page_content, c.page_content)
        chunk_with_context = f"{context}\n\n{c.page_content}"
        contextual_chunks.append(Document(page_content=chunk_with_context, metadata=c.metadata))
    return contextual_chunks


def ingest_files(files: List[File]) -> BaseRetriever:
    documents = [Document(file.content, metadata={"source": file.name}) for file in files]
    chunks = []
    for doc in documents:
        chunks.extend(_create_chunks(doc))

    if not chunks:
        raise ValueError("No text could be extracted from these files. Please check if the PDFs are empty or scanned images.")

    semantic_retriever = InMemoryVectorStore.from_documents(chunks, create_embeddings()).as_retriever(search_kwargs={"k":Config.Preprocessing.N_SEMANTIC_RESULTS})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.Preprocessing.N_BM25_RESULTS
    
    ensemble_retriever = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.6, 0.4],)

    return ContextualCompressionRetriever(
        base_compressor=create_reranker(), base_retriever=ensemble_retriever
    )