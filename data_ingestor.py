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
from langchain_chroma import Chroma
import hashlib
import re
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

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

# def create_llm() -> ChatOllama:
#     return ChatOllama(model=Config.Preprocessing.LLM, temperature=0, keep_alive=-1)

def create_reranker():
    """
    Create a heavy-duty reranker 
    """
    model_name = Config.Preprocessing.RERANKER
    model_kwargs = {'device': 'cuda'} 
    
    # Initialize the model
    model = HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    
    return CrossEncoderReranker(model=model, top_n=Config.Chatbot.N_CONTEXT_RESULTS)

def create_embeddings():
    """
    Load embedding model for high-quality, long-context embeddings.
    """
    model_name = Config.Preprocessing.EMBEDDING_MODEL
    model_kwargs = {"device": "cuda", 'trust_remote_code': True} # Force GPU
    encode_kwargs = {"normalize_embeddings": True} # Recommended for BGE models

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

# def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
#     messages = CONTEXT_PROMPT.format_messages(document=document, chunk=chunk)
#     response = llm.invoke(messages)
#     return response.content

def _detect_content_type(text: str) -> str:
    """
    Content type detection based on markers
    """
    if "[TABLE:" in text and "[/TABLE]" in text:
        return 'table'
    if "[FIGURE]" in text and "[/FIGURE]" in text:
        return 'figure'
    return 'text'

def _create_chunks(document: Document) -> List[Document]:
    """
    Create chunks - tables/figures get surrounding context, text gets split normally.
    FIXED: Tables/figures include nearby text for better retrieval.
    """
    content = document.page_content
    
    # Pattern to find tables and figures
    pattern = r'(\[TABLE:.*?\[/TABLE\]|\[FIGURE\].*?\[/FIGURE\])'
    
    # Find all structured content with their positions
    structured_matches = list(re.finditer(pattern, content, flags=re.DOTALL))
    
    if not structured_matches:
        # No tables/figures - just split normally
        chunks = text_splitter.create_documents(
            [content],
            metadatas=[{**document.metadata, 'content_type': 'text'}]
        )
        return chunks
    
    final_chunks = []
    last_end = 0
    
    for match in structured_matches:
        start, end = match.start(), match.end()
        structured_content = match.group()
        
        # Get text BEFORE this table/figure
        text_before = content[last_end:start].strip()
        
        # Get some context AFTER (look ahead up to 500 chars or next structure)
        context_after_end = min(end + 500, len(content))
        next_match = re.search(pattern, content[end:context_after_end], flags=re.DOTALL)
        if next_match:
            context_after_end = end + next_match.start()
        text_after = content[end:context_after_end].strip()
        
        # Chunk the text before (if substantial)
        if text_before and len(text_before) > 50:
            text_chunks = text_splitter.create_documents(
                [text_before],
                metadatas=[{**document.metadata, 'content_type': 'text'}]
            )
            final_chunks.extend(text_chunks)
        
        # Determine content type
        content_type = 'table' if '[TABLE:' in structured_content else 'figure'
        
        # Create chunk for table/figure WITH surrounding context
        # Include last 200 chars of text_before + table + first 200 chars of text_after
        context_before = text_before[-200:] if len(text_before) > 200 else text_before
        context_after_snippet = text_after[:200] if len(text_after) > 200 else text_after
        
        chunk_with_context = ""
        if context_before:
            chunk_with_context += f"[CONTEXT]\n{context_before}\n[/CONTEXT]\n\n"
        chunk_with_context += structured_content
        if context_after_snippet:
            chunk_with_context += f"\n\n[CONTEXT]\n{context_after_snippet}\n[/CONTEXT]"
        
        final_chunks.append(Document(
            page_content=chunk_with_context,
            metadata={**document.metadata, 'content_type': content_type}
        ))
        
        last_end = end
    
    # Handle remaining text after last table/figure
    remaining_text = content[last_end:].strip()
    if remaining_text and len(remaining_text) > 50:
        text_chunks = text_splitter.create_documents(
            [remaining_text],
            metadatas=[{**document.metadata, 'content_type': 'text'}]
        )
        final_chunks.extend(text_chunks)
    
    return final_chunks

def _calculate_file_hash(content: str) -> str:
    """
    Calculate hash of file content for deduplication
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def ingest_files(files: List[File]) -> BaseRetriever:
    """
    Ingests into a Persistent Vector Database (Chroma)
    Enhanced with table intelligence
    """
    # initialize embeddings
    embedding_model = create_embeddings()

    # connect to persistent db on disk
    vector_store = Chroma(
        collection_name='private-rag',
        embedding_function=embedding_model,
        persist_directory=str(Config.Path.VECTOR_DB_DIR)
    )

    # check for duplicated files
    try:
        existing_data = vector_store.get()
        existing_sources = {}
        if existing_data and 'metadatas' in existing_data:
            for m in existing_data['metadatas']:
                if m and 'source' in m:
                    source_name = m['source']
                    file_hash = m.get('content_hash', None)
                    existing_sources[source_name] = file_hash
        
        table_count = sum(1 for m in existing_data.get('metadatas', []) 
                         if m and m.get('content_type') == 'table')
        
        print(f'Found {len(existing_sources)} files in database ({table_count} table chunks)')
        print(f'Files: {list(existing_sources.keys())}')
    except Exception as e:
        print(f'Error reading database: {e}')
        existing_sources = {}

    # filter new files only
    new_chunks = []
    skipped_files = []

    for f in files:
        file_hash = _calculate_file_hash(f.content)
        
        if f.name in existing_sources:
            stored_hash = existing_sources[f.name]
            if stored_hash == file_hash:
                print(f'Skipping {f.name} (already indexed)')
                skipped_files.append(f.name)
                continue
            else:
                print(f'File {f.name} content changed - reprocessing')

        # process new file
        print(f"Indexing: {f.name}")
        doc = Document(
            f.content, 
            metadata={
                'source': f.name, 
                'content_hash': file_hash
            }
        )
        
        file_chunks = _create_chunks(doc)
        
        # add hash to all chunks
        for chunk in file_chunks:
            chunk.metadata['content_hash'] = file_hash
        
        # count tables
        table_chunks = sum(1 for c in file_chunks if c.metadata.get('content_type') == 'table')
        if table_chunks > 0:
            print(f"Found {table_chunks} table(s) in {f.name}")
        
        new_chunks.extend(file_chunks)

    if skipped_files:
        print(f'Loaded {len(skipped_files)} file(s) from cache')

    # add new chunks to the db only
    if new_chunks:
        print(f'Adding {len(new_chunks)} new chunks to Vector Database')
        
        # count content types
        tables = sum(1 for c in new_chunks if c.metadata.get('content_type') == 'table')
        figures = sum(1 for c in new_chunks if c.metadata.get('content_type') == 'figure')
        text = len(new_chunks) - tables - figures
        
        print(f'{text} text chunks, {tables} tables, {figures} figures')
        
        vector_store.add_documents(new_chunks)
    else:
        print('No new content to index')

    # create vector retriever
    semantic_retriever = vector_store.as_retriever(
        search_kwargs={'k': Config.Preprocessing.N_SEMANTIC_RESULTS}
    )

    # create bm25 retriever 
    db_state = vector_store.get()
    stored_texts = db_state.get('documents', [])
    stored_metadatas = db_state.get('metadatas', [])
    
    if not stored_texts:
        raise ValueError('Database is empty! Please upload a document.')
    
    # reconstruct document objects for langchain
    global_corpus = []
    for t, m in zip(stored_texts, stored_metadatas):
        safe_m = m if m else {}
        global_corpus.append(Document(page_content=t, metadata=safe_m))

    print(f'Building BM25 Index on {len(global_corpus)} total chunks')
    bm25_retriever = BM25Retriever.from_documents(global_corpus)
    bm25_retriever.k = Config.Preprocessing.N_BM25_RESULTS

    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return ContextualCompressionRetriever(
        base_compressor=create_reranker(), 
        base_retriever=ensemble_retriever
    )