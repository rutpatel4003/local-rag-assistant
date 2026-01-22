from pdf_loader import File
from data_ingestor import ingest_files, expand_to_parents
from typing import List, TypedDict, Iterable, Literal
from enum import Enum
from config import Config
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from pdf_loader import File
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import json
import re

close_tag = '</think>'
tag_length = len(close_tag)

@dataclass 
class SourcesEvent: 
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

class State(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer: str
    retry_count: int 

class QueryType(Enum):
    STANDALONE = "standalone"       # new topic, search directly
    FOLLOWUP = "followup"           # references history, needs condensing
    CLARIFICATION = "clarification" # asking about previous answer
    CHITCHAT = "chitchat" 

SYSTEM_PROMPT = """
You are a highly precise technical expert. Answer the question using ONLY the provided context.
- START your answer immediately with the facts.
- DO NOT use filler phrases like "Based on the context" or "According to the excerpts."
- If the answer is not in the context, state: "Information not found in document."
- Format math using LaTeX.
- Do NOT use outside knowledge.
""".strip()

PROMPT = """
Here's the information you have about the excerpts of the files:

<context>
{context}
</context>

One file can have multiple excerpts.

Please respond to the query below

<question>
{question}
</question>

Answer:
"""

FILE_TEMPLATE = """
<file>
    <name>{name}</name>
    <content>{content}</content>
</file>
""".strip()

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical document author. Given a question, write a short paragraph (3-5 sentences) that would appear in a document answering this question.

RULES:
- Write as if you are the document, not answering directly
- Include technical terms and specific details that would appear in the actual document
- Do NOT say "The document explains..." — just write the content itself
- Keep it factual and dense with keywords

Example:
Question: "What is the time complexity of quicksort?"
Output: "Quicksort has an average-case time complexity of O(n log n) and a worst-case complexity of O(n²). The algorithm uses a divide-and-conquer approach, selecting a pivot element and partitioning the array. Space complexity is O(log n) due to recursive stack frames."
"""),
    ("human", "{question}"),
])

GRADER_SYSTEM_PROMPT = """
You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
The question does not have to be exactly the same or too specific when checking it with the context for it to be relevant, it can be a broad question with similar semantic meaning too if it makes sense. For example, a question might be based for mathematical reasoning, so check if the context contains the mathematical terms, do not just discard it if it looks gibberish without proper reasoning.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            SYSTEM_PROMPT
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', PROMPT)
    ]
)

class Role(Enum):
    USER = 'user'
    ASSISTANT = 'assistant'

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class ChunkEvent:
    content: str

@dataclass
class SourcesEvent:
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

def _remove_thinking_from_message(message: str) -> str:
    # handle cases where the tag might not exist
    if close_tag in message:
        # find the end of the tag and then .lstrip() to remove 
        return message[message.find(close_tag) + tag_length:].lstrip()
    return message.strip()

def create_history(welcome_message: Message) -> List[Message]:
    return [welcome_message]

class Chatbot:
    def __init__(self, files: List[File]):
        self.files = files
        self.retriever = ingest_files(files)
        self.llm = ChatOllama(model=Config.Model.NAME,
                            temperature=Config.Model.TEMPERATURE,
                            num_ctx=4096,
                            num_predict=1028, 
                            num_thread=8,      
                            verbose=False,
                            keep_alive=-1,
                            streaming=True,
                            callbacks=[StreamingStdOutCallbackHandler()])
        self.workflow = self._create_workflow()

    def _format_docs(self, docs: List[Document], max_chars_per_doc: int = 2000) -> str:
        """
        Format documents for LLM context - clean and truncate
        """
        formatted = []
        for doc in docs:
            content = doc.page_content
            
            # Remove page markers
            content = re.sub(r'--- PAGE \d+ ---', '', content)
            
            # Clean context markers but keep the context text
            content = content.replace('[CONTEXT]', '').replace('[/CONTEXT]', '')
            content = content.replace('[TABLE:', '[TABLE:').replace('[/TABLE]', '[/TABLE]')
            
            # Truncate if too long
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc] + "\n[...truncated...]"
            
            content = content.strip()
            
            formatted.append(FILE_TEMPLATE.format(
                name=doc.metadata.get('source', 'Unknown'),
                content=content
            ))
        
        return "\n\n".join(formatted)

    # def _format_docs(self, docs: List[Document]) -> str:
    #     return "\n\n".join(FILE_TEMPLATE.format(name=doc.metadata['source'], content=doc.page_content) for doc in docs)
    
    def _retrieve(self, state: State):
        print(f"RETRIEVING: {state['question']}")
        context = self.retriever.invoke(state['question'])
        if Config.Preprocessing.ENABLE_PARENT_CHILD:
            original_count = len(context)
            context = expand_to_parents(context)
            print(f"  Expanded {original_count} children → {len(context)} parents")
        
        return {"context": context}

    def _hyde_retrieve(self, state: State):
        """
        HyDE: Hypothetical Document Embeddings Retrieval
        1. Generate a hypothetical document that would answer the question
        2. Embed that hypothetical document
        3. Retrieve using hypothetical embedding
        """
        question = state['question']
        print(f'HyDE RETRIEVING: {question}')

        try:
            # generate hypothetical document
            hyde_chain = HYDE_PROMPT | self.llm | StrOutputParser()
            hypothetical_doc = hyde_chain.invoke({'question': question})
            # clean thinking tags if any
            if '</think>' in hypothetical_doc:
                hypothetical_doc = hypothetical_doc.split('</think>')[-1].strip()
            
            print(f"HyDE generated: {hypothetical_doc[:100]}...")

            # retrieve using hypothetical document as query
            # the retriever will embed this hypothetical doc and find similar real docs
            context = self.retriever.invoke(hypothetical_doc)
            
            # normal retrieval and merge (optional boost)
            normal_context = self.retriever.invoke(question)
            
            # merge and deduplicate, prioritizing HyDE results
            seen_content = set()
            merged = []
            for doc in context + normal_context:
                if doc.page_content not in seen_content:
                    merged.append(doc)
                    seen_content.add(doc.page_content)
            if Config.Preprocessing.ENABLE_PARENT_CHILD:
                merged = expand_to_parents(merged)
            
            # limit to configured max
            merged = merged[:Config.Chatbot.N_CONTEXT_RESULTS * 2]
            
            print(f"   HyDE retrieved {len(context)} + normal {len(normal_context)} = {len(merged)} unique docs")
            
            return {"context": merged}
            
        except Exception as e:
            print(f"HyDE failed ({e}), falling back to normal retrieval")
            context = self.retriever.invoke(question)
            return {"context": context}
    
    def _grade_documents(self, state: State):
        """
        Filter out irrelevant documents.
        """
        print('Document Relevance Checking in Process!')
        question = state['question']
        documents = state['context']
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:500]}" for i, doc in enumerate(documents) 
        ]) # batching documents to query faster
        prompt = ChatPromptTemplate.from_messages([
            ('system', """You are a document grader. For each document, decide if it's relevant.
            Return a JSON Array with the format defined below:
            [
                {{"doc_id": 1, "relevant": true}},
                {{"doc_id": 2, "relevant": false}}
            ]
            Use double braces in the response.
            """),
            ('human', 'Question: {question}\n\nDocuments:\n{docs_text}'),
        ])
        grader_chain = prompt | self.llm | StrOutputParser()
        try:
            result = grader_chain.invoke({'question': question, 'docs_text': docs_text})
            scores = json.loads(result)
            score_map = {item['doc_id']: item.get('relevant', False) for item in scores}    
            filtered_docs = [
                doc for i, doc in enumerate(documents) 
                if score_map.get(i + 1, False) # default to False (remove) if LLM didn't mention it
            ]
            
            print(f"Filtered: {len(filtered_docs)}/{len(documents)} relevant")
            return {'context': filtered_docs}
        except Exception as e:
            print(f'Grading failed: {e}')
            return {'context': documents}
    
    def _transform_query(self, state: State):
        """
        Transform the query to produce a better question
        """
        print("Transforming Query!")
        question = state['question']
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are generating a better search query for a vector database. Rephrase the input question to be more specific. Just rephrase the input question, do not add any preamble or explanation, your output should only contain the rephrased question, nothing else."),
            ("human", f"Look at the input and try to reason about the underlying semantic intent / meaning. \n\n Initial Question: {question} \n\n Formulate an improved question: "),
        ])
        chain = prompt | self.llm | StrOutputParser()
        better_question = chain.invoke({})
        current_retry = state.get('retry_count', 0)
        return {'question': better_question, 'retry_count': current_retry+1}

    def _generate(self, state: State):
        print('Generating Answer!')
        chat_history = state['chat_history']
        if len(chat_history) <= 1: 
            chat_history = []
        messages = PROMPT_TEMPLATE.invoke(
            {
                "question": state['question'],
                "context": self._format_docs(state['context']),
                'chat_history': chat_history,
            }
        )
        answer = self.llm.invoke(messages)
        return {"answer": answer}
    
    def _decide_to_generate(self, state: State) -> Literal['_transform_query', '_generate']:
        """
        Determines whether to generate an answer or re-generate a question.
        """
        filtered_documents = state['context']
        retry_count = state.get('retry_count', 0)
        # if no relevant documents and not self-corrected a lot of times
        if not filtered_documents and retry_count<1: # loop 1 time
            print('Decision: Documents irrelevant, rerouting to transform!')
            return '_transform_query'
        else:
            print('Decision: Generating Answer!')
            return '_generate'

    def _classify_query(self, question: str, chat_history: List[BaseMessage]) -> QueryType:
        """
        Classify query type to determine processing path
        """
        if len(chat_history) <= 1:
            return QueryType.STANDALONE
        
        # get recent context for classification
        recent_context = ""
        for m in chat_history[-2:]:
            if isinstance(m, HumanMessage):
                recent_context += f'User asked: {m.content[:100]}\n'
            elif isinstance(m, AIMessage) and 'how can I help' not in m.content.lower():
                recent_context += f'Assistant answered about: {m.content[:100]}\n'

        classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify the query into ONE category:

STANDALONE - Question contains ALL information needed to answer it:
  - Mentions specific table numbers, section names, or topics explicitly
  - Examples: "What is table 3.1?", "What are preprocessing options for NLTK in table 3.2?", "Explain PEFT"
  
FOLLOWUP - Question CANNOT be understood without prior conversation:
  - Uses pronouns (it, they, this, that) referring to unknown subject
  - Examples: "What about the other one?", "How does it work?", "And the second table?"

CLARIFICATION - Asks to expand on previous answer:
  - Examples: "Can you explain more?", "Give an example", "What do you mean?"

CHITCHAT - Greeting or thanks:
  - Examples: "Thanks", "Hello", "Great"

IMPORTANT: If the question mentions specific names, tables, or topics explicitly, it is STANDALONE even if it relates to prior discussion.

Output ONLY: STANDALONE or FOLLOWUP or CLARIFICATION or CHITCHAT"""),
        ("human", f"Recent conversation:\n{recent_context}\n\nNew query: {question}\n\nClassification:"),
    ])
        
        try:
            result = self.llm.invoke(classification_prompt.format_messages()).content.strip().upper()
            if '</think>' in result:
                result = result.split('</think>')[-1].strip().upper()

            if 'FOLLOWUP' in result:
                return QueryType.FOLLOWUP
            elif 'CLARIFICATION' in result:
                return QueryType.CLARIFICATION
            elif 'CHITCHAT' in result:
                return QueryType.CHITCHAT
            return QueryType.STANDALONE
    
        except Exception as e:
            print(f'Classification failed: {e}')
            return QueryType.STANDALONE
    
    def _condense_question(self, state: State):
        """
        Smart routing: classify query type, then process accordingly.
        """
        question = state['question']
        chat_history = state.get('chat_history', [])
        
        # filter out welcome message
        real_history = [m for m in chat_history 
                    if not (isinstance(m, AIMessage) and "how can I help" in m.content.lower())]
        
        if not Config.Chatbot.ENABLE_QUERY_ROUTER:
             return {"question": question}
        
        # classify the query
        query_type = self._classify_query(question, real_history)
        print(f"ROUTER: '{question[:50]}...' -> {query_type.value}")
        
        # Route based on classification
        if query_type == QueryType.STANDALONE:
            return {"question": question}
        
        if query_type == QueryType.CHITCHAT:
            return {"question": question}
        
        if query_type in [QueryType.FOLLOWUP, QueryType.CLARIFICATION]:
            # condense with recent context only
            recent_history = real_history[-6:]  
            
            condense_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant.

TASK:
Rewrite the user's latest message into a complete, standalone QUESTION using the chat history.

RULES:
- Output MUST be a single question ending with a '?'.
- Do NOT answer the question.
- Do NOT output explanations, definitions, or extra text—ONLY the rewritten question.
- Keep it short and specific.
- If you cannot think of any context related question, just output the EXACT same question asked by the user, nothing else.

GOOD:
History:
User: what is the formula of positive definite matrix?
AI: ...
User input: and for positive semi definite matrix?
Output: What is the formula of a positive semidefinite matrix?

BAD:
Output: A symmetric matrix A is positive semidefinite if x^T A x >= 0 for all x.
"""),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human", "{question}"),
            ])
            
            try:
                chain = condense_prompt | self.llm | StrOutputParser()
                reformulated = chain.invoke({
                    "chat_history": recent_history, 
                    "question": question
                }).strip()
                
                # clean thinking tags
                if '</think>' in reformulated:
                    reformulated = reformulated.split('</think>')[-1].strip()
                
                # if the model produced a statement (no '?'), treat it as a search query
                if not reformulated.endswith('?'):
                    print(f"CONDENSE: Output '{reformulated}' has no '?', appending one.")
                    reformulated += "?"

                print(f"CONDENSE: '{question}' -> '{reformulated}'")
                return {"question": reformulated}
                
            except Exception as e:
                print(f"CONDENSE: Failed ({e}), using original")
                return {"question": question}
        
        return {"question": question}
    
    def _create_workflow(self) -> CompiledStateGraph:
        graph_builder = StateGraph(State)
        retrieve_fn = self._hyde_retrieve if Config.Chatbot.ENABLE_HYDE else self._retrieve
        
        if not Config.Chatbot.GRADING_MODE:
            graph_builder.add_node('_condense_question', self._condense_question)
            graph_builder.add_node('_retrieve', retrieve_fn)
            graph_builder.add_node('_generate', self._generate)
            
            graph_builder.add_edge(START, '_condense_question')
            graph_builder.add_edge('_condense_question', '_retrieve')
            graph_builder.add_edge('_retrieve', '_generate')
            graph_builder.add_edge('_generate', END)
        else:
            graph_builder.add_node('_condense_question', self._condense_question)
            graph_builder.add_node('_retrieve', retrieve_fn)
            graph_builder.add_node('_grade_documents', self._grade_documents)
            graph_builder.add_node('_transform_query', self._transform_query)
            graph_builder.add_node('_generate', self._generate)
            
            graph_builder.add_edge(START, '_condense_question')
            graph_builder.add_edge('_condense_question', '_retrieve')
            graph_builder.add_edge('_retrieve', '_grade_documents')
            graph_builder.add_conditional_edges(
                '_grade_documents', self._decide_to_generate,
                {
                    '_transform_query': '_transform_query',
                    '_generate': '_generate'
                },
            )
            graph_builder.add_edge('_transform_query', '_retrieve')
            graph_builder.add_edge('_generate', END)
        
        return graph_builder.compile()

    def _ask_model(
            self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        history = [
            AIMessage(m.content) if m.role == Role.ASSISTANT else HumanMessage(m.content) for m in chat_history
        ]
        payload = {"question": prompt, "chat_history": history, 'retry_count': 0}

        config = {
            "configurable": {"thread_id": 42}
        }

        for event_type, event_data in self.workflow.stream(
            payload, config=config, stream_mode=['updates', 'messages']
        ):
            if event_type =='messages':
                chunk, metadata = event_data
                if metadata.get('langgraph_node') == '_generate':
                    if chunk.content:
                        yield ChunkEvent(chunk.content)
            if event_type == 'updates':
                if "_retrieve" in event_data:
                    documents = event_data['_retrieve']['context']
                    unique_docs = []
                    seen_content = set()
                    for doc in documents:
                        if doc.page_content not in seen_content:
                            unique_docs.append(doc)
                            seen_content.add(doc.page_content)
                    yield SourcesEvent(unique_docs)
                if "_generate" in event_data:
                    answer = event_data['_generate']['answer']
                    yield FinalAnswerEvent(answer.content)

    def ask(self, prompt: str, chat_history: List[Message]) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        for event in self._ask_model(prompt, chat_history):
            yield event
            if isinstance(event, FinalAnswerEvent):
                response = _remove_thinking_from_message("".join(event.content))
                chat_history.append(Message(role=Role.USER, content=prompt))
                chat_history.append(Message(role=Role.ASSISTANT, content=response))
