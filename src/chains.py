from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import List

from operator import itemgetter

CUSTOM_RAG_PROMPT = """
Використай наступні **надійні** елементи, для того, щоб вибрати відповідь на питання з запропонованих.
Якщо вони не містять відповіді, зверни увагу на інформацію з інтернету. 
Якщо ти не знаєш відповіді, використаши всі свої джерела, то просто скажи про це, не потрібно вигадувати відповідь.
Напиши у відповіді номер правильного варіанту відповіді. Якщо серед варантів немає правильної відповіді, напиши коротко відповідь самостійно.


{context}

Відповідь з інтернету: {internet}

Приклад: 

Чия типологія поділяється на традиційні, харизматичні й раціональні системи?
1) Вебер 
2) Ленін
3) Сталін
4) Обама

Правильна відповідь: 1 - Вебер. 


Питання: {question}

Варіанти відповіді: 
{options}

Правильна відповідь:"""

CUSTOM_RAG_PROMPT = PromptTemplate.from_template(CUSTOM_RAG_PROMPT)

VERIFICATION_PROMPT = """
Вам було задано наступне питання: 
{question}
З варіантами відповіді: 
{options}
На яку було запропоновано відповідь:
{answer}

Повторіть відповідь, якщо вона правильна. Інакше, скажіть "відповідь відсутня".

Відповідь:
"""
VERIFICATION_PROMPT = PromptTemplate.from_template(VERIFICATION_PROMPT)


def documents_parser(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def prepare_options(options):
    return "\n".join([f"{i + 1}) {option}" for i, option in enumerate(options)])

def flatten_input(d):
    ret = d.pop('input')
    ret.update(d)
    return ret

class PdfAndGoogleChain:

    def use_google_search(self, query):
        try:
            return self.search.run(query)
        except Exception as ex:
            return 'NONE'

    def retrieve_multiple(self, query_dict):
        query = query_dict['query']
        options = query_dict['options']
        ret = self.retriever.get_relevant_documents(query)
        for option in options:
            ret.extend(self.retriever.get_relevant_documents(option)[:2])
        return ret

    def __init__(self, retriever, llm_name: str = "gpt-3.5-turbo-0125"):
        self.search = GoogleSerperAPIWrapper()
        self.retriever = retriever
        self.llm = ChatOpenAI(model=llm_name)

        self.rag_chain = (
                {"context": RunnableLambda(self.retrieve_multiple) | documents_parser,
                 "internet": itemgetter("query") | RunnableLambda(self.use_google_search),
                 "question": itemgetter("query") | RunnablePassthrough(),
                 "options": itemgetter("options") | RunnableLambda(prepare_options)}
                | CUSTOM_RAG_PROMPT
                | self.llm
                | StrOutputParser()
        )

        self.verification_chain = (
                {"question": itemgetter("query") | RunnablePassthrough(),
                 "options": itemgetter("options") | RunnableLambda(prepare_options),
                 "answer": itemgetter("answer") | RunnablePassthrough()}
                | VERIFICATION_PROMPT
                | self.llm
                | StrOutputParser()
        )

        self.global_chain = (RunnableParallel(input=RunnablePassthrough(), answer=self.rag_chain)
                             | RunnableLambda(flatten_input)
                             | self.verification_chain)

    def answer(self, query: str, options: List[str]):
        options = list(filter(lambda x: x is not None and len(x) > 0, options))
        return self.global_chain.invoke({"query": query, "options": options})

