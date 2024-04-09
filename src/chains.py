from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper

CUSTOM_RAG_PROMPT = """
Використай наступні **надійні** елементи, для того, щоб відповісти на питання в кінці. 
Якщо вони не містять відповіді, зверни увагу на відповідь з інтернету, хоча вона може бути не надійною. 
Якщо ти не знаєш відповіді, використаши всі свої джерела, то просто скажи про це, не потрібно вигадувати відповідь.
Використовуй не більше трьох речень, та намагайся відповісти коротко та чітко.

{context}

Відповідь з інтернету: {internet}

Питання: {question}

Корисна відповідь:"""

CUSTOM_RAG_PROMPT = PromptTemplate.from_template(CUSTOM_RAG_PROMPT)


def documents_parser(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class PdfAndGoogleChain:

    def use_google_search(self, query):
        try:
            return self.search.run(query)
        except Exception as ex:
            return 'NONE'

    def __init__(self, retriever, llm_name: str = "gpt-3.5-turbo-0125"):
        self.search = GoogleSerperAPIWrapper()
        self.llm = ChatOpenAI(model=llm_name)

        self.rag_chain = (
                {"context": retriever | documents_parser, "internet": RunnableLambda(self.use_google_search),
                 "question": RunnablePassthrough()}
                | CUSTOM_RAG_PROMPT
                | self.llm
                | StrOutputParser()
        )

    def answer(self, query: str):
        return self.rag_chain.invoke(query)
