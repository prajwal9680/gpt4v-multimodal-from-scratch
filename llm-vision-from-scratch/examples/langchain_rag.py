from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool

# Sample knowledge base
documents = [
    "Milk spoils above 10 degree Celsius",
    "Milk should be stored between 2 and 4 degree Celsius",
    "High humidity increases bacterial growth in milk",
    "Bacteria grow rapidly at high temperature"
]

# Build vector store + RAG chain
vectordb = Chroma.from_texts(documents, OpenAIEmbeddings())
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever()
)

# Tool for agent
def search_knowledge_base(query):
    docs = vectordb.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

tools = [
    Tool(
        name="KnowledgeBaseSearch",
        func=search_knowledge_base,
        description="Use this to search the knowledge base for relevant facts."
    )
]

# Zero-shot ReAct agent
agent = initialize_agent(
    tools,
    OpenAI(),
    agent="zero-shot-react-description",
    verbose=True
)

if __name__ == "__main__":
    print("RAG answer:")
    print(qa_chain.run("What happens to milk stored at 15 degree Celsius?"))

    print("\nAgent answer:")
    print(agent.run("Search knowledge base and tell what happens if temperature is increased"))
