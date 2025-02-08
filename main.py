from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Load OpenAI embeddings and FAISS index
embeddings = OpenAIEmbeddings()
new_vectorstore = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = new_vectorstore.as_retriever(search_kwargs={"k": 3})

# Define prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, request: Request):
    return StreamingResponse(generate_chat_responses(message=message), media_type="text/event-stream")

async def generate_chat_responses(message: str):
    async for chunk in retrieval_chain.astream(message):
        content = chunk.replace("\n", "<br>")
        yield f"data: {content}\n\n"
    yield "event: end\ndata: \n\n"  # Indicate the end of the stream

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found"},
    )

# Start app with dynamic port for Render
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
