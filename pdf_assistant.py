import typer
from typing import Optional, List
from phi.agent import Agent
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Database URL
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize knowledge base
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url)
)

# Load the knowledge base
knowledge_base.load()

# Initialize storage
storage = PgAgentStorage(table_name="pdf_agent", db_url=db_url)

def pdf_agent(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    # Handle existing run
    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if existing_run_ids:
            run_id = existing_run_ids[0]

    # Initialize assistant without OpenAI dependencies
    agent = Agent(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    # Manage run ID
    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    # Run CLI application
    agent.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_agent)
