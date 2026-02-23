from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from src.tools import square_root, web_search
from src.config import logger


def build_llm() -> ChatGroq:
    """Instancier et retourne le modèle Groq."""
    logger.info("Initialisation du modèle LLM (Groq)...")
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )


def build_agent(llm: ChatGroq):
    """Construit et retourne l'agent ReAct avec ses outils."""
    logger.info("Construction de l'agent ReAct...")
    return create_agent(
        model=llm,
        tools=[square_root, web_search],
        system_prompt=(
            "Tu es un assistant expert polyvalent. "
            "Tu réponds toujours en français, de manière claire et structurée. "
            "Utilise les outils disponibles quand c'est nécessaire."
        ),
    )
