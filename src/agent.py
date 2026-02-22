"""
Construction du LLM Groq et de l'agent ReAct LangGraph.
"""

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from src.tools import square_root, web_search
from src.config import logger


def build_llm() -> ChatGroq:
    """Instancie et retourne le modèle Groq."""
    logger.info("Initialisation du modèle LLM (Groq)...")
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )


def build_agent(llm: ChatGroq):
    """Construit et retourne l'agent ReAct avec ses outils."""
    logger.info("Construction de l'agent ReAct...")
    return create_react_agent(
        model=llm,
        tools=[square_root, web_search],
        prompt=(
            "Tu es un assistant expert polyvalent. "
            "Tu réponds toujours en français, de manière claire et structurée. "
            "Utilise les outils disponibles quand c'est nécessaire."
        ),
    )
