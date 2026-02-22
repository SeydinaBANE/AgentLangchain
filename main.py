"""
Point d'entrée principal — boucle interactive de l'agent ReAct.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

from src.agent import build_llm, build_agent
from src.config import logger, get_langfuse_handler


def ask_agent(agent, question: str, langfuse_handler: CallbackHandler) -> str:
    """
    Envoie une question à l'agent et retourne la réponse.

    Args:
        agent: L'agent LangGraph.
        question: La question posée par l'utilisateur.
        langfuse_handler: Le handler Langfuse pour le tracking.

    Returns:
        La réponse de l'agent sous forme de chaîne de caractères.
    """
    logger.info("Question envoyée à l'agent : %s", question)
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"callbacks": [langfuse_handler]},
        )
        answer = response["messages"][-1].content
        logger.info("Réponse reçue (%d caractères)", len(answer))
        return answer
    except Exception as e:
        logger.error("Erreur lors de l'invocation de l'agent : %s", e)
        return f"❌ Une erreur s'est produite : {e}"


def run_interactive(agent, langfuse_handler: CallbackHandler) -> None:
    """Lance une boucle interactive permettant à l'utilisateur de poser des questions."""
    print("\n" + "=" * 60)
    print("  🤖  Agent ReAct — Mode interactif")
    print("  Tapez 'quitter' ou 'exit' pour arrêter.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAu revoir !")
            break

        if not question:
            continue

        if question.lower() in {"quitter", "exit", "quit", "q"}:
            print("Au revoir !")
            break

        answer = ask_agent(agent, question, langfuse_handler)
        print(f"\nAgent : {answer}\n")


if __name__ == "__main__":
    try:
        langfuse_handler = get_langfuse_handler()
        llm = build_llm()
        agent = build_agent(llm)
        run_interactive(agent, langfuse_handler)
    except Exception as e:
        logger.critical("Échec critique au démarrage : %s", e, exc_info=True)
        raise
