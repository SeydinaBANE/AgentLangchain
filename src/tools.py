import math
from typing import Dict, Any

from langchain.tools import tool
from tavily import TavilyClient

from src.config import logger


@tool
def square_root(x: float) -> float:
    """Calcule la racine carrée d'un nombre non négatif."""
    if x < 0:
        raise ValueError(f"Impossible de calculer la racine carrée d'un nombre négatif : {x}")
    result = math.sqrt(x)
    logger.debug("square_root(%s) = %s", x, result)
    return result


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Effectue une recherche sur le web via Tavily et retourne les résultats."""
    logger.info("Recherche web : %s", query)
    try:
        client = TavilyClient()
        return client.search(query)
    except Exception as e:
        logger.error("Erreur lors de la recherche web : %s", e)
        return {"error": str(e), "results": []}
