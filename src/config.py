"""
Configuration globale : logging et handler Langfuse.
"""

import logging
from langfuse.langchain import CallbackHandler


def setup_logging() -> logging.Logger:
    """Configure et retourne le logger principal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_langfuse_handler() -> CallbackHandler:
    """Instancie et retourne le handler Langfuse."""
    return CallbackHandler()


logger = setup_logging()
