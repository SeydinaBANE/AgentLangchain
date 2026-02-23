"""
Interface Streamlit pour l'agent ReAct.
Fonctionnalités : historique, choix du modèle, étapes de l'agent, coûts/tokens.
"""

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from langfuse.langchain import CallbackHandler

from src.agent import build_agent
from src.config import get_langfuse_handler
from langchain_groq import ChatGroq
from src.tools import square_root, web_search
from langchain.agents import create_agent

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Langchain",
    page_icon="🤖",
    layout="wide",
)

# ── Modèles disponibles ───────────────────────────────────────────────────────
MODELS = {
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "LLaMA 3.3 70B": "llama-3.3-70b-versatile",

}

# Coût approximatif par token (en $) — à titre indicatif
COST_PER_TOKEN = 0.000001


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "agent" not in st.session_state:
    st.session_state.agent = None
if "current_model" not in st.session_state:
    st.session_state.current_model = "GPT-OSS 120B"


# ── Fonctions utilitaires ─────────────────────────────────────────────────────
def build_agent_for_model(model_name: str):
    """Construit un agent pour le modèle sélectionné."""
    llm = ChatGroq(model=MODELS[model_name], temperature=0)
    return create_agent(
        model=llm,
        tools=[square_root, web_search],
        system_prompt=(
            "Tu es un assistant expert polyvalent. "
            "Tu réponds toujours en français, de manière claire et structurée. "
            "Utilise les outils disponibles quand c'est nécessaire."
        ),
    )


def extract_steps(messages: list) -> list:
    """Extrait les étapes intermédiaires de l'agent (appels d'outils)."""
    steps = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "tool": tc.get("name", "inconnu"),
                    "input": tc.get("args", {}),
                })
        if hasattr(msg, "name") and msg.name in ["square_root", "web_search"]:
            steps.append({
                "tool": msg.name,
                "output": msg.content[:300] if isinstance(msg.content, str) else str(msg.content)[:300],
            })
    return steps


def estimate_tokens(text: str) -> int:
    """Estimation approximative du nombre de tokens (1 token ≈ 4 caractères)."""
    return max(1, len(text) // 4)


def run_agent(question: str, model_name: str) -> dict:
    """Invoque l'agent et retourne la réponse + métadonnées."""
    if st.session_state.agent is None or st.session_state.current_model != model_name:
        st.session_state.agent = build_agent_for_model(model_name)
        st.session_state.current_model = model_name

    langfuse_handler = get_langfuse_handler()

    # Construire l'historique pour le contexte
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    history.append(HumanMessage(content=question))

    response = st.session_state.agent.invoke(
        {"messages": history},
        config={"callbacks": [langfuse_handler]},
    )

    all_messages = response["messages"]
    answer = all_messages[-1].content
    steps = extract_steps(all_messages)

    tokens = estimate_tokens(question + answer)
    cost = tokens * COST_PER_TOKEN

    return {
        "answer": answer,
        "steps": steps,
        "tokens": tokens,
        "cost": cost,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.divider()

    # Sélecteur de modèle
    st.subheader("🧠 Modèle")
    selected_model = st.selectbox(
        "Choisir le modèle",
        options=list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    if selected_model != st.session_state.current_model:
        st.session_state.agent = None

    st.divider()

    # Statistiques
    st.subheader("📊 Session")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔤 Tokens", f"{st.session_state.total_tokens:,}")
    with col2:
        st.metric("💰 Coût", f"${st.session_state.total_cost:.4f}")

    st.caption("⚠️ Estimation indicative")

    st.divider()

    # Outils disponibles
    st.subheader("🛠️ Outils disponibles")
    st.markdown("🔍 **web_search** — Recherche web via Tavily")
    st.markdown("🧮 **square_root** — Calcul de racine carrée")

    st.divider()

    # Bouton reset
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

    st.divider()
    st.caption("Propulsé par LangGraph · Groq · Langfuse")
    st.caption("[![GitHub](https://img.shields.io/badge/GitHub-repo-black?logo=github)](https://github.com/SeydinaBANE/AgentLangchain)")


# ── Interface principale ──────────────────────────────────────────────────────
st.title("🤖 Agent Langchain")
st.caption(f"Modèle actif : **{selected_model}** — {MODELS[selected_model]}")
st.divider()

# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Afficher les étapes si présentes
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander(f"🔍 Étapes de l'agent ({len(msg['steps'])} action(s))", expanded=False):
                for i, step in enumerate(msg["steps"], 1):
                    if "input" in step:
                        st.markdown(f"**Étape {i} — Appel outil : `{step['tool']}`**")
                        st.json(step["input"])
                    elif "output" in step:
                        st.markdown(f"**Étape {i} — Résultat : `{step['tool']}`**")
                        st.text(step["output"])

        # Afficher les tokens
        if msg["role"] == "assistant" and msg.get("tokens"):
            st.caption(f"~{msg['tokens']} tokens · ~${msg['cost']:.4f}")


# ── Input utilisateur ─────────────────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question..."):
    # Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Appel à l'agent
    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit..."):
            try:
                result = run_agent(prompt, selected_model)

                st.markdown(result["answer"])

                # Étapes de l'agent
                if result["steps"]:
                    with st.expander(f"🔍 Étapes de l'agent ({len(result['steps'])} action(s))", expanded=True):
                        for i, step in enumerate(result["steps"], 1):
                            if "input" in step:
                                st.markdown(f"**Étape {i} — Appel outil : `{step['tool']}`**")
                                st.json(step["input"])
                            elif "output" in step:
                                st.markdown(f"**Étape {i} — Résultat : `{step['tool']}`**")
                                st.text(step["output"])

                st.caption(f"~{result['tokens']} tokens · ~${result['cost']:.4f}")

                # Mettre à jour les stats
                st.session_state.total_tokens += result["tokens"]
                st.session_state.total_cost += result["cost"]

                # Sauvegarder dans l'historique
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "steps": result["steps"],
                    "tokens": result["tokens"],
                    "cost": result["cost"],
                })

            except Exception as e:
                st.error(f"❌ Une erreur s'est produite : {e}")