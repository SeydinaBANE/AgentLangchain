# 🤖 Agent ReAct — LangGraph + Groq + Tavily + Langfuse

Agent conversationnel en mode interactif, construit avec **LangGraph**, propulsé par **Groq**, équipé d'une recherche web via **Tavily**, et tracé avec **Langfuse**.

---

## ✨ Fonctionnalités

- 💬 **Mode interactif** — boucle de questions/réponses en continu
- 🔍 **Recherche web en temps réel** via Tavily
- 🧮 **Calcul de racines carrées** (outil mathématique)
- 📊 **Observabilité complète** avec Langfuse (traces, coûts, latences)
- 🛡️ **Gestion des erreurs** robuste à chaque étape
- 📝 **Logging professionnel** avec horodatage

---

## 🗂️ Structure du projet

```
langgraph-agent/
├── main.py               # Point d'entrée — boucle interactive
├── src/
│   ├── __init__.py
│   ├── agent.py          # Construction du LLM et de l'agent ReAct
│   ├── tools.py          # Outils (square_root, web_search)
│   └── config.py         # Logging et handler Langfuse
├── requirements.txt      # Dépendances Python
├── .env.example          # Template des variables d'environnement
├── .gitignore            # Fichiers à ignorer par Git
└── README.md             # Ce fichier
```

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/langgraph-agent.git
cd AgentLangchain`
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# ou
.venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
```

Puis éditez `.env` et renseignez vos clés API :

| Variable              | Où l'obtenir                          |
|-----------------------|---------------------------------------|
| `GROQ_API_KEY`        | https://console.groq.com/keys        |
| `TAVILY_API_KEY`      | https://app.tavily.com               |
| `LANGFUSE_PUBLIC_KEY` | https://cloud.langfuse.com           |
| `LANGFUSE_SECRET_KEY` | https://cloud.langfuse.com           |

---

## ▶️ Utilisation

```bash
python main.py
```

L'agent démarre en mode interactif :

```
============================================================
  🤖  Agent ReAct — Mode interactif
  Tapez 'quitter' ou 'exit' pour arrêter.
============================================================

Vous : Quelle est la température à Dakar aujourd'hui ?
Agent : D'après mes recherches, ...

Vous : Calcule la racine carrée de 144
Agent : La racine carrée de 144 est 12.0

Vous : quitter
Au revoir !
```

---

## 🛠️ Outils disponibles

| Outil          | Description                                      |
|----------------|--------------------------------------------------|
| `square_root`  | Calcule la racine carrée d'un nombre             |
| `web_search`   | Recherche en temps réel via l'API Tavily         |

---

## 🧩 Stack technique

| Composant    | Rôle                                      |
|--------------|-------------------------------------------|
| LangGraph    | Orchestration de l'agent (ReAct pattern)  |
| Groq         | Inférence rapide (openai/gpt-oss-120b)    |
| Tavily       | Recherche web                             |
| Langfuse     | Observabilité & traçage des LLM           |
| python-dotenv| Gestion des variables d'environnement     |

---

## 📄 Licence

MIT