# MistralChat.py (version RAG + SQL)
import streamlit as st
import os
import logging
import logfire
from mistralai import Mistral, UserMessage
from dotenv import load_dotenv

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
    from database.sql_tool import query_nba_database
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()


# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

try:
    client = Mistral(api_key=api_key)
    logging.info("Client Mistral initialisé.")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    logging.exception("Erreur initialisation client Mistral")
    st.stop()

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         st.error("Fichiers d'index ou de chunks non trouvés.")
         st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None

vector_store_manager = get_vector_store_manager()

# --- Prompt Système pour RAG ---
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

# --- Prompt pour le Router LLM ---
ROUTER_PROMPT = """Tu es un classificateur de questions. Ta tâche est de déterminer si une question nécessite des données chiffrées/statistiques ou des informations textuelles/narratives.

Réponds UNIQUEMENT par "SQL" ou "RAG" (un seul mot, rien d'autre).

- Réponds "SQL" si la question porte sur :
  - Des statistiques précises (points, rebonds, passes, pourcentages, etc.)
  - Des classements (meilleur marqueur, top 5, etc.)
  - Des comparaisons chiffrées entre joueurs
  - Des données d'équipes (nombre de victoires, défaites, etc.)
  - Des questions avec "combien", "quel pourcentage", "moyenne", etc.

- Réponds "RAG" si la question porte sur :
  - Des analyses, opinions ou discussions
  - L'histoire ou le contexte d'un joueur/équipe
  - Des événements, rumeurs ou actualités
  - Des questions ouvertes sans réponse chiffrée précise

Question : {question}

Classification :"""


# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

# --- Fonctions ---

def router_question(question: str) -> str:
    """
    Utilise le LLM pour classifier la question : SQL ou RAG.
    """
    with logfire.span("router-llm", question=question[:80]):
        try:
            prompt = ROUTER_PROMPT.format(question=question)
            response = client.chat.complete(
                model=model,
                messages=[UserMessage(content=prompt)],
                temperature=0,
            )
            classification = response.choices[0].message.content.strip().upper()

            # Normaliser la réponse
            if "SQL" in classification:
                result = "SQL"
            elif "RAG" in classification:
                result = "RAG"
            else:
                # Par défaut, utiliser RAG si la classification est ambiguë
                result = "RAG"
                logging.warning(f"Classification ambiguë '{classification}', utilisation de RAG par défaut.")

            logfire.info("Question classifiée", classification=result)
            logging.info(f"Question classifiée comme : {result}")
            return result

        except Exception as e:
            logging.exception("Erreur lors de la classification de la question")
            logfire.error("Erreur router", erreur=str(e))
            return "RAG"  # Fallback sur RAG en cas d'erreur


def generer_reponse(prompt_messages: list[UserMessage]) -> str:
    """
    Envoie le prompt (qui inclut maintenant le contexte) à l'API Mistral.
    """
    if not prompt_messages:
         logging.warning("Tentative de génération de réponse avec un prompt vide.")
         return "Je ne peux pas traiter une demande vide."
    try:
        logging.info(f"Appel à l'API Mistral modèle '{model}' avec {len(prompt_messages)} message(s).")

        response = client.chat.complete(
            model=model,
            messages=prompt_messages,
            temperature=0.1,
        )
        if response.choices and len(response.choices) > 0:
            logging.info("Réponse reçue de l'API Mistral.")
            return response.choices[0].message.content
        else:
            logging.warning("L'API n'a pas retourné de choix valide.")
            return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
        logging.exception("Erreur API Mistral pendant client.chat")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."


def handle_sql_question(question: str) -> tuple[str, str]:
    """
    Traite une question via le Tool SQL.
    Retourne (réponse, source_info).
    """
    with logfire.span("pipeline-sql", question=question[:80]):
        try:
            result = query_nba_database(question)
            response = result["answer"]
            source_info = f"Source: Base de données SQL\nRequête: {result['sql_query']}"
            logfire.info("Réponse SQL générée", longueur=len(response))
            return response, source_info
        except Exception as e:
            logging.exception("Erreur lors du traitement SQL")
            logfire.error("Erreur pipeline SQL", erreur=str(e))
            return f"Erreur lors de la requête SQL : {e}", ""


def handle_rag_question(question: str) -> tuple[str, str]:
    """
    Traite une question via le pipeline RAG.
    Retourne (réponse, source_info).
    """
    with logfire.span("pipeline-rag", question=question[:80]):
        # Vérifier si le Vector Store est disponible
        if vector_store_manager is None:
            return "Le service de recherche de connaissances n'est pas disponible.", ""

        # Rechercher le contexte dans le Vector Store
        try:
            search_results = vector_store_manager.search(question, k=SEARCH_K)
        except Exception as e:
            logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {question}")
            search_results = []

        # Formater le contexte pour le prompt LLM
        context_str = "\n\n---\n\n".join([
            f"Source: {res.metadata.source} (Score: {res.score:.1f}%)\nContenu: {res.text}"
            for res in search_results
        ])

        if not search_results:
            context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
            logging.warning(f"Aucun contexte trouvé pour la query: {question}")

        # Construire le prompt final pour l'API Mistral
        final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=question)
        messages_for_api = [UserMessage(content=final_prompt_for_llm)]

        # Générer la réponse
        with logfire.span("appel-llm", modele=model):
            response_content = generer_reponse(messages_for_api)
            logfire.info(
                "Réponse LLM générée",
                longueur_reponse=len(response_content),
                nb_contextes=len(search_results),
            )

        # Préparer les infos de source
        if search_results:
            sources = [f"- {res.metadata.source} ({res.score:.0f}%)" for res in search_results[:3]]
            source_info = "Sources RAG:\n" + "\n".join(sources)
        else:
            source_info = ""

        return response_content, source_info


# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Afficher l'indicateur de chargement
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("Analyse de votre question...")

        with logfire.span("pipeline-agent-complet", question=prompt):
            # 3. Router la question (SQL ou RAG)
            question_type = router_question(prompt)

            message_placeholder.text(f"Recherche en cours ({question_type})...")

            # 4. Traiter selon le type
            if question_type == "SQL":
                response_content, source_info = handle_sql_question(prompt)
            else:
                response_content, source_info = handle_rag_question(prompt)

            logfire.info(
                "Pipeline terminé",
                type=question_type,
                longueur_reponse=len(response_content),
            )

        # 5. Afficher la réponse
        message_placeholder.write(response_content)

        # Afficher les sources dans un expander (optionnel)
        if source_info:
            with st.expander("Voir les sources"):
                st.code(source_info, language=None)

    # 6. Ajouter la réponse de l'assistant à l'historique
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss & SQLite | Data-driven NBA Insights")
