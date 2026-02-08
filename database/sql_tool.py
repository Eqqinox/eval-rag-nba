"""
Tool SQL LangChain pour interroger la base de données NBA.
Génère des requêtes SQL à partir de questions en langage naturel.

Usage :
    from database.sql_tool import query_nba_database
    result = query_nba_database("Quel joueur a marqué le plus de points ?")
"""

import logging
import os
import sys

import logfire
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import DATABASE_URL, MISTRAL_API_KEY, MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration de la base de données ---

db = SQLDatabase.from_uri(DATABASE_URL)

# --- Configuration du LLM ---

llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model=MODEL_NAME,
    temperature=0,
)

# --- Exemples few-shot pour guider la génération SQL ---

FEW_SHOT_EXAMPLES = """
Exemples de questions et requêtes SQL correspondantes :

Question : Quel joueur a marqué le plus de points ?
SQL : SELECT name, team_code, points_per_game FROM players ORDER BY points_per_game DESC LIMIT 1;

Question : Qui est le meilleur rebondeur ?
SQL : SELECT name, team_code, total_rebounds FROM players ORDER BY total_rebounds DESC LIMIT 1;

Question : Quelles sont les statistiques de Nikola Jokic ?
SQL : SELECT name, team_code, points_per_game, total_rebounds, assists, field_goal_pct FROM players WHERE name LIKE '%Jokic%';

Question : Quel joueur a le plus de passes décisives ?
SQL : SELECT name, team_code, assists FROM players ORDER BY assists DESC LIMIT 1;

Question : Combien de joueurs ont les Lakers ?
SQL : SELECT COUNT(*) as nb_joueurs FROM players WHERE team_code = 'LAL';

Question : Quels sont les 5 meilleurs marqueurs ?
SQL : SELECT name, team_code, points_per_game FROM players ORDER BY points_per_game DESC LIMIT 5;

Question : Compare Jokic et Giannis
SQL : SELECT name, points_per_game, total_rebounds, assists, field_goal_pct FROM players WHERE name LIKE '%Jokic%' OR name LIKE '%Giannis%';

Question : Quel joueur a le meilleur pourcentage au tir ?
SQL : SELECT name, team_code, field_goal_pct FROM players WHERE field_goal_pct IS NOT NULL ORDER BY field_goal_pct DESC LIMIT 1;

Question : Quelles équipes sont dans la base ?
SQL : SELECT code, name FROM teams ORDER BY name;

Question : Quel est le meilleur passeur des Clippers ?
SQL : SELECT name, assists FROM players WHERE team_code = 'LAC' ORDER BY assists DESC LIMIT 1;

Question : Moyenne de points par équipe
SQL : SELECT t.name, AVG(p.points_per_game) as moyenne_points FROM players p JOIN teams t ON p.team_code = t.code GROUP BY t.code ORDER BY moyenne_points DESC;

Question : Joueurs avec plus de 20 points par match
SQL : SELECT name, team_code, points_per_game FROM players WHERE points_per_game > 20 ORDER BY points_per_game DESC;
"""

# --- Prompt pour la génération SQL ---

SQL_GENERATION_PROMPT = PromptTemplate.from_template("""
Tu es un expert SQL. Ta tâche est de convertir une question en langage naturel en une requête SQL valide pour une base de données SQLite contenant des statistiques NBA.

Schéma de la base de données :
{schema}

{few_shot_examples}

Règles importantes :
- Génère UNIQUEMENT la requête SQL, sans explication ni commentaire
- Utilise des noms de colonnes exacts du schéma
- Pour les recherches de noms, utilise LIKE avec des % (ex: WHERE name LIKE '%Jokic%')
- Les codes d'équipe sont en majuscules (LAL, BOS, OKC, etc.)
- Limite les résultats avec LIMIT quand c'est approprié
- N'invente pas de colonnes qui n'existent pas

Question : {question}

SQL :""")

# --- Prompt pour la synthèse de la réponse ---

ANSWER_SYNTHESIS_PROMPT = PromptTemplate.from_template("""
Tu es un analyste NBA. On t'a posé une question et tu as obtenu des résultats d'une base de données.

Question originale : {question}

Résultats de la requête SQL :
{sql_result}

Génère une réponse claire et concise en français qui répond à la question en utilisant les données fournies.
Si les résultats sont vides, indique qu'aucune donnée n'a été trouvée.
""")


def get_schema() -> str:
    """Retourne le schéma de la base de données."""
    return db.get_table_info()


def generate_sql_query(question: str) -> str:
    """Génère une requête SQL à partir d'une question en langage naturel."""
    with logfire.span("generation-sql", question=question[:80]):
        schema = get_schema()
        prompt = SQL_GENERATION_PROMPT.format(
            schema=schema,
            few_shot_examples=FEW_SHOT_EXAMPLES,
            question=question,
        )

        response = llm.invoke(prompt)
        sql_query = response.content.strip()

        # Nettoyer la requête (enlever les backticks markdown si présents)
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()

        logfire.info("Requête SQL générée", sql=sql_query)
        logger.info(f"SQL généré : {sql_query}")

        return sql_query


def execute_sql_query(sql_query: str) -> str:
    """Exécute une requête SQL et retourne les résultats."""
    with logfire.span("execution-sql", sql=sql_query[:100]):
        try:
            result = db.run(sql_query)
            logfire.info("Requête exécutée", nb_caracteres=len(str(result)))
            return result
        except Exception as e:
            error_msg = f"Erreur SQL : {e}"
            logfire.error("Erreur exécution SQL", erreur=str(e))
            logger.error(error_msg)
            return error_msg


def synthesize_answer(question: str, sql_result: str) -> str:
    """Synthétise une réponse en langage naturel à partir des résultats SQL."""
    with logfire.span("synthese-reponse"):
        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            question=question,
            sql_result=sql_result,
        )

        response = llm.invoke(prompt)
        answer = response.content.strip()

        logfire.info("Réponse synthétisée", longueur=len(answer))
        return answer


def query_nba_database(question: str) -> dict:
    """
    Fonction principale du Tool SQL.
    Prend une question en langage naturel, génère le SQL, exécute et synthétise la réponse.

    Args:
        question: Question en langage naturel

    Returns:
        dict avec les clés : question, sql_query, sql_result, answer
    """
    with logfire.span("tool-sql-nba", question=question[:80]):
        logger.info(f"Question reçue : {question}")

        # 1. Générer la requête SQL
        sql_query = generate_sql_query(question)

        # 2. Exécuter la requête
        sql_result = execute_sql_query(sql_query)

        # 3. Synthétiser la réponse
        answer = synthesize_answer(question, sql_result)

        result = {
            "question": question,
            "sql_query": sql_query,
            "sql_result": sql_result,
            "answer": answer,
        }

        logfire.info("Tool SQL terminé")
        return result


# --- Test du module ---

if __name__ == "__main__":
    # Questions de test
    test_questions = [
        "Quel joueur a marqué le plus de points ?",
        "Quelles sont les statistiques de Nikola Jokic ?",
        "Qui est le meilleur passeur des Clippers ?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question : {q}")
        print("=" * 60)

        result = query_nba_database(q)

        print(f"\nSQL : {result['sql_query']}")
        print(f"\nRésultat brut : {result['sql_result'][:200]}...")
        print(f"\nRéponse : {result['answer']}")
