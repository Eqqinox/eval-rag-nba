"""
evaluate_ragas.py
Script d'évaluation du système RAG+SQL NBA avec le framework RAGAS.
Reproduit le pipeline de MistralChat.py (avec router) en mode headless,
puis évalue les réponses avec des métriques RAGAS.

Usage : KMP_DUPLICATE_LIB_OK=TRUE python evaluate_ragas.py
        KMP_DUPLICATE_LIB_OK=TRUE python evaluate_ragas.py --mode rag  # RAG uniquement
        KMP_DUPLICATE_LIB_OK=TRUE python evaluate_ragas.py --mode hybrid  # Router (défaut)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import logfire
import numpy as np
import pandas as pd
from mistralai import Mistral, UserMessage
from openai import OpenAI

from langchain_mistralai import MistralAIEmbeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas.metrics import (
    answer_relevancy,
    context_recall,
    faithfulness,
    context_precision,
)

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager
from database.sql_tool import query_nba_database

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Constantes ---
TEST_QUESTIONS_FILE = "test_questions.json"
RESULTS_DIR = "evaluation_results"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DELAY_BETWEEN_CALLS = 1  # secondes entre chaque appel API

SYSTEM_PROMPT_TEMPLATE = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

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


# --- Chargement des questions de test ---

def load_test_questions(filepath: str = TEST_QUESTIONS_FILE) -> List[Dict[str, Any]]:
    """Charge les questions de test depuis un fichier JSON."""
    logger.info(f"Chargement des questions depuis {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)
    logger.info(f"{len(questions)} questions chargées.")
    return questions


# --- Router LLM ---

def router_question(question: str, mistral_client: Mistral) -> str:
    """Classifie la question : SQL ou RAG."""
    with logfire.span("router-llm", question=question[:80]):
        try:
            prompt = ROUTER_PROMPT.format(question=question)
            response = mistral_client.chat.complete(
                model=MODEL_NAME,
                messages=[UserMessage(content=prompt)],
                temperature=0,
            )
            classification = response.choices[0].message.content.strip().upper()

            if "SQL" in classification:
                result = "SQL"
            elif "RAG" in classification:
                result = "RAG"
            else:
                result = "RAG"
                logger.warning(f"Classification ambiguë '{classification}', utilisation de RAG.")

            logfire.info("Question classifiée", classification=result)
            return result

        except Exception as e:
            logger.exception("Erreur lors de la classification")
            logfire.error("Erreur router", erreur=str(e))
            return "RAG"


# --- Pipeline RAG ---

def run_rag_pipeline(
    question: str,
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
    model_name: str = MODEL_NAME,
    search_k: int = SEARCH_K,
) -> Dict[str, Any]:
    """Exécute le pipeline RAG pour une question."""
    with logfire.span("pipeline-rag", question=question[:80]):
        # 1. Recherche dans le vector store
        search_results = vector_store.search(question, k=search_k)

        # 2. Formatage du contexte
        if search_results:
            context_str = "\n\n---\n\n".join([
                f"Source: {res.metadata.source} "
                f"(Score: {res.score:.1f}%)\n"
                f"Contenu: {res.text}"
                for res in search_results
            ])
        else:
            context_str = "Aucune information pertinente trouvée."

        # 3. Construction du prompt
        final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context_str=context_str, question=question
        )

        # 4. Appel LLM
        with logfire.span("appel-llm", modele=model_name):
            messages = [UserMessage(content=final_prompt)]
            response = mistral_client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=0.1,
            )
            answer = response.choices[0].message.content

        # 5. Extraction des contextes pour RAGAS
        retrieved_contexts = [res.text for res in search_results]

        logfire.info("Pipeline RAG terminé", nb_contextes=len(retrieved_contexts))

        return {
            "question": question,
            "response": answer,
            "retrieved_contexts": retrieved_contexts,
            "pipeline_type": "RAG",
        }


# --- Pipeline SQL ---

def run_sql_pipeline(question: str) -> Dict[str, Any]:
    """Exécute le pipeline SQL pour une question."""
    with logfire.span("pipeline-sql", question=question[:80]):
        try:
            result = query_nba_database(question)
            answer = result["answer"]
            # Pour RAGAS, on utilise le résultat SQL comme contexte
            sql_context = f"Requête SQL: {result['sql_query']}\nRésultat: {result['sql_result']}"

            logfire.info("Pipeline SQL terminé")

            return {
                "question": question,
                "response": answer,
                "retrieved_contexts": [sql_context],
                "pipeline_type": "SQL",
            }
        except Exception as e:
            logger.error(f"Erreur SQL : {e}")
            logfire.error("Erreur pipeline SQL", erreur=str(e))
            return {
                "question": question,
                "response": f"Erreur : {e}",
                "retrieved_contexts": [],
                "pipeline_type": "SQL",
            }


# --- Pipeline hybride (avec router) ---

def run_hybrid_pipeline(
    question: str,
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
) -> Dict[str, Any]:
    """Exécute le pipeline hybride : router -> SQL ou RAG."""
    with logfire.span("pipeline-hybrid", question=question[:80]):
        # 1. Classification
        pipeline_type = router_question(question, mistral_client)

        # 2. Exécution du bon pipeline
        if pipeline_type == "SQL":
            result = run_sql_pipeline(question)
        else:
            result = run_rag_pipeline(question, vector_store, mistral_client)

        logfire.info("Pipeline hybride terminé", type=pipeline_type)
        return result


def run_all_questions(
    questions: List[Dict[str, Any]],
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
    mode: str = "hybrid",
) -> List[Dict[str, Any]]:
    """Exécute le pipeline sur toutes les questions de test."""
    results = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        logger.info(f"[{i}/{total}] Traitement : {question_text[:60]}...")

        try:
            if mode == "hybrid":
                result = run_hybrid_pipeline(question_text, vector_store, mistral_client)
            else:  # mode == "rag"
                result = run_rag_pipeline(question_text, vector_store, mistral_client)

            result["category"] = q["category"]
            result["reference"] = q.get("reference")
            results.append(result)
            logger.info(f"[{i}/{total}] {result['pipeline_type']} - Réponse ({len(result['response'])} car.)")

        except Exception as e:
            logger.error(f"[{i}/{total}] Erreur : {e}")
            results.append({
                "question": question_text,
                "response": "",
                "retrieved_contexts": [],
                "pipeline_type": "ERROR",
                "category": q["category"],
                "reference": q.get("reference"),
            })

        if i < total:
            time.sleep(DELAY_BETWEEN_CALLS)

    succeeded = sum(1 for r in results if r["response"])
    sql_count = sum(1 for r in results if r.get("pipeline_type") == "SQL")
    rag_count = sum(1 for r in results if r.get("pipeline_type") == "RAG")
    logger.info(f"Pipeline terminé : {succeeded}/{total} réussies (SQL: {sql_count}, RAG: {rag_count})")
    return results


# --- Construction du dataset RAGAS ---

def build_evaluation_dataset(results: List[Dict[str, Any]]) -> EvaluationDataset:
    """Convertit les résultats du pipeline en EvaluationDataset RAGAS."""
    samples = []
    for r in results:
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r.get("reference"),
        )
        samples.append(sample)
    logger.info(f"EvaluationDataset construit avec {len(samples)} échantillons.")
    return EvaluationDataset(samples=samples)


# --- Configuration RAGAS ---

def create_ragas_llm():
    """Crée le LLM évaluateur RAGAS."""
    client = OpenAI(api_key=MISTRAL_API_KEY, base_url=MISTRAL_BASE_URL)
    return llm_factory("mistral-small-latest", provider="openai", client=client)


def create_ragas_embeddings():
    """Crée les embeddings évaluateur RAGAS."""
    langchain_embeddings = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model="mistral-embed",
    )
    return LangchainEmbeddingsWrapper(langchain_embeddings)


# --- Évaluation RAGAS ---

def run_evaluation(
    results: List[Dict[str, Any]],
    ragas_llm,
    ragas_embeddings,
) -> pd.DataFrame:
    """Lance l'évaluation RAGAS."""
    # --- Passe 1 : métriques sans ground_truth ---
    logger.info("=== Passe 1 : Faithfulness, AnswerRelevancy ===")

    dataset_all = build_evaluation_dataset(results)
    metrics_no_ref = [faithfulness, answer_relevancy]

    result_no_ref = evaluate(
        dataset=dataset_all,
        metrics=metrics_no_ref,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df_scores = result_no_ref.to_pandas()
    logger.info(f"Passe 1 terminée. Colonnes : {list(df_scores.columns)}")

    # --- Passe 2 : métriques avec référence ---
    results_with_ref = [r for r in results if r.get("reference") is not None]

    if results_with_ref:
        logger.info(f"=== Passe 2 : ContextPrecision, ContextRecall ({len(results_with_ref)} questions) ===")
        dataset_ref = build_evaluation_dataset(results_with_ref)
        result_ref = evaluate(
            dataset=dataset_ref,
            metrics=[context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        df_ref = result_ref.to_pandas()

        ref_questions = [r["question"] for r in results_with_ref]
        df_scores["context_precision"] = np.nan
        df_scores["context_recall"] = np.nan
        for idx, row in df_ref.iterrows():
            q = ref_questions[idx]
            mask = df_scores["user_input"] == q
            if mask.any():
                df_scores.loc[mask, "context_precision"] = row["context_precision"]
                df_scores.loc[mask, "context_recall"] = row["context_recall"]

        logger.info("Passe 2 terminée.")
    else:
        df_scores["context_precision"] = np.nan
        df_scores["context_recall"] = np.nan

    # Ajouter catégorie et pipeline_type
    df_scores["category"] = [r["category"] for r in results]
    df_scores["pipeline_type"] = [r.get("pipeline_type", "unknown") for r in results]

    return df_scores


# --- Formatage et export ---

def compute_aggregate_scores(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Calcule les scores moyens par catégorie et au global."""
    agg = df.groupby("category")[metric_cols].mean()
    agg.loc["all"] = df[metric_cols].mean()
    return agg.round(4)


def print_results(df: pd.DataFrame, agg_df: pd.DataFrame, metric_cols: List[str], mode: str) -> None:
    """Affiche les résultats dans la console."""
    print("\n" + "=" * 80)
    print(f"ÉVALUATION RAGAS DU SYSTÈME NBA (mode: {mode.upper()})")
    print("=" * 80)

    # Stats par pipeline
    pipeline_counts = df["pipeline_type"].value_counts()
    print(f"\nRépartition des pipelines : {dict(pipeline_counts)}")

    print("\n--- Scores agrégés par catégorie ---\n")
    print(agg_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()


def save_results(
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    metric_cols: List[str],
    mode: str,
    output_dir: str = RESULTS_DIR,
) -> None:
    """Sauvegarde les résultats en CSV et JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Suffixe pour différencier les modes
    suffix = f"_{mode}" if mode != "hybrid" else ""

    # CSV détaillé
    detailed_path = os.path.join(output_dir, f"detailed_scores{suffix}.csv")
    df.to_csv(detailed_path, index=False, encoding="utf-8")
    logger.info(f"Scores détaillés sauvegardés dans {detailed_path}")

    # CSV agrégé
    aggregate_path = os.path.join(output_dir, f"aggregate_scores{suffix}.csv")
    agg_df.to_csv(aggregate_path, encoding="utf-8")
    logger.info(f"Scores agrégés sauvegardés dans {aggregate_path}")

    # JSON récapitulatif
    pipeline_counts = df["pipeline_type"].value_counts().to_dict()
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "model": MODEL_NAME,
        "search_k": SEARCH_K,
        "num_questions": len(df),
        "num_with_reference": int(df["context_recall"].notna().sum()),
        "pipeline_distribution": pipeline_counts,
        "metrics": metric_cols,
        "aggregate_scores": {},
    }
    for category in agg_df.index:
        summary["aggregate_scores"][category] = {
            col: round(float(agg_df.loc[category, col]), 4)
            if not np.isnan(agg_df.loc[category, col])
            else None
            for col in metric_cols
        }

    summary_path = os.path.join(output_dir, f"evaluation_summary{suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Résumé sauvegardé dans {summary_path}")


# --- Point d'entrée ---

def main():
    """Fonction principale d'évaluation."""
    parser = argparse.ArgumentParser(description="Évaluation RAGAS du système NBA")
    parser.add_argument(
        "--mode",
        choices=["rag", "hybrid"],
        default="hybrid",
        help="Mode d'évaluation : 'rag' (RAG seul) ou 'hybrid' (router SQL+RAG, défaut)",
    )
    args = parser.parse_args()

    start_time = time.time()

    with logfire.span("evaluation-ragas-complete", mode=args.mode):
        # 1. Charger les questions de test
        questions = load_test_questions()

        # 2. Initialiser le vector store et le client Mistral
        vector_store = VectorStoreManager()
        if vector_store.index is None:
            logger.error("Index FAISS non chargé. Exécutez d'abord 'python indexer.py'.")
            return

        mistral_client = Mistral(api_key=MISTRAL_API_KEY)

        # 3. Exécuter le pipeline sur toutes les questions
        with logfire.span("execution-pipeline", mode=args.mode, nb_questions=len(questions)):
            results = run_all_questions(questions, vector_store, mistral_client, mode=args.mode)
            succeeded = sum(1 for r in results if r["response"])
            logfire.info("Pipeline terminé", reussies=succeeded, total=len(questions))

        # 4. Créer le LLM et les embeddings évaluateurs RAGAS
        ragas_llm = create_ragas_llm()
        ragas_embeddings = create_ragas_embeddings()

        # 5. Lancer l'évaluation RAGAS
        with logfire.span("evaluation-metriques-ragas"):
            df_scores = run_evaluation(results, ragas_llm, ragas_embeddings)

        # 6. Calculer les agrégats et afficher/sauvegarder
        metric_cols = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        metric_cols = [c for c in metric_cols if c in df_scores.columns]

        agg_df = compute_aggregate_scores(df_scores, metric_cols)
        print_results(df_scores, agg_df, metric_cols, args.mode)
        save_results(df_scores, agg_df, metric_cols, args.mode)

        elapsed = time.time() - start_time
        logger.info(f"Évaluation terminée en {elapsed:.1f} secondes")
        logfire.info("Évaluation terminée", duree_secondes=round(elapsed, 1))


if __name__ == "__main__":
    main()
