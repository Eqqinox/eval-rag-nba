"""
evaluate_ragas.py
Script d'évaluation du système RAG NBA avec le framework RAGAS.
Reproduit le pipeline RAG de MistralChat.py en mode headless,
puis évalue les réponses avec des métriques RAGAS.

Usage : python evaluate_ragas.py
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

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


# --- Chargement des questions de test ---

def load_test_questions(filepath: str = TEST_QUESTIONS_FILE) -> List[Dict[str, Any]]:
    """Charge les questions de test depuis un fichier JSON."""
    logger.info(f"Chargement des questions depuis {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)
    logger.info(f"{len(questions)} questions chargées.")
    return questions


# --- Pipeline RAG (mode headless) ---

def run_rag_pipeline(
    question: str,
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
    model_name: str = MODEL_NAME,
    search_k: int = SEARCH_K,
) -> Dict[str, Any]:
    """
    Exécute le pipeline RAG complet pour une question.
    Reproduit la logique de MistralChat.py sans Streamlit.
    """
    # 1. Recherche dans le vector store
    search_results = vector_store.search(question, k=search_k)

    # 2. Formatage du contexte (identique à MistralChat.py)
    if search_results:
        context_str = "\n\n---\n\n".join([
            f"Source: {res.metadata.source} "
            f"(Score: {res.score:.1f}%)\n"
            f"Contenu: {res.text}"
            for res in search_results
        ])
    else:
        context_str = (
            "Aucune information pertinente trouvée dans la base "
            "de connaissances pour cette question."
        )

    # 3. Construction du prompt
    final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context_str=context_str, question=question
    )

    # 4. Appel LLM via le nouveau SDK Mistral
    messages = [UserMessage(content=final_prompt)]
    response = mistral_client.chat.complete(
        model=model_name,
        messages=messages,
        temperature=0.1,
    )
    answer = response.choices[0].message.content

    # 5. Extraction des contextes pour RAGAS
    retrieved_contexts = [res.text for res in search_results]

    return {
        "question": question,
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
        "search_results": search_results,
    }


def run_all_questions(
    questions: List[Dict[str, Any]],
    vector_store: VectorStoreManager,
    mistral_client: Mistral,
) -> List[Dict[str, Any]]:
    """Exécute le pipeline RAG sur toutes les questions de test."""
    results = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        logger.info(f"[{i}/{total}] Traitement : {question_text[:80]}...")

        try:
            result = run_rag_pipeline(question_text, vector_store, mistral_client)
            result["category"] = q["category"]
            result["reference"] = q.get("reference")
            results.append(result)
            logger.info(f"[{i}/{total}] Réponse générée ({len(result['response'])} caractères)")
        except Exception as e:
            logger.error(f"[{i}/{total}] Erreur : {e}")
            results.append({
                "question": question_text,
                "response": "",
                "retrieved_contexts": [],
                "search_results": [],
                "category": q["category"],
                "reference": q.get("reference"),
            })

        if i < total:
            time.sleep(DELAY_BETWEEN_CALLS)

    succeeded = sum(1 for r in results if r["response"])
    logger.info(f"Pipeline terminé : {succeeded}/{total} questions traitées avec succès.")
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


# --- Configuration RAGAS (via endpoint OpenAI-compatible de Mistral) ---

def create_ragas_llm():
    """Crée le LLM évaluateur RAGAS via llm_factory et l'endpoint OpenAI-compatible de Mistral."""
    client = OpenAI(api_key=MISTRAL_API_KEY, base_url=MISTRAL_BASE_URL)
    return llm_factory("mistral-small-latest", provider="openai", client=client)


def create_ragas_embeddings():
    """Crée les embeddings évaluateur RAGAS via MistralAIEmbeddings (LangChain natif)."""
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
    """
    Lance l'évaluation RAGAS en deux passes :
    - Passe 1 : métriques sans ground_truth sur toutes les questions
    - Passe 2 : ContextRecall sur les questions avec référence
    """
    # --- Passe 1 : métriques sans ground_truth (toutes les questions) ---
    logger.info("=== Passe 1 : Faithfulness, AnswerRelevancy (30 questions) ===")

    dataset_all = build_evaluation_dataset(results)
    metrics_no_ref = [
        faithfulness,
        answer_relevancy,
    ]

    result_no_ref = evaluate(
        dataset=dataset_all,
        metrics=metrics_no_ref,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df_scores = result_no_ref.to_pandas()
    logger.info(f"Passe 1 terminée. Colonnes : {list(df_scores.columns)}")

    # --- Passe 2 : métriques avec référence (questions simples uniquement) ---
    results_with_ref = [r for r in results if r.get("reference") is not None]

    if results_with_ref:
        logger.info(
            f"=== Passe 2 : ContextPrecision, ContextRecall sur "
            f"{len(results_with_ref)} questions avec référence ==="
        )
        dataset_ref = build_evaluation_dataset(results_with_ref)
        result_ref = evaluate(
            dataset=dataset_ref,
            metrics=[context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        df_ref = result_ref.to_pandas()

        # Fusionner les scores dans le DataFrame principal
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
        logger.info("Aucune question avec référence, passe 2 ignorée.")

    # Ajouter la catégorie
    df_scores["category"] = [r["category"] for r in results]

    return df_scores


# --- Formatage et export des résultats ---

def compute_aggregate_scores(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Calcule les scores moyens par catégorie et au global."""
    agg = df.groupby("category")[metric_cols].mean()
    agg.loc["all"] = df[metric_cols].mean()
    return agg.round(4)


def print_results(df: pd.DataFrame, agg_df: pd.DataFrame, metric_cols: List[str]) -> None:
    """Affiche les résultats dans la console."""
    print("\n" + "=" * 80)
    print("ÉVALUATION RAGAS DU SYSTÈME RAG NBA")
    print("=" * 80)

    print("\n--- Scores détaillés par question ---\n")
    display_cols = ["user_input", "category"] + metric_cols
    display_df = df[display_cols].copy()
    display_df.columns = ["question", "catégorie"] + metric_cols
    display_df["question"] = display_df["question"].str[:70] + "..."
    print(display_df.to_string(index=True, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Scores agrégés par catégorie ---\n")
    print(agg_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()


def save_results(
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    metric_cols: List[str],
    output_dir: str = RESULTS_DIR,
) -> None:
    """Sauvegarde les résultats en CSV et JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV détaillé
    detailed_path = os.path.join(output_dir, "detailed_scores.csv")
    df.to_csv(detailed_path, index=False, encoding="utf-8")
    logger.info(f"Scores détaillés sauvegardés dans {detailed_path}")

    # CSV agrégé
    aggregate_path = os.path.join(output_dir, "aggregate_scores.csv")
    agg_df.to_csv(aggregate_path, encoding="utf-8")
    logger.info(f"Scores agrégés sauvegardés dans {aggregate_path}")

    # JSON récapitulatif
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "search_k": SEARCH_K,
        "num_questions": len(df),
        "num_with_reference": int(df["context_recall"].notna().sum()),
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

    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Résumé sauvegardé dans {summary_path}")


# --- Point d'entrée ---

def main():
    """Fonction principale d'évaluation."""
    start_time = time.time()
    logger.info("Démarrage de l'évaluation RAGAS...")

    # 1. Charger les questions de test
    questions = load_test_questions()

    # 2. Initialiser le vector store et le client Mistral
    logger.info("Initialisation du VectorStoreManager...")
    vector_store = VectorStoreManager()
    if vector_store.index is None:
        logger.error("Index FAISS non chargé. Exécutez d'abord 'python indexer.py'.")
        return

    logger.info("Initialisation du client Mistral...")
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)

    # 3. Exécuter le pipeline RAG sur toutes les questions
    logger.info("Exécution du pipeline RAG...")
    results = run_all_questions(questions, vector_store, mistral_client)

    # 4. Créer le LLM et les embeddings évaluateurs RAGAS
    logger.info("Initialisation du LLM et des embeddings RAGAS (wrappers LangChain Mistral)...")
    ragas_llm = create_ragas_llm()
    ragas_embeddings = create_ragas_embeddings()

    # 5. Lancer l'évaluation RAGAS
    logger.info("Lancement de l'évaluation RAGAS...")
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
    print_results(df_scores, agg_df, metric_cols)
    save_results(df_scores, agg_df, metric_cols)

    elapsed = time.time() - start_time
    logger.info(f"Évaluation terminée en {elapsed:.1f} secondes.")


if __name__ == "__main__":
    main()
