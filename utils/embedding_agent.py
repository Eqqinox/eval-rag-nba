"""
Agent Pydantic AI pour la génération d'embeddings.
Encapsule l'appel à l'API Mistral Embeddings dans un Agent
avec entrées/sorties typées et validées.
"""

from dataclasses import dataclass
from typing import List

from mistralai import Mistral
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from .config import settings
from .schemas import EmbeddingResult


@dataclass
class EmbeddingDependencies:
    """Dépendances injectées dans l'agent d'embedding."""

    api_key: str
    embedding_model: str


# Agent configuré avec Mistral, sortie structurée EmbeddingResult
embedding_agent = Agent(
    MistralModel(
        model_name=settings.model_name,
        provider=MistralProvider(api_key=settings.mistral_api_key),
    ),
    output_type=EmbeddingResult,
    deps_type=EmbeddingDependencies,
    system_prompt=(
        "Tu es un agent de génération d'embeddings. "
        "Utilise l'outil generate_embeddings pour générer les vecteurs "
        "à partir des textes fournis, puis retourne le résultat validé."
    ),
)


@embedding_agent.tool
def generate_embeddings(
    ctx: RunContext[EmbeddingDependencies],
    texts: List[str],
) -> dict:
    """Génère les embeddings via l'API Mistral pour une liste de textes."""
    client = Mistral(api_key=ctx.deps.api_key)
    response = client.embeddings.create(
        model=ctx.deps.embedding_model,
        inputs=texts,
    )
    embeddings = [data.embedding for data in response.data]
    dimension = len(embeddings[0]) if embeddings else 0
    return {
        "embeddings": embeddings,
        "model": ctx.deps.embedding_model,
        "dimension": dimension,
        "count": len(embeddings),
    }


def run_embedding_agent(texts: List[str]) -> EmbeddingResult:
    """
    Fonction synchrone pour générer des embeddings via l'agent Pydantic AI.
    Retourne un EmbeddingResult validé.
    """
    deps = EmbeddingDependencies(
        api_key=settings.mistral_api_key,
        embedding_model=settings.embedding_model,
    )
    prompt = f"Génère les embeddings pour les {len(texts)} textes suivants."
    result = embedding_agent.run_sync(prompt, deps=deps)
    return result.output
