"""
Configuration du pipeline RAG, validée avec pydantic-settings.
Les paramètres sont chargés depuis les variables d'environnement et le fichier .env.
"""

import os
import logging
from typing import Optional

import logfire
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PipelineSettings(BaseSettings):
    """Paramètres validés du pipeline de préparation des données et du système RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Clé API ---
    mistral_api_key: str = Field(..., description="Clé API Mistral (MISTRAL_API_KEY)")

    # --- Modèles Mistral ---
    embedding_model: str = Field(default="mistral-embed", description="Modèle d'embedding")
    model_name: str = Field(default="mistral-small-latest", description="Modèle de chat")

    # --- Configuration de l'indexation ---
    input_dir: str = Field(default="inputs", description="Dossier des données sources")
    vector_db_dir: str = Field(default="vector_db", description="Dossier de l'index FAISS")
    chunk_size: int = Field(default=1500, gt=0, description="Taille des chunks en caractères")
    chunk_overlap: int = Field(default=150, ge=0, description="Chevauchement en caractères")
    embedding_batch_size: int = Field(default=32, gt=0, description="Taille des lots pour l'API d'embedding")

    # --- Configuration de la recherche ---
    search_k: int = Field(default=5, gt=0, description="Nombre de documents à récupérer")

    # --- Configuration de la base de données ---
    database_dir: str = Field(default="database", description="Dossier de la base de données")

    # --- Logfire ---
    logfire_token: Optional[str] = Field(default=None, description="Token Pydantic Logfire")

    # --- Configuration de l'application ---
    app_title: str = Field(default="NBA Analyst AI", description="Titre de l'application")
    name: str = Field(default="NBA", description="Nom affiché dans l'interface")

    @field_validator("chunk_overlap")
    @classmethod
    def chevauchement_inferieur_taille(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1500)
        if v >= chunk_size:
            raise ValueError(
                f"Le chevauchement ({v}) doit être inférieur à la taille des chunks ({chunk_size})"
            )
        return v

    @property
    def faiss_index_file(self) -> str:
        return os.path.join(self.vector_db_dir, "faiss_index.idx")

    @property
    def document_chunks_file(self) -> str:
        return os.path.join(self.vector_db_dir, "document_chunks.pkl")

    @property
    def database_file(self) -> str:
        return os.path.join(self.database_dir, "nba.db")

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_file}"


# --- Instance globale ---
settings = PipelineSettings()

# --- Initialisation Logfire ---
if settings.logfire_token:
    logfire.configure(token=settings.logfire_token)
    logfire.instrument_pydantic()
    logging.info("Logfire configuré et instrumentation Pydantic activée.")
else:
    logging.info("Logfire non configuré (LOGFIRE_TOKEN absent).")

# --- Aliases de compatibilité ---
# Permet aux imports existants (from utils.config import MISTRAL_API_KEY) de continuer à fonctionner
MISTRAL_API_KEY = settings.mistral_api_key
EMBEDDING_MODEL = settings.embedding_model
MODEL_NAME = settings.model_name
INPUT_DIR = settings.input_dir
VECTOR_DB_DIR = settings.vector_db_dir
FAISS_INDEX_FILE = settings.faiss_index_file
DOCUMENT_CHUNKS_FILE = settings.document_chunks_file
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
EMBEDDING_BATCH_SIZE = settings.embedding_batch_size
SEARCH_K = settings.search_k
APP_TITLE = settings.app_title
NAME = settings.name
DATABASE_DIR = settings.database_dir
DATABASE_FILE = settings.database_file
DATABASE_URL = settings.database_url
