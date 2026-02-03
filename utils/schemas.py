"""
Schémas Pydantic pour la validation des données du pipeline RAG.
Sécurise les flux d'entrée et de sortie à chaque étape du pipeline
de préparation des données (chargement, chunking, embedding, recherche).
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# --- Modèles pour le chargement des documents ---


class DocumentMetadata(BaseModel):
    """Métadonnées d'un document source chargé depuis le système de fichiers."""

    source: str = Field(..., min_length=1, description="Chemin relatif du fichier source")
    filename: str = Field(..., min_length=1, description="Nom du fichier source")
    category: str = Field(..., description="Catégorie ou dossier d'origine du document")
    full_path: str = Field(..., description="Chemin absolu du fichier sur le disque")
    sheet: Optional[str] = Field(default=None, description="Nom de la feuille Excel (si applicable)")


class Document(BaseModel):
    """Document source chargé et parsé, prêt pour le découpage en chunks."""

    page_content: str = Field(..., min_length=1, description="Contenu textuel extrait du fichier")
    metadata: DocumentMetadata

    @field_validator("page_content")
    @classmethod
    def contenu_non_vide(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le contenu du document ne peut pas être uniquement des espaces blancs")
        return v


# --- Modèles pour le chunking ---


class ChunkMetadata(BaseModel):
    """Métadonnées d'un chunk, héritées du document parent et enrichies."""

    source: str
    filename: str
    category: str
    full_path: str
    sheet: Optional[str] = None
    chunk_id_in_doc: int = Field(..., ge=0, description="Position du chunk dans le document d'origine")
    start_index: int = Field(..., ge=-1, description="Position de début en caractères (-1 si inconnu)")


class Chunk(BaseModel):
    """Chunk de texte découpé depuis un document, avec métadonnées enrichies."""

    id: str = Field(..., min_length=1, description="Identifiant unique (format: docIdx_chunkIdx)")
    text: str = Field(..., min_length=1, description="Contenu textuel du chunk")
    metadata: ChunkMetadata

    @field_validator("text")
    @classmethod
    def texte_non_vide(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le texte du chunk ne peut pas être vide")
        return v


# --- Modèles pour les embeddings ---


class EmbeddingResult(BaseModel):
    """Résultat validé de la génération d'embeddings pour un lot de textes."""

    embeddings: List[List[float]] = Field(..., description="Vecteurs d'embedding générés")
    model: str = Field(..., description="Modèle utilisé pour la génération")
    dimension: int = Field(..., gt=0, description="Dimension des vecteurs")
    count: int = Field(..., gt=0, description="Nombre de vecteurs générés")

    @field_validator("embeddings")
    @classmethod
    def dimensions_coherentes(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("La liste d'embeddings ne peut pas être vide")
        dim = len(v[0])
        if any(len(emb) != dim for emb in v):
            raise ValueError("Tous les vecteurs doivent avoir la même dimension")
        return v


# --- Modèles pour la recherche ---


class SearchResult(BaseModel):
    """Résultat d'une recherche de similarité dans l'index FAISS."""

    score: float = Field(..., description="Score de similarité en pourcentage (0-100)")
    raw_score: float = Field(..., description="Score brut du produit scalaire")
    text: str = Field(..., description="Contenu textuel du chunk retrouvé")
    metadata: ChunkMetadata
