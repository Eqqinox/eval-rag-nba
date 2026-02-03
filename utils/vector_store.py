# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
from typing import List, Optional
from mistralai import Mistral
from mistralai.models import SDKError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)
from .schemas import (
    Document, Chunk, ChunkMetadata, EmbeddingResult, SearchResult,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Chunk] = []
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks depuis {DOCUMENT_CHUNKS_FILE}...")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    raw_chunks = pickle.load(f)
                # Reconversion des dicts en modèles Chunk (compatibilité anciens fichiers)
                self.document_chunks = []
                for item in raw_chunks:
                    if isinstance(item, Chunk):
                        self.document_chunks.append(item)
                    elif isinstance(item, dict):
                        self.document_chunks.append(Chunk(**item))
                    else:
                        self.document_chunks.append(item)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(self, documents: List[Document]) -> List[Chunk]:
        """Découpe les documents en chunks validés avec Pydantic."""
        logging.info(f"Découpage de {len(documents)} documents en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        all_chunks = []
        doc_counter = 0
        for doc in documents:
            # Convertit le Document Pydantic en Document LangChain pour le splitter
            langchain_doc = LangchainDocument(
                page_content=doc.page_content,
                metadata=doc.metadata.model_dump(),
            )
            chunks = text_splitter.split_documents([langchain_doc])
            logging.info(f"  Document '{doc.metadata.filename}' découpé en {len(chunks)} chunks.")

            for i, chunk in enumerate(chunks):
                validated_chunk = Chunk(
                    id=f"{doc_counter}_{i}",
                    text=chunk.page_content,
                    metadata=ChunkMetadata(
                        source=chunk.metadata.get("source", ""),
                        filename=chunk.metadata.get("filename", ""),
                        category=chunk.metadata.get("category", ""),
                        full_path=chunk.metadata.get("full_path", ""),
                        sheet=chunk.metadata.get("sheet"),
                        chunk_id_in_doc=i,
                        start_index=chunk.metadata.get("start_index", -1),
                    ),
                )
                all_chunks.append(validated_chunk)
            doc_counter += 1

        logging.info(f"Total de {len(all_chunks)} chunks créés et validés.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Chunk]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk.text for chunk in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
            try:
                response = self.mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=texts_to_embed,
                )
                batch_embeddings = [data.embedding for data in response.data]

                # Validation Pydantic du lot d'embeddings
                validated = EmbeddingResult(
                    embeddings=batch_embeddings,
                    model=EMBEDDING_MODEL,
                    dimension=len(batch_embeddings[0]),
                    count=len(batch_embeddings),
                )
                all_embeddings.extend(validated.embeddings)
            except SDKError as e:
                logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
            except Exception as e:
                logging.error(f"Erreur lors de la génération d'embeddings (lot {batch_num}): {e}")
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                    logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                    continue
                logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype='float32').tolist()] * num_failed)

        if not all_embeddings:
            logging.error("Aucun embedding n'a pu être généré.")
            return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Document]):
        """Construit l'index Faiss à partir des documents validés."""
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        # 1. Découper en chunks
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        # 2. Générer les embeddings
        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks.")
            self.document_chunks = []
            self.index = None
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            if os.path.exists(DOCUMENT_CHUNKS_FILE): os.remove(DOCUMENT_CHUNKS_FILE)
            return

        # 3. Créer l'index Faiss optimisé pour la similarité cosinus
        dimension = embeddings.shape[1]
        logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        # 4. Sauvegarder l'index et les chunks
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks (sérialisés en dicts)."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            # Sérialiser en dicts pour la compatibilité pickle
            chunks_as_dicts = [chunk.model_dump() for chunk in self.document_chunks]
            with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(chunks_as_dicts, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[SearchResult]:
        """
        Recherche les k chunks les plus pertinents pour une requête.
        Retourne une liste de SearchResult validés.
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
            logging.error("Recherche impossible: MISTRAL_API_KEY manquante.")
            return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            # 1. Générer l'embedding de la requête
            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[query_text],
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')

            faiss.normalize_L2(query_embedding)

            # 2. Rechercher dans l'index Faiss
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Formater les résultats avec validation Pydantic
            results = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks):
                        chunk = self.document_chunks[idx]
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            continue

                        result = SearchResult(
                            score=similarity,
                            raw_score=raw_score,
                            text=chunk.text,
                            metadata=chunk.metadata,
                        )
                        results.append(result)
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")

            results.sort(key=lambda x: x.score, reverse=True)

            if len(results) > k:
                results = results[:k]

            logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        except SDKError as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []
