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
import logfire

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
        with logfire.span("chargement-index-et-chunks"):
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
                    logfire.info(
                        "Index et chunks chargés",
                        nb_vecteurs=self.index.ntotal,
                        nb_chunks=len(self.document_chunks),
                    )
                except Exception as e:
                    logfire.error("Erreur chargement index/chunks", erreur=str(e))
                    self.index = None
                    self.document_chunks = []
            else:
                logfire.warn("Fichiers d'index non trouvés, index vide")

    def _split_documents_to_chunks(self, documents: List[Document]) -> List[Chunk]:
        """Découpe les documents en chunks validés avec Pydantic."""
        with logfire.span(
            "decoupage-chunks",
            nb_documents=len(documents),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )

            all_chunks = []
            doc_counter = 0
            for doc in documents:
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

            logfire.info("Chunks créés et validés", nb_chunks=len(all_chunks))
            return all_chunks

    def _generate_embeddings(self, chunks: List[Chunk]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        with logfire.span(
            "generation-embeddings",
            nb_chunks=len(chunks),
            modele=EMBEDDING_MODEL,
            nb_lots=total_batches,
        ):
            all_embeddings = []

            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
                batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
                texts_to_embed = [chunk.text for chunk in batch_chunks]

                with logfire.span("lot-embeddings", lot=batch_num, total=total_batches, taille=len(texts_to_embed)):
                    try:
                        response = self.mistral_client.embeddings.create(
                            model=EMBEDDING_MODEL,
                            inputs=texts_to_embed,
                        )
                        batch_embeddings = [data.embedding for data in response.data]

                        validated = EmbeddingResult(
                            embeddings=batch_embeddings,
                            model=EMBEDDING_MODEL,
                            dimension=len(batch_embeddings[0]),
                            count=len(batch_embeddings),
                        )
                        all_embeddings.extend(validated.embeddings)
                    except SDKError as e:
                        logfire.error("Erreur API Mistral embeddings", lot=batch_num, erreur=str(e))
                    except Exception as e:
                        logfire.error("Erreur génération embeddings", lot=batch_num, erreur=str(e))
                        num_failed = len(texts_to_embed)
                        if all_embeddings:
                            dim = len(all_embeddings[0])
                        else:
                            logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                            continue
                        all_embeddings.extend([np.zeros(dim, dtype='float32').tolist()] * num_failed)

            if not all_embeddings:
                logfire.error("Aucun embedding généré")
                return None

            embeddings_array = np.array(all_embeddings).astype('float32')
            logfire.info(
                "Embeddings générés",
                shape_0=embeddings_array.shape[0],
                shape_1=embeddings_array.shape[1],
            )
            return embeddings_array

    def build_index(self, documents: List[Document]):
        """Construit l'index Faiss à partir des documents validés."""
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        with logfire.span("construction-index", nb_documents=len(documents)):
            # 1. Découper en chunks
            self.document_chunks = self._split_documents_to_chunks(documents)
            if not self.document_chunks:
                logfire.error("Aucun chunk produit, index non construit")
                return

            # 2. Générer les embeddings
            embeddings = self._generate_embeddings(self.document_chunks)
            if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
                logfire.error("Incohérence embeddings/chunks, index non construit")
                self.document_chunks = []
                self.index = None
                if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
                if os.path.exists(DOCUMENT_CHUNKS_FILE): os.remove(DOCUMENT_CHUNKS_FILE)
                return

            # 3. Créer l'index Faiss optimisé pour la similarité cosinus
            dimension = embeddings.shape[1]
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            logfire.info("Index FAISS construit", nb_vecteurs=self.index.ntotal, dimension=dimension)

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

        with logfire.span("recherche-faiss", question=query_text, k=k):
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

                logfire.info(
                    "Résultats recherche FAISS",
                    nb_resultats=len(results),
                    score_max=results[0].score if results else 0,
                    score_min=results[-1].score if results else 0,
                )

                return results

            except SDKError as e:
                logfire.error("Erreur API Mistral embedding requête", erreur=str(e))
                return []
            except Exception as e:
                logfire.error("Erreur recherche FAISS", erreur=str(e))
                return []
