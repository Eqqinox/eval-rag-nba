# Assistant RAG avec Mistral

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée. L'assistant est conçu pour l'analyse de la performance sportive en basketball (NBA).

## Fonctionnalités

- **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- **Génération de réponses** avec le modèle Mistral (mistral-small-latest)
- **Extraction multi-format** avec OCR intégré pour les PDF scannés
- **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets Python)
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer et activer l'environnement virtuel**

```bash
uv venv
source .venv/bin/activate
```

3. **Initialiser le projet et installer les dépendances**

```bash
uv init
uv add -r requirements.txt
```

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
.
├── MistralChat.py              # Application Streamlit (interface RAG)
├── indexer.py                  # Script d'indexation des documents
├── main.py                    # Point d'entrée (placeholder)
├── requirements.txt           # Dépendances Python
├── pyproject.toml             # Métadonnées du projet (uv/pip)
├── .env                       # Clé API Mistral (non versionné)
├── inputs/                    # Documents sources
│   ├── Reddit 1.pdf           # Archive textuelle NBA
│   ├── Reddit 2.pdf           # Archive textuelle NBA
│   ├── Reddit 3.pdf           # Archive textuelle NBA
│   ├── Reddit 4.pdf           # Archive textuelle NBA
│   └── regular NBA.xlsx       # Données statistiques NBA
├── vector_db/                 # Index vectoriel pré-construit
│   ├── faiss_index.idx        # Index FAISS (recherche par similarité)
│   └── document_chunks.pkl    # Métadonnées des chunks (pickle)
└── utils/                     # Modules utilitaires
    ├── config.py              # Configuration centralisée
    ├── data_loader.py         # Extraction et parsing des documents
    └── vector_store.py        # Gestion de l'index vectoriel FAISS
```

## Pipeline RAG

Le flux de données du système suit les étapes suivantes :

```
Documents sources (inputs/)
        |
        v
  data_loader.py        -- Extraction de texte (PDF, DOCX, CSV, Excel, TXT) + OCR
        |
        v
  vector_store.py       -- Découpage en chunks (1500 car., 150 chevauchement)
        |
        v
  API Mistral Embed     -- Génération des embeddings (modèle mistral-embed)
        |
        v
  Index FAISS           -- Stockage vectoriel (similarité cosinus via IndexFlatIP)
        |
        v
  vector_db/            -- Persistance sur disque (faiss_index.idx + document_chunks.pkl)
        |
        v
  MistralChat.py        -- Question utilisateur -> recherche FAISS -> contexte + prompt
        |
        v
  API Mistral LLM       -- Génération de la réponse (mistral-small-latest, température 0.1)
        |
        v
  Réponse + sources     -- Affichage avec documents sources et scores de pertinence
```

## Utilisation

### 1. Ajouter des documents

Placez vos documents dans le dossier `inputs/`. Les formats supportés sont :
- PDF (avec fallback OCR via EasyOCR pour les documents scannés)
- DOCX
- TXT
- CSV
- Excel (XLSX, XLS)

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
python indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Extraire le texte de chaque document (avec OCR si nécessaire)
3. Découper les documents en chunks de 1500 caractères (chevauchement de 150)
4. Générer des embeddings via l'API Mistral (modèle mistral-embed, par lots de 32)
5. Créer un index FAISS normalisé pour la recherche par similarité cosinus
6. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

Options disponibles :

```bash
python indexer.py --input-dir <dossier>       # Dossier source alternatif
python indexer.py --data-url <url>            # Télécharger un zip de documents
```

### 3. Lancer l'application

```bash
streamlit run MistralChat.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.

## Modules principaux

### `utils/config.py`

Configuration centralisée de l'application :
- Clé API Mistral (chargée depuis `.env`)
- Modèles utilisés (embedding et génération)
- Paramètres de chunking (taille, chevauchement)
- Paramètres de recherche (nombre de résultats, score minimum)
- Chemins des répertoires (inputs, vector_db)

### `utils/data_loader.py`

Extraction et parsing des documents sources :
- Lecture des PDF avec PyPDF2 et fallback OCR via EasyOCR (anglais et français)
- Parsing des fichiers DOCX, TXT, CSV et Excel (multi-feuilles)
- Parcours récursif des dossiers avec suivi des sous-dossiers
- Conservation des métadonnées (nom de fichier, chemin source, catégorie)
- Téléchargement et extraction de fichiers zip depuis une URL

### `utils/vector_store.py`

Gestion de l'index vectoriel FAISS et recherche sémantique :
- Découpage des documents en chunks via LangChain (RecursiveCharacterTextSplitter)
- Génération des embeddings par lots via l'API Mistral
- Création de l'index FAISS (IndexFlatIP avec normalisation L2 pour similarité cosinus)
- Recherche sémantique avec scores de pertinence (0-100%)
- Persistance de l'index et des chunks sur disque

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `mistral-embed` | Modèle pour la vectorisation du texte |
| `MODEL_NAME` | `mistral-small-latest` | Modèle LLM pour la génération de réponses |
| `CHUNK_SIZE` | `1500` | Taille des chunks en caractères |
| `CHUNK_OVERLAP` | `150` | Chevauchement entre chunks en caractères |
| `EMBEDDING_BATCH_SIZE` | `32` | Taille des lots pour l'API d'embedding |
| `SEARCH_K` | `5` | Nombre de résultats retournés par recherche |
