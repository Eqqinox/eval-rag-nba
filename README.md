# Assistant RAG + SQL avec Mistral

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant une architecture hybride RAG + SQL pour fournir des réponses précises et contextuelles. L'assistant est conçu pour l'analyse de la performance sportive en basketball (NBA).

## Fonctionnalités

- **Router intelligent** : le LLM classifie automatiquement les questions (SQL ou RAG)
- **Recherche sémantique** avec FAISS pour les questions textuelles/narratives
- **Requêtes SQL** pour les questions chiffrées/statistiques
- **Génération de réponses** avec le modèle Mistral (mistral-small-latest)
- **Extraction multi-format** avec OCR intégré pour les PDF scannés
- **Validation Pydantic** à chaque étape du pipeline
- **Traçabilité Logfire** pour le monitoring en temps réel

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
├── MistralChat.py              # Application Streamlit (interface RAG + SQL)
├── indexer.py                  # Script d'indexation des documents
├── evaluate_ragas.py           # Script d'évaluation RAGAS
├── test_questions.json         # Questions de test (30 questions, 3 catégories)
├── requirements.txt            # Dépendances Python
├── pyproject.toml              # Métadonnées du projet (uv/pip)
├── .env                        # Clé API Mistral (non versionné)
├── inputs/                     # Documents sources
│   ├── Reddit 1.pdf            # Archive textuelle NBA
│   ├── Reddit 2.pdf            # Archive textuelle NBA
│   ├── Reddit 3.pdf            # Archive textuelle NBA
│   ├── Reddit 4.pdf            # Archive textuelle NBA
│   └── regular NBA.xlsx        # Données statistiques NBA
├── vector_db/                  # Index vectoriel pré-construit
│   ├── faiss_index.idx         # Index FAISS (recherche par similarité)
│   └── document_chunks.pkl     # Métadonnées des chunks (pickle)
├── database/                   # Base de données SQL
│   ├── models.py               # Modèles SQLAlchemy + schémas Pydantic
│   ├── load_excel_to_db.py     # Pipeline d'ingestion Excel -> SQLite
│   ├── sql_tool.py             # Tool SQL LangChain
│   └── nba.db                  # Base SQLite (30 équipes, 568 joueurs)
├── evaluation_results/         # Résultats de l'évaluation RAGAS
│   ├── detailed_scores.csv     # Scores par question
│   ├── aggregate_scores.csv    # Moyennes par catégorie
│   └── evaluation_summary.json # Récapitulatif JSON
└── utils/                      # Modules utilitaires
    ├── config.py               # Configuration centralisée (Pydantic Settings)
    ├── schemas.py              # Schémas de validation Pydantic
    ├── data_loader.py          # Extraction et parsing des documents
    ├── vector_store.py         # Gestion de l'index vectoriel FAISS
    └── embedding_agent.py      # Agent Pydantic AI pour embeddings
```

## Architecture de l'agent

L'agent utilise un router LLM pour diriger automatiquement les questions vers le bon pipeline :

```
                            Question utilisateur
                                    │
                                    ▼
                            ┌───────────────┐
                            │  Router LLM   │
                            │ (classification)│
                            └───────┬───────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │     SQL       │               │     RAG       │
            │  (chiffres)   │               │   (texte)     │
            └───────┬───────┘               └───────┬───────┘
                    │                               │
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │  sql_tool.py  │               │ vector_store  │
            │  Génère SQL   │               │ Recherche     │
            │  Exécute      │               │ FAISS         │
            └───────┬───────┘               └───────┬───────┘
                    │                               │
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │  database/    │               │  vector_db/   │
            │   nba.db      │               │  faiss_index  │
            └───────┬───────┘               └───────┬───────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │  Mistral LLM  │
                            │  (synthèse)   │
                            └───────┬───────┘
                                    │
                                    ▼
                            Réponse + sources
```

### Classification du router

| Type | Critères | Exemples |
|------|----------|----------|
| **SQL** | Statistiques, chiffres, classements, comparaisons chiffrées | "Meilleur marqueur", "Stats de Jokic", "Top 5 rebondeurs" |
| **RAG** | Analyses, histoire, opinions, événements | "Pourquoi les Lakers ont perdu ?", "Rumeurs de transfert" |

## Pipeline de préparation des données

### Pipeline RAG (documents textuels)

```
Documents sources (inputs/)
        │
        ▼
  data_loader.py        -- Extraction de texte (PDF, DOCX, CSV, Excel, TXT) + OCR
        │
        ▼
  vector_store.py       -- Découpage en chunks (1500 car., 150 chevauchement)
        │
        ▼
  API Mistral Embed     -- Génération des embeddings (modèle mistral-embed)
        │
        ▼
  Index FAISS           -- Stockage vectoriel (similarité cosinus via IndexFlatIP)
        │
        ▼
  vector_db/            -- Persistance sur disque (faiss_index.idx + document_chunks.pkl)
```

### Pipeline SQL (données structurées)

```
inputs/regular NBA.xlsx
        │
        ▼
  load_excel_to_db.py   -- Lecture Excel + validation Pydantic
        │
        ▼
  database/nba.db       -- Base SQLite (tables teams, players)
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

## Base de données SQL

Le système utilise une base de données SQLite (`database/nba.db`) pour stocker les statistiques structurées des joueurs NBA. Cela permet de répondre aux questions chiffrées via des requêtes SQL plutôt que par recherche sémantique.

### Schéma relationnel

```
┌─────────────────────┐       ┌─────────────────────────────────────┐
│       teams         │       │              players                │
├─────────────────────┤       ├─────────────────────────────────────┤
│ code (PK, VARCHAR)  │◄──────│ team_code (FK)                      │
│ name (VARCHAR)      │       │ id (PK, INTEGER)                    │
└─────────────────────┘       │ name (VARCHAR)                      │
                              │ age (INTEGER)                       │
                              │ games_played (INTEGER)              │
                              │ wins (INTEGER)                      │
                              │ losses (INTEGER)                    │
                              │ minutes_per_game (FLOAT)            │
                              │ points_per_game (FLOAT)             │
                              │ field_goals_made (FLOAT)            │
                              │ field_goals_attempted (FLOAT)       │
                              │ field_goal_pct (FLOAT)              │
                              │ three_pointers_made (FLOAT)         │
                              │ three_pointers_attempted (FLOAT)    │
                              │ three_point_pct (FLOAT)             │
                              │ free_throws_made (FLOAT)            │
                              │ free_throws_attempted (FLOAT)       │
                              │ free_throw_pct (FLOAT)              │
                              │ offensive_rebounds (FLOAT)          │
                              │ defensive_rebounds (FLOAT)          │
                              │ total_rebounds (FLOAT)              │
                              │ assists (FLOAT)                     │
                              │ turnovers (FLOAT)                   │
                              │ steals (FLOAT)                      │
                              │ blocks (FLOAT)                      │
                              │ personal_fouls (FLOAT)              │
                              │ fantasy_points (FLOAT)              │
                              │ double_doubles (INTEGER)            │
                              │ triple_doubles (INTEGER)            │
                              │ plus_minus (FLOAT)                  │
                              │ offensive_rating (FLOAT)            │
                              │ defensive_rating (FLOAT)            │
                              │ net_rating (FLOAT)                  │
                              │ assist_pct (FLOAT)                  │
                              │ assist_to_turnover (FLOAT)          │
                              │ assist_ratio (FLOAT)                │
                              │ offensive_rebound_pct (FLOAT)       │
                              │ defensive_rebound_pct (FLOAT)       │
                              │ total_rebound_pct (FLOAT)           │
                              │ turnover_ratio (FLOAT)              │
                              │ effective_fg_pct (FLOAT)            │
                              │ true_shooting_pct (FLOAT)           │
                              │ usage_rate (FLOAT)                  │
                              │ pace (FLOAT)                        │
                              │ player_impact_estimate (FLOAT)      │
                              │ possessions (FLOAT)                 │
                              └─────────────────────────────────────┘
```

### Tables

**Table `teams`** (30 lignes)
| Colonne | Type | Description |
|---------|------|-------------|
| `code` | VARCHAR(3), PK | Code équipe (ex: LAL, BOS, OKC) |
| `name` | VARCHAR(100) | Nom complet (ex: Los Angeles Lakers) |

**Table `players`** (569 lignes)
| Catégorie | Colonnes |
|-----------|----------|
| Identité | `id`, `name`, `team_code`, `age` |
| Matchs | `games_played`, `wins`, `losses`, `minutes_per_game` |
| Points | `points_per_game` |
| Tirs | `field_goals_made/attempted`, `field_goal_pct`, `three_pointers_made/attempted`, `three_point_pct`, `free_throws_made/attempted`, `free_throw_pct` |
| Rebonds | `offensive_rebounds`, `defensive_rebounds`, `total_rebounds` |
| Passes/Pertes | `assists`, `turnovers`, `assist_to_turnover`, `assist_pct`, `assist_ratio` |
| Défense | `steals`, `blocks` |
| Avancées | `offensive_rating`, `defensive_rating`, `net_rating`, `effective_fg_pct`, `true_shooting_pct`, `usage_rate`, `pace`, `player_impact_estimate` |

### Exemples de requêtes SQL

```sql
-- Meilleur marqueur de la saison
SELECT name, team_code, points_per_game
FROM players
ORDER BY points_per_game DESC
LIMIT 1;

-- Statistiques d'un joueur spécifique
SELECT name, points_per_game, total_rebounds, assists
FROM players
WHERE name LIKE '%Jokic%';

-- Meilleurs passeurs par équipe
SELECT team_code, name, assists
FROM players
WHERE assists = (
    SELECT MAX(assists) FROM players p2 WHERE p2.team_code = players.team_code
)
ORDER BY assists DESC;

-- Comparaison de deux joueurs
SELECT name, points_per_game, total_rebounds, assists, field_goal_pct
FROM players
WHERE name IN ('Nikola Jokic', 'Giannis Antetokounmpo');
```

### Commandes

```bash
# Charger les données Excel dans la base SQLite
python database/load_excel_to_db.py
```

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
