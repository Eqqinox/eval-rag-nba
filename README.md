# Assistant RAG + SQL avec Mistral

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Mistral AI](https://img.shields.io/badge/Mistral_AI-FF7000?logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-1.10-0467DF?logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-évaluation-7C3AED?logoColor=white)
![Logfire](https://img.shields.io/badge/Logfire-monitoring-E92063?logoColor=white)

Ce projet implémente un assistant virtuel piloté par Mistral AI, dédié à l'analyse des performances des joueurs et équipes de la NBA. Il articule un moteur de recherche vectorielle (RAG) pour le traitement des données textuelles et un agent SQL pour l'interrogation de données structurées, alliant analyse contextuelle et précision statistique.
---

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Architecture de l'agent](#architecture-de-lagent)
- [Pipeline de préparation des données](#pipeline-de-préparation-des-données)
- [Base de données SQL](#base-de-données-sql)
- [Modules principaux](#modules-principaux)
- [Évaluation RAGAS](#évaluation-ragas)
- [Traçabilité Logfire](#traçabilité-logfire)
- [Résultats et comparatif des métriques](#résultats-et-comparatif-des-métriques)
- [Personnalisation](#personnalisation)
- [Licence](#licence)
- [Auteur](#auteur)
---

## Fonctionnalités

- **Router intelligent** : le LLM classifie automatiquement les questions (SQL ou RAG)
- **Recherche sémantique** avec FAISS pour les questions textuelles/narratives
- **Requêtes SQL** pour les questions chiffrées/statistiques
- **Génération de réponses** avec le modèle Mistral (mistral-small-latest)
- **Extraction multi-format** avec OCR intégré pour les PDF scannés
- **Validation Pydantic** à chaque étape du pipeline
- **Traçabilité Logfire** pour le monitoring en temps réel
---

## Prérequis

- Python 3.11+
- Gestionnaire de paquets Python ([uv](https://docs.astral.sh/uv/))
- Clé API Mistral ([console.mistral.ai](https://console.mistral.ai/))
-  _Token Logfire (optionnel -_ [logfire-eu.pydantic.dev](https://logfire-eu.pydantic.dev/))
---

## Installation

1. **Cloner le dépôt**

```bash
$ git clone https://github.com/Eqqinox/eval-rag-nba
$ cd eval-rag-nba
```

2. **Créer et activer l'environnement virtuel**

```bash
$ uv venv
$ source .venv/bin/activate
```

3. **Initialiser le projet et installer les dépendances**

```bash
$ uv init
$ uv add -r requirements.txt
```

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=clé_api_mistral
LOGFIRE_TOKEN=token_logfire
```
---

## Utilisation

### <u>1. Ajouter des documents</u>

Placez vos documents dans le dossier `inputs/`. Les formats supportés sont :
- PDF (avec fallback OCR via EasyOCR pour les documents scannés)
- DOCX
- TXT
- CSV
- Excel (XLSX, XLS)

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### <u>2. Indexer les documents</u>

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
$ python indexer.py
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
$ python indexer.py --input-dir <dossier>       # Dossier source alternatif
$ python indexer.py --data-url <url>            # Télécharger un zip de documents
```

### <u>3. Lancer l'application</u>

```bash
$ streamlit run MistralChat.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.

---

## Structure du projet

```
.
├── database/                   # Base de données SQL
│   ├── models.py               # Modèles SQLAlchemy + schémas Pydantic
│   ├── load_excel_to_db.py     # Pipeline d'ingestion Excel -> SQLite
│   ├── sql_tool.py             # Tool SQL LangChain
│   └── nba.db                  # Base SQLite (30 équipes, 568 joueurs)
│ 
├── evaluation_results/         # Résultats de l'évaluation RAGAS
│   ├── detailed_scores.csv     # Scores par question
│   ├── aggregate_scores.csv    # Moyennes par catégorie
│   └── evaluation_summary.json # Récapitulatif JSON
│ 
├── inputs/                     # Documents sources
│   ├── Reddit 1.pdf            # Archive textuelle NBA
│   ├── Reddit 2.pdf            # Archive textuelle NBA
│   ├── Reddit 3.pdf            # Archive textuelle NBA
│   ├── Reddit 4.pdf            # Archive textuelle NBA
│   └── regular NBA.xlsx        # Données statistiques NBA
│ 
├── utils/                      # Modules utilitaires
│   ├── config.py               # Configuration centralisée (Pydantic Settings)
│   ├── schemas.py              # Schémas de validation Pydantic
│   ├── data_loader.py          # Extraction et parsing des documents
│   ├── vector_store.py         # Gestion de l'index vectoriel FAISS
│   └── embedding_agent.py      # Agent Pydantic AI pour embeddings
│ 
├── vector_db/                  # Index vectoriel pré-construit
│   ├── faiss_index.idx         # Index FAISS (recherche par similarité)
│   └── document_chunks.pkl     # Métadonnées des chunks (pickle)
│ 
├── .env                        # Clé API / Token (non versionné)
├── .gitignore                  # Fichiers exclus du versionnement
├── .python-version             # Version Python fixée pour uv
├── evaluate_ragas.py           # Script d'évaluation RAGAS
├── indexer.py                  # Script d'indexation des documents
├── MistralChat.py              # Application Streamlit (interface RAG + SQL)
├── pyproject.toml              # Métadonnées du projet (uv/pip)
├── README.md                   # Documentation du projet
├── requirements.txt            # Dépendances Python
├── test_questions.json         # Questions de test (30 questions, 3 catégories)
└── uv.lock                     # Verrou des dépendances (uv)
```
---

## Architecture de l'agent

L'agent utilise un router LLM pour diriger automatiquement les questions vers le bon pipeline :

```
                            Question utilisateur
                                    │
                                    ▼
                            ┌───────────────┐
                            │  Router LLM   │
                            │ classification│
                            └───────┬───────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │      SQL      │               │      RAG      │
            │   (chiffres)  │               │    (texte)    │
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
            │    nba.db     │               │  faiss_index  │
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
                                 Réponse
```

### <u>Critères de classification du router</u>

| Type | Critères | Exemples |
|------|----------|----------|
| **SQL** | Statistiques, chiffres, classements, comparaisons chiffrées | "Meilleur marqueur", "Stats de Jokic", "Top 5 rebondeurs" |
| **RAG** | Analyses, histoire, opinions, événements | "Pourquoi les Lakers ont perdu ?", "Rumeurs de transfert" |
---

## Pipeline de préparation des données

### <u>Pipeline RAG (documents textuels)</u>

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

### <u>Pipeline SQL (données structurées)</u>

```
inputs/regular NBA.xlsx
        │
        ▼
 load_excel_to_db.py    -- Lecture Excel + validation Pydantic
        │
        ▼
  database/nba.db       -- Base SQLite (tables teams, players)
```

Pour charger (ou recharger) les données Excel dans la base SQLite :

```bash
$ python database/load_excel_to_db.py
```
---

## Base de données SQL

Le système utilise une base de données SQLite (`database/nba.db`) pour stocker les statistiques structurées des joueurs NBA. Cela permet de répondre aux questions chiffrées via des requêtes SQL plutôt que par recherche sémantique.

### <u>Schéma relationnel</u>

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

<u>**Table `teams`**</u>
| Colonne | Type | Description |
|---------|------|-------------|
| `code` | VARCHAR(3), PK | Code équipe (ex: LAL, BOS, OKC) |
| `name` | VARCHAR(100) | Nom complet (ex: Los Angeles Lakers) |  

<br>

<u>**Table `players`**</u>
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

### <u>Exemples de requêtes SQL</u>

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
---

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
---

## Évaluation RAGAS

Le script `evaluate_ragas.py` reproduit le pipeline RAG en mode headless et évalue les réponses avec le framework RAGAS.

### <u>Commande d'exécution</u>

```bash
$ KMP_DUPLICATE_LIB_OK=TRUE python evaluate_ragas.py
```

> La variable `KMP_DUPLICATE_LIB_OK=TRUE` est nécessaire sur macOS pour éviter un conflit OpenMP entre FAISS et PyTorch.

### <u>Jeu de questions de test</u>

Le fichier `test_questions.json` contient 30 questions réparties en 3 catégories :

| Catégorie | Nombre | Description | Exemple |
|-----------|--------|-------------|---------|
| `simple` | 10 | Questions factuelles avec réponse attendue (ground truth) | "Quel joueur a marqué le plus de points ?" |
| `complex` | 10 | Questions d'analyse multi-critères | "Compare les performances de Jokic et Giannis" |
| `noisy` | 10 | Questions mal formulées (langage SMS, fautes) | "mec c koi le joueur ki score le + de 3pts" |

### <u>Métriques RAGAS</u>

| Métrique | Description | Nécessite ground truth |
|----------|-------------|------------------------|
| `faithfulness` | La réponse est-elle fidèle aux contextes récupérés ? | Non |
| `answer_relevancy` | La réponse répond-elle à la question ? | Non |
| `context_precision` | Les contextes récupérés sont-ils pertinents ? | Oui |
| `context_recall` | Les contextes couvrent-ils la réponse attendue ? | Oui |

Les résultats sont enregistrés dans le dossier `evaluation_results/` :
- `detailed_scores.csv` : scores par question
- `aggregate_scores.csv` : moyennes par catégorie
- `evaluation_summary.json` : récapitulatif JSON
---

## Traçabilité Logfire

Logfire permet de visualiser l'exécution du pipeline RAG en temps réel via un dashboard dédié.

Le token `LOGFIRE_TOKEN` doit être défini dans le fichier `.env`. S'il est absent, Logfire est désactivé sans erreur.

**Dashboard** : https://logfire-eu.pydantic.dev/eqqinox/llm-eval

### <u>Spans instrumentés</u>

| Composant | Span | Informations tracées |
|-----------|------|---------------------|
| `vector_store.py` | `chargement-index-et-chunks` | Nombre de vecteurs et chunks chargés |
| `vector_store.py` | `decoupage-chunks` | Taille, chevauchement, nombre de chunks produits |
| `vector_store.py` | `generation-embeddings` | Nombre de lots, dimension, erreurs par lot |
| `vector_store.py` | `construction-index` | Nombre de vecteurs indexés, dimension |
| `vector_store.py` | `recherche-faiss` | Question, k, scores min/max |
| `embedding_agent.py` | `agent-embedding` | Nombre de textes traités |
| `MistralChat.py` | `pipeline-agent-complet` | Question posée (span parent) |
| `MistralChat.py` | `router-llm` | Classification de la question |
| `MistralChat.py` | `pipeline-sql` | Traitement SQL |
| `MistralChat.py` | `pipeline-rag` | Traitement RAG |
| `MistralChat.py` | `appel-llm` | Modèle, longueur réponse |
| `database/sql_tool.py` | `tool-sql-nba` | Question, SQL généré |
| `database/sql_tool.py` | `generation-sql` | Requête SQL produite |
| `database/sql_tool.py` | `execution-sql` | Résultats de la requête |
| `evaluate_ragas.py` | `evaluation-ragas-complete` | Évaluation complète |
---

## Résultats et comparatif des métriques

### <u>Évaluation initiale</u>

| Catégorie | faithfulness | answer_relevancy | context_precision | context_recall |
|-----------|-------------|-----------------|------------------|---------------|
| simple | 0.6759 | 0.4064 | 0.3000 | 0.1867 |
| complex | 0.6667 | 0.7943 | -- | -- |
| noisy | 0.1494 | 0.6155 | -- | -- |
| **global** | **0.4639** | **0.6054** | **0.3000** | **0.1867** |

### <u>Évaluation après enrichissement SQL</u>

Répartition : 22 questions traitées par SQL, 8 par RAG.

| Catégorie | faithfulness | answer_relevancy | context_precision | context_recall |
|-----------|-------------|-----------------|------------------|---------------|
| simple | 0.8000 | 0.8864 | 0.7000 | 0.6000 |
| complex | 0.8145 | 0.7200 | -- | -- |
| noisy | 0.6378 | 0.7324 | -- | -- |
| **global** | **0.7544** | **0.7796** | **0.7000** | **0.6000** |

### <u>Comparatif global</u>

| Métrique | Avant (RAG seul) | Après (RAG + SQL) | Amélioration |
|----------|------------------|-------------------|--------------|
| faithfulness | 0.4639 | 0.7544 | **+62.6 %** |
| answer_relevancy | 0.6054 | 0.7796 | **+28.8 %** |
| context_precision | 0.3000 | 0.7000 | **+133.3 %** |
| context_recall | 0.1867 | 0.6000 | **+221.4 %** |

### <u>Conclusions</u>

L'intégration du pipeline SQL a produit des améliorations significatives sur l'ensemble des métriques :

- **faithfulness (+62.6 %)** : les réponses sont plus fidèles aux données sources car le Tool SQL retourne des valeurs exactes issues de la base.
- **answer_relevancy (+28.8 %)** : le router LLM dirige les questions chiffrées vers le bon pipeline, produisant des réponses plus pertinentes.
- **context_precision (+133.3 %) et context_recall (+221.4 %)** : les résultats SQL remplacent la recherche sémantique FAISS pour les questions factuelles, éliminant les faux positifs du retriever.
- **Questions bruitées** : la faithfulness passe de 0.15 à 0.64 car le Tool SQL est insensible au bruit lexical (il extrait la sémantique de la question avant de générer la requête SQL).
- **Questions complexes** : légère baisse de answer_relevancy (0.79 à 0.72) compensée par une meilleure faithfulness (0.67 à 0.81), indiquant des réponses plus factuellement ancrées mais parfois moins exhaustives sur les aspects narratifs.

La recherche sémantique seule est peu adaptée pour interroger des données structurées et chiffrées. L'architecture hybride RAG + SQL est l'approche recommandée pour ce type d'assistant.

---

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
---

## Licence

Projet académique - Formation Expert en Ingénierie et Science des Données

Ce projet a été réalisé dans le cadre d'un parcours de formation et n'est pas destiné à un usage commercial.

---

## Auteur

**Mounir Meknaci**

- Email : meknaci81@gmail.com
- LinkedIn : [Mounir Meknaci](https://www.linkedin.com/in/mounir-meknaci/)
- Formation : Expert en ingénierie et science des données
- Projet : Évaluez les performances d'un LLM

---

*Dernière mise à jour: Février 2026*  
*Projet eval-rag-nba  - OpenClassrooms*.  
*Auteur : Mounir Meknaci*.  
*Version : 1.0*
