# Rapport de mise en place et d'évaluation du système RAG

**Projet** : Évaluation des performances d'un LLM <br>
**Entreprise** : SportSee<br>
**Domaine** : Analyse de performances sportives (NBA)<br>
**Auteur** : Mounir Meknaci<br>
**Date** : Février 2026<br>

---

## Table des matières

- [Introduction](#introduction)
- [Contexte et problématique](#contexte-et-problématique)
- [Méthodologie](#méthodologie)
- [Choix techniques et justifications](#choix-techniques-et-justifications)
- [Architecture du système](#architecture-du-système)
- [Résultats de l'évaluation](#résultats-de-lévaluation)
- [Points de vigilance](#points-de-vigilance)
- [Recommandations](#recommandations)
- [Conclusion](#conclusion)

---

## Introduction

Ce rapport présente la mise en place et l'évaluation d'un système RAG (Retrieval-Augmented Generation) hybride destiné à l'analyse de performances sportives dans le contexte de la NBA. Le projet s'inscrit dans le cadre d'une mission pour SportSee, startup spécialisée en intelligence artificielle appliquée au sport.

L'objectif principal consiste à fiabiliser un prototype d'assistant intelligent permettant aux entraîneurs, analystes et préparateurs physiques d'interroger rapidement des archives de données (textuelles et chiffrées) pour obtenir des informations précises sur les performances des joueurs et des équipes.

---

## Contexte et problématique

### Situation initiale

SportSee dispose d'un prototype d'assistant IA fonctionnel basé sur un système RAG, capable d'interroger des archives textuelles (discussions Reddit, analyses de matchs). Les premiers tests montrent des **résultats encourageants sur les questions d'analyse textuelle**, mais les réponses ne sont **pas satisfaisantes** pour certains types de requêtes. Le système n'est **pas encore assez robuste** pour envisager une mise en production.

Une évaluation objective est nécessaire pour identifier précisément les forces et faiblesses du prototype.

### Problématique métier

Les utilisateurs cibles (entraîneurs, analystes et préparateurs physiques) ont besoin de réponses à deux types de questions :

1. **Questions statistiques** : "Quel joueur a le meilleur pourcentage à 3 points sur les 5 derniers matchs ?"
2. **Questions d'analyse** : "Pourquoi les Lakers sont-ils considérés comme favoris pour les playoffs ?"

Le système RAG initial, conçu pour la recherche sémantique dans des documents textuels, doit être évalué pour déterminer s'il peut répondre efficacement aux deux types de questions.

### Objectifs du projet

1. Évaluer objectivement les performances du système RAG initial
2. Identifier les axes d'amélioration prioritaires
3. Mettre en œuvre les solutions adaptées aux axes identifiés
4. Mesurer l'impact des améliorations sur la qualité des réponses
5. Documenter les biais, limites et recommandations pour une future mise en production

---

## Méthodologie

### Phase 1 : Audit du système existant

Dans le but d'établir une baseline de performances du système, nous avons :
- Constitué un jeu de test de 30 questions catégorisées (simples, complexes, bruitées).
- Mis en place un script d'évaluation automatisé avec le framework RAGAS (`evaluate_ragas.py`).
- Mesuré 4 métriques clés : **faithfulness**, **answer_relevancy**, **context_precision**, **context_recall**.

**Résultats** :

| Catégorie | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|-----------|--------------|------------------|-------------------|----------------|
| Simple    | 0.68         | 0.41             | 0.30              | 0.19           |
| Complex   | 0.67         | 0.79             | --                | --             |
| Noisy     | 0.15         | 0.62             | --                | --             |
| **Global** | **0.46**    | **0.61**         | **0.30**          | **0.19**       |


Le **context_recall** est très faible (0.19) ce qui indique que le retriever FAISS ne trouve pas les bons documents pour les questions factuelles.<br>
Les **questions bruitées** obtiennent un faithfulness de 0.15, montrant une forte sensibilité au bruit lexical.<br>
Le **score global de faithfulness** (0.46) est insuffisant pour une mise en production.

**Conclusion** :<br>
Le système RAG seul est peu adapté pour traiter des questions nécessitant des données chiffrées précises.

---
### Phase 2 : Sécurisation du pipeline avec Pydantic

Pour garantir la cohérence des données à chaque étape du pipeline, nous avons :
- Créé des schémas Pydantic pour valider les entrées/sorties (Document, Chunk, EmbeddingResult, SearchResult).
- Remplacé les constantes de configuration par Pydantic Settings.
- Mis en place une validation automatique des données Excel avant insertion en base (TeamSchema, PlayerSchema).
- Instrumenté le code avec Pydantic AI pour les agents d'embedding.

**Avantages** :
- Détection précoce des données invalides (exemple : 1 joueur écarté car statistiques > 100%).
- Traçabilité des erreurs de validation.
- Code plus robuste et maintenable.

---
### Phase 3 : Intégration de Pydantic Logfire

Afin d'obtenir un suivi complet de l'exécution du pipeline, nous avons :
- Configuré Logfire avec instrumentation automatique des modèles Pydantic.
- Ajouté 29 spans manuels pour tracer les étapes critiques (chargement index, génération embeddings, recherche FAISS, appels LLM, ingestion données).
- Mis en place un dashboard de monitoring en temps réel.

**Avantages** :
- Diagnostic rapide des problèmes de performance.
- Identification des goulots d'étranglement.
- Validation du bon fonctionnement du pipeline en production.

---
### Phase 4 : Enrichissement avec un système hybride RAG + SQL

De manière à traiter les questions chiffrées avec précision, nous avons :
- Modélisé une base de données SQLite (tables teams et players).
- Créé un pipeline d'ingestion Excel → SQLite avec validation Pydantic.
- Créé un Tool SQL LangChain avec génération dynamique de requêtes.
- Implémenté un router LLM pour classifier automatiquement les questions (SQL ou RAG).
- Intégré le router dans l'application Streamlit.

**Architecture finale** :

```
                        ┌───────────────┐    ┌──────────────────────────┐ 
                    ┌──►│      SQL      ├───►│ Synthèse par Mistral LLM │
 ┌───────────────┐  │   │(si chiffrées) │    └──────────────────────────┘
 │  Router LLM   ├──┤   └───────────────┘
 └───────────────┘  │   ┌───────────────┐    ┌──────────────────────────┐
                    └──►│      RAG      ├───►│ Synthèse par Mistral LLM │
                        │(si textuelles)│    └──────────────────────────┘
                        └───────────────┘
```
Vous pouvez retrouver l'architecture complète dans la section Architecture du système.

---
### Phase 5 : Seconde évaluation et analyse comparative

Pour mesurer l'impact de l'enrichissement SQL, nous avons :
- Réexécuté le script d'évaluation RAGAS sur le système hybride.
- Comparé les métriques avant/après.
- Analysé les biais du router et les limites du mapping NL→SQL.
- Documenté les cas limites et les axes d'amélioration.

---

## Choix techniques et justifications

### Base de données : SQLite vs PostgreSQL

**Choix** : SQLite

**Justification** :
- Volume de données modéré (30 équipes, 568 joueurs).
- Déploiement simplifié (fichier unique, pas de serveur).
- Performance suffisante pour les requêtes analytiques du projet.
- Portabilité maximale (base versionnée avec le code).

---
### Router : LLM vs règles fixes

**Choix** : Router LLM (Mistral)

**Justification** :
- Flexibilité : s'adapte à des formulations variées sans reprogrammation.
- Robustesse au bruit : capable de classifier correctement malgré des fautes d'orthographe.
- Évolutivité : amélioration possible via ajustement du prompt (pas de refonte du code).

---
### Vector store : FAISS vs alternatives

**Choix** : FAISS (Facebook AI Similarity Search)

**Justification** :
- Performance : recherche de similarité optimisée (normalisation L2 + IndexFlatIP pour similarité cosinus).
- Maturité : bibliothèque éprouvée, largement utilisée en production.
- Simplicité : pas de serveur externe requis (stockage local sur disque).

**Alternatives possibles** :
- Pinecone, Weaviate (dépendance à un service cloud externe, coût récurrent).

---
### LLM : Mistral vs alternatives

**Choix** : Mistral AI (modèle mistral-small-latest)

**Justification** :
- Équilibre performance/coût optimal pour le cas d'usage.
- Endpoint compatible OpenAI (facilite l'intégration avec LangChain).
- Latence acceptable (<2s pour la génération de réponses).

**Alternatives envisagées** :
- GPT-4 (coût trop élevé pour un prototype).
- Llama 2 local (infrastructure GPU requise).

---
### Framework d'évaluation : RAGAS

**Choix** : RAGAS (Retrieval-Augmented Generation Assessment)

**Justification** :
- Métriques spécifiquement conçues pour les systèmes RAG.
- Support natif de Pydantic pour la validation des datasets.
- Open source et bien documenté.

**Métriques retenues** :
- `faithfulness` : fidélité de la réponse aux contextes récupérés (détecte les hallucinations).
- `answer_relevancy` : pertinence de la réponse par rapport à la question.
- `context_precision` : pertinence des contextes récupérés (nécessite ground truth).
- `context_recall` : couverture de la réponse attendue par les contextes (nécessite ground truth).

---
### Validation des données : Pydantic

**Choix** : Pydantic v2 + Pydantic AI

**Justification** :
- Validation à la compilation (détection précoce des erreurs).
- Génération automatique de schémas JSON pour la documentation.
- Intégration native avec LangChain.
- Support des agents structurés (Pydantic AI).

**Schémas créés** :
- Pipeline RAG : Document, Chunk, EmbeddingResult, SearchResult.
- Base SQL : TeamSchema, PlayerSchema.
- Configuration : PipelineSettings (Pydantic Settings).

---

## Architecture du système

### Vue d'ensemble

Le système hybride combine deux pipelines complémentaires orchestrés par un router LLM :

```
                         Question utilisateur
                                 │
                                 ▼
                         ┌───────────────┐
                         │  Router LLM   │
                         │ (Mistral API) │
                         └───────┬───────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
         ┌───────────────┐               ┌───────────────┐
         │   Pipeline    │               │   Pipeline    │
         │      SQL      │               │      RAG      │
         └───────┬───────┘               └───────┬───────┘
                 │                               │
                 ▼                               ▼
         ┌───────────────┐               ┌───────────────┐
         │  sql_tool.py  │               │ vector_store  │
         │ (LangChain)   │               │    (FAISS)    │
         └───────┬───────┘               └───────┬───────┘
                 │                               │
                 ▼                               ▼
         ┌───────────────┐               ┌───────────────┐
         │  SQLite DB    │               │  Index FAISS  │
         │ (568 joueurs) │               │ (PDF Reddit)  │
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

---
### Router LLM

Le router LLM classifie automatiquement chaque question vers le bon pipeline (SQL ou RAG).

**Fonctionnement** :
- Prompt spécialisé définissant les critères de classification (statistiques → SQL, analyses → RAG).
- Appel au LLM Mistral avec température=0 (déterministe).
- Réponse attendue : "SQL" ou "RAG" (un seul mot).
- Fallback : RAG en cas d'erreur du router.

**Critères de classification** :

| Type | Critères | Exemples |
|------|----------|----------|
| SQL | Statistiques, chiffres, classements, comparaisons chiffrées | "Meilleur marqueur", "Stats de Jokic", "Top 5 rebondeurs" |
| RAG | Analyses, histoire, opinions, événements | "Pourquoi les Lakers ont perdu ?", "Rumeurs de transfert" |

---
### Pipeline RAG (documents textuels)

**Entrée** : Question ouverte nécessitant une analyse ou une opinion.<br>
**Sortie** : Réponse contextuelle basée sur les archives textuelles.

**Étapes** :
1. **Embedding de la question** : Génération d'un vecteur via l'API Mistral Embeddings (modèle mistral-embed).
2. **Recherche FAISS** : Récupération des k=5 chunks les plus similaires (similarité cosinus).
3. **Construction du prompt** : Injection des contextes dans un template avec la question.
4. **Génération LLM** : Appel à l'API Mistral Chat pour synthétiser la réponse.
5. **Validation Pydantic** : Vérification des schémas SearchResult et cohérence des métadonnées.

**Données sources** : 4 fichiers PDF (archives Reddit) → 1500 caractères par chunk, chevauchement 150.

---
### Pipeline SQL (données structurées)

**Entrée** : Question chiffrée nécessitant des statistiques précises.<br>
**Sortie** : Réponse factuelle basée sur la base de données.

**Étapes** :
1. **Génération SQL** : LLM Mistral génère une requête SQL à partir de la question + exemples few-shot.
2. **Exécution** : Requête exécutée sur la base SQLite via SQLAlchemy.
3. **Synthèse** : LLM Mistral reformule les résultats bruts en réponse en langage naturel.
4. **Validation Pydantic** : Vérification des schémas PlayerSchema et TeamSchema.

**Données sources** : Fichier Excel (569 joueurs, 45 colonnes) → validation Pydantic → SQLite (568 joueurs, 1 écarté).

**Exemples few-shot** : 12 exemples de paires (question, requête SQL) pour guider la génération.

---
## Résultats de l'évaluation

### Méthodologie d'évaluation

**Jeu de test** : 30 questions réparties en 3 catégories.

| Catégorie | Nombre | Description | Exemple |
|-----------|--------|-------------|---------|
| `simple` | 10 | Questions factuelles avec ground truth | "Quel joueur a marqué le plus de points ?" |
| `complex` | 10 | Questions d'analyse multi-critères | "Compare les performances de Jokic et Giannis" |
| `noisy` | 10 | Questions mal formulées (langage SMS, fautes) | "mec c koi le joueur ki score le + de 3pts" |

**Métriques RAGAS** :
- `faithfulness` : fidélité de la réponse aux contextes (0 = hallucinations, 1 = fidèle).
- `answer_relevancy` : pertinence de la réponse (0 = hors-sujet, 1 = pertinent).
- `context_precision` : pertinence des contextes récupérés (nécessite ground truth).
- `context_recall` : couverture de la ground truth par les contextes (nécessite ground truth).

---
### Résultats avant enrichissement (RAG seul)

| Catégorie | faithfulness | answer_relevancy | context_precision | context_recall |
|-----------|-------------|-----------------|------------------|---------------|
| simple | 0.68 | 0.41 | 0.30 | 0.19 |
| complex | 0.67 | 0.79 | -- | -- |
| noisy | 0.15 | 0.62 | -- | -- |
| **global** | **0.46** | **0.61** | **0.30** | **0.19** |

**Observations** :
- **Questions simples** : Context_recall très faible (0.19) → le retriever FAISS ne récupère pas les bons documents pour les questions factuelles nécessitant des chiffres précis.
- **Questions complexes** : Meilleure performance (0.79 answer_relevancy) car ces questions portent sur l'analyse textuelle, domaine adapté au RAG.
- **Questions bruitées** : Faithfulness catastrophique (0.15) → le système est très sensible au bruit lexical (fautes d'orthographe, langage SMS).

---
### Résultats après enrichissement (RAG + SQL)

**Répartition des pipelines** : 22 questions SQL (73%), 8 questions RAG (27%).

| Catégorie | faithfulness | answer_relevancy | context_precision | context_recall |
|-----------|-------------|-----------------|------------------|---------------|
| simple | 0.80 | 0.89 | 0.70 | 0.60 |
| complex | 0.81 | 0.72 | -- | -- |
| noisy | 0.64 | 0.73 | -- | -- |
| **global** | **0.75** | **0.78** | **0.70** | **0.60** |

---
### Tableau comparatif

| Métrique | Avant (RAG seul) | Après (RAG + SQL) | Amélioration |
|----------|------------------|-------------------|--------------|
| faithfulness | 0.46 | 0.75 | **+63 %** |
| answer_relevancy | 0.61 | 0.78 | **+28 %** |
| context_precision | 0.30 | 0.70 | **+133 %** |
| context_recall | 0.19 | 0.60 | **+216 %** |

---
### Interprétation et limites des scores

#### Échantillon réduit pour certaines métriques

Les métriques `context_precision` et `context_recall` **ne portent que sur 10 questions** (catégorie "simple"), car elles nécessitent un ground truth manuel qui n'a été produit que pour cette sous-catégorie.<br>
Ces deux métriques doivent être interprétées comme des **indicateurs directionnels** (tendance à l'amélioration) plutôt que comme des mesures absolues précises. Pour une évaluation statistiquement robuste, il faudrait un jeu de test de 100+ questions avec ground truth.

#### Architecture hybride et limites de RAGAS

L'architecture hybride de ce projet (RAG + SQL avec router) introduit une complexité que le framework RAGAS ne capture pas complètement :

- Pour les questions traitées par le **pipeline SQL** (73% du jeu de test), la notion de "contexte récupéré" diffère du RAG textuel classique. Le "contexte" est le résultat d'une requête SQL, pas un chunk de document.
- La métrique `faithfulness` mesure la fidélité aux contextes, mais avec des données SQL structurées, cette fidélité est quasiment garantie par construction (pas d'hallucination possible).
- Les métriques `context_precision` et `context_recall` évaluent la qualité du retriever, mais le "retriever" SQL fonctionne différemment d'un retriever vectoriel (recherche exacte vs recherche approximative).

#### Progression entre les deux versions du système

Les résultats démontrent clairement que l'architecture hybride surpasse le RAG seul :
- **Faithfulness +63 %** : Les réponses du système enrichi sont nettement plus fidèles aux données sources.
- **Context_recall +216 %** : Le système enrichi récupère des informations beaucoup plus pertinentes pour répondre aux questions factuelles.
- **Amélioration cohérente** : Les 4 métriques progressent dans le même sens, validant l'hypothèse que le pipeline SQL comble les faiblesses du RAG seul.

---
### Analyse des résultats par catégorie

#### Questions simples (factuelles)

On a une amélioration majeure sur toutes les métriques :
- Context_recall : 0.19 → 0.60 (+216%).
- Answer_relevancy : 0.41 → 0.89 (+117%).

Le pipeline SQL permet d'interroger directement la base de données structurée, éliminant le problème de récupération de contexte inadapté du RAG.

_Exemple_ :
- Question : "Quel joueur a marqué le plus de points ?"
- Avant : Recherche FAISS → documents Reddit hors-sujet → réponse approximative
- Après : Requête SQL `SELECT name, points_per_game FROM players ORDER BY points_per_game DESC LIMIT 1` **→** réponse précise.

#### Questions complexes (analyses)

On observe une légère baisse de answer_relevancy (0.79 → 0.72) mais meilleure faithfulness (0.67 → 0.81).<br>
Le router dirige certaines questions complexes vers SQL alors qu'elles contiendraient aussi des éléments d'analyse textuelle. Cependant, la fidélité aux données s'améliore car les chiffres cités sont exacts.

_Exemple_ :
- Question : "Compare les performances offensives et défensives de Jokic et Giannis".
- Routage : SQL (car contient "performances", "compare" + noms de joueurs).
- Résultat : Statistiques précises mais analyse contextuelle moins riche que le RAG.

#### Questions bruitées (robustesse)

On voit une importante amélioration de faithfulness (0.15 → 0.64).<br>
Le LLM utilisé pour la génération SQL est capable d'extraire la sémantique de la question malgré les fautes, puis génère une requête SQL correcte.

_Exemple_ :
- Question : "mec c koi le joueur ki score le + de 3pts".
- Avant : Recherche FAISS sensible au bruit lexical → documents non pertinents → réponse hallucinée.
- Après : LLM normalise la question → SQL `SELECT name, three_pointers_made FROM players ORDER BY three_pointers_made DESC LIMIT 1` → réponse correcte.

---

## Points de vigilance

### Biais du router

Le router a dirigé 73% des questions (22/30) vers le pipeline SQL et 27% (8/30) vers le pipeline RAG.

**Distribution par catégorie** :
- **Questions simples** : 100% SQL (10/10) - classification cohérente pour des questions factuelles
- **Questions complexes** : 70% SQL (7/10), 30% RAG (3/10) - les 3 questions mentionnant explicitement "discussions Reddit" ont été correctement dirigées vers RAG
- **Questions bruitées** : 50% SQL (5/10), 50% RAG (5/10) - répartition équilibrée

Le router utilise un prompt avec critères de classification et exemples few-shot. L'efficacité du router sur des questions hybrides ou à double intention n'a pas été mesurée.

---
### Limites méthodologiques de l'évaluation

#### 1. Taille du jeu de test

Le nombre de questions peut être augmenté (30 questions dont 10 avec ground truth).<br>
Du fait du faible échantillon, la variance statistique est élevée ; par conséquent, les résultats doivent être interprétés avec précaution.

_Axe d'amélioration_ : Étendre à 100+ questions avec ground truth pour toutes les catégories.

#### 2. Métriques RAGAS

Les métriques `context_precision` et `context_recall` nécessitent un ground truth manuel (disponible uniquement pour les questions simples).<br>
Il est impossible de mesurer la qualité de la récupération de contexte pour 67% des questions.

_Axe d'amélioration_ : Générer des ground truth synthétiques ou utiliser des métriques alternatives (`BERTScore`, `ROUGE`).

#### 3. Subjectivité du ground truth

Les réponses de référence ont été rédigées manuellement par une seule personne, introduisant un biais potentiel.<br> 
Une réponse différente mais correcte peut être pénalisée par les métriques RAGAS (notamment context_recall et context_precision).

_Axe d'amélioration_ : Validation croisée des ground truth par plusieurs experts du domaine.

---

## Recommandations

### Améliorations à court terme

#### 1. Implémenter un fallback RAG

Pour éviter les réponses vides quand SQL échoue, nous pourrions :
- Modifier le gestionnaire d'erreur actuel du pipeline SQL pour rediriger automatiquement vers le pipeline RAG.
- Logger les cas de fallback pour analyser les types de questions problématiques.
- Afficher à l'utilisateur que la réponse provient du fallback RAG.

#### 2. Affiner le prompt du router

En vue de gérer les questions hybrides (statistiques + analyse contextuelle), actuellement non testées, nous pourrions :
- Ajouter des exemples de questions ambiguës dans le prompt du router (ex : « Pourquoi [joueur] est-il considéré comme le meilleur ? »).
- Définir des critères plus précis (ex : si la question commence par « Pourquoi » ou « Comment », privilégier RAG ou utiliser les deux pipelines).
- Constituer un jeu de validation de 50 questions hybrides annotées manuellement pour évaluer le router.

---
### Améliorations à moyen terme

#### 1. Implémenter un cache pour les requêtes SQL fréquentes

Réduire la latence et le coût API pour les questions récurrentes en mettant en place un cache SQL :
- Hacher les questions SQL normalisées (ex : lowercase, suppression des articles).
- Stocker les résultats SQL en mémoire avec une durée de vie de 1h.
- Retourner directement la réponse mise en cache si la question a déjà été posée.

#### 2. Implémenter un système de feedback utilisateur

Mettre en place un système de collecte de retours utilisateurs pour améliorer le système :
- Ajouter des boutons « Réponse utile / Pas utile » dans l'interface Streamlit.
- Enregistrer les évaluations avec la question, la réponse et le pipeline utilisé.
- Analyser mensuellement les retours négatifs pour identifier les patterns d'échec récurrents.

---
### Améliorations à long terme

#### 1. Transition vers un système de routage hybride (LLM + règles)

Réduire les appels au router LLM (économie de coût et latence) en combinant règles heuristiques et classification intelligente :
- Identifier les motifs de questions récurrents (ex : « Quel joueur... » → SQL 95% du temps).
- Implémenter des règles simples de pré-classification pour les cas évidents (mots-clés : « statistiques », « combien », « quel joueur »).
- Conserver le router LLM uniquement pour les questions ambiguës.

#### 2. Migration vers une base de données PostgreSQL

Supporter un volume de données plus important et des requêtes concurrentes :
- Migrer de SQLite vers PostgreSQL pour bénéficier de meilleures performances.
- Ajouter des index sur les colonnes fréquemment requêtées (`name`, `team_code`, `points_per_game`, `assists`).
- Configurer un pool de connexions pour gérer les requêtes simultanées.

---

## Conclusion

### Synthèse des résultats

Ce projet a permis de démontrer qu'un système RAG classique, bien qu'efficace pour l'analyse de documents textuels, présente des limites majeures pour le traitement de questions nécessitant des données structurées et chiffrées. L'intégration d'un pipeline SQL orchestré par un router LLM a produit des améliorations significatives sur l'ensemble des métriques d'évaluation :

- **Faithfulness** : +63% (de 0.46 à 0.75).
- **Answer relevancy** : +28% (de 0.61 à 0.78).
- **Context precision** : +133% (de 0.30 à 0.70).
- **Context recall** : +216% (de 0.19 à 0.60).

**Interprétation des résultats** :<br> L'amélioration relative avant/après constitue une preuve robuste de l'apport du pipeline SQL. La progression cohérente sur les 4 métriques valide l'hypothèse que l'architecture hybride RAG + SQL est l'approche adaptée pour un assistant intelligent devant traiter à la fois des questions d'analyse (nécessitant du contexte textuel) et des questions statistiques (nécessitant des calculs précis).

---
### Validation des objectifs initiaux

| Objectif | Statut | Résultats |
|----------|--------|-----------|
| Évaluer objectivement le système RAG initial | **Atteint** | Baseline établie avec 4 métriques RAGAS sur 30 questions |
| Identifier les axes d'amélioration prioritaires | **Atteint** | Faiblesse majeure identifiée : traitement des questions factuelles |
| Mettre en œuvre les solutions adaptées aux axes identifiés | **Atteint** | Architecture hybride RAG + SQL avec router LLM implémentée |
| Mesurer l'impact des améliorations | **Atteint** | Gains mesurés : +63% faithfulness, +216% context_recall |
| Documenter les biais et limites | **Atteint** | 1 biais identifié (router), 3 limites méthodologiques documentées |

---
### Perspectives d'évolution

Le système actuel constitue une base solide pour un déploiement en environnement de production, sous réserve de la mise en œuvre des recommandations à court terme (fallback RAG en cas d'erreur SQL, affinage du prompt du router). Les améliorations à moyen et long terme permettront de faire évoluer le système vers une solution robuste, scalable et intégrable dans l'écosystème applicatif de SportSee.

**Priorités court terme** (pré-production) :
- Implémenter un fallback automatique vers RAG en cas d'échec SQL
- Affiner le prompt du router pour gérer les questions hybrides

**Priorités moyen terme** (optimisation) :
- Mettre en place un cache SQL pour réduire latence et coûts API
- Collecter les retours utilisateurs pour amélioration continue

**Priorités long terme** (scalabilité) :
- Optimiser le routage avec un système hybride (règles + LLM)
- Migrer vers PostgreSQL pour supporter un volume de données accru

---
### Enseignements méthodologiques

Ce projet illustre l'importance d'une approche structurée en data science appliquée aux LLM :
1. **Baseline objective** : L'évaluation quantitative avec RAGAS a permis de mesurer précisément les gains.
2. **Validation Pydantic** : La sécurisation des flux de données a évité de nombreux bugs silencieux (exemple : détection du joueur avec statistiques > 100%).
3. **Traçabilité Logfire** : Le monitoring en temps réel a facilité le debugging et l'optimisation du pipeline.
4. **Tests de robustesse** : Les questions bruitées ont révélé une force inattendue du pipeline SQL (robustesse au bruit lexical via normalisation LLM).
5. **Évaluation contextuelle** : L'amélioration relative (delta avant/après) constitue un indicateur plus fiable que les scores absolus pour valider l'apport d'une évolution d'architecture.

**Leçon principale** : Un système RAG performant n'est pas un système RAG seul, mais une orchestration intelligente de plusieurs paradigmes complémentaires (recherche sémantique, bases de données, génération de code, etc.) adaptés aux spécificités de chaque type de question.

---

**Dernière mise à jour** : Février 2026<br>
**Projet** : eval-rag-nba<br>
**Auteur** : Mounir Meknaci<br>
**Formation** : Expert en ingénierie et science des données
