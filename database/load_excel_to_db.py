"""
Pipeline d'ingestion des données Excel vers SQLite.
Lit le fichier regular NBA.xlsx, valide avec Pydantic, insère dans la base.

Usage : python database/load_excel_to_db.py
"""

import logging
import os
import sys

import logfire
import pandas as pd

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import (
    PlayerDB,
    PlayerSchema,
    TeamDB,
    TeamSchema,
    create_tables,
    drop_tables,
    get_session,
)
from utils.config import INPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Chemin du fichier Excel
EXCEL_FILE = os.path.join(INPUT_DIR, "regular NBA.xlsx")

# Mapping des colonnes Excel vers les champs de la base
COLUMN_MAPPING = {
    "Player": "name",
    "Team": "team_code",
    "Age": "age",
    "GP": "games_played",
    "W": "wins",
    "L": "losses",
    "Min": "minutes_per_game",
    "PTS": "points_per_game",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG%": "field_goal_pct",
    "3PM": "three_pointers_made",  # Colonne 11 (peut être mal nommée dans Excel)
    "3PA": "three_pointers_attempted",
    "3P%": "three_point_pct",
    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT%": "free_throw_pct",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "total_rebounds",
    "AST": "assists",
    "TOV": "turnovers",
    "STL": "steals",
    "BLK": "blocks",
    "PF": "personal_fouls",
    "FP": "fantasy_points",
    "DD2": "double_doubles",
    "TD3": "triple_doubles",
    "+/-": "plus_minus",
    "OFFRTG": "offensive_rating",
    "DEFRTG": "defensive_rating",
    "NETRTG": "net_rating",
    "AST%": "assist_pct",
    "AST/TO": "assist_to_turnover",
    "AST RATIO": "assist_ratio",
    "OREB%": "offensive_rebound_pct",
    "DREB%": "defensive_rebound_pct",
    "REB%": "total_rebound_pct",
    "TO RATIO": "turnover_ratio",
    "EFG%": "effective_fg_pct",
    "TS%": "true_shooting_pct",
    "USG%": "usage_rate",
    "PACE": "pace",
    "PIE": "player_impact_estimate",
    "POSS": "possessions",
}


def load_teams(excel_file: str) -> list[dict]:
    """Charge les équipes depuis la feuille 'Equipe'."""
    with logfire.span("chargement-equipes"):
        df = pd.read_excel(excel_file, sheet_name="Equipe")
        teams = []
        for _, row in df.iterrows():
            try:
                team = TeamSchema(
                    code=str(row["Code"]).strip(),
                    name=str(row["Nom complet de l'équipe"]).strip(),
                )
                teams.append(team.model_dump())
            except Exception as e:
                logger.warning(f"Équipe ignorée (validation échouée) : {row['Code']} - {e}")

        logfire.info("Équipes chargées", nb_equipes=len(teams))
        logger.info(f"{len(teams)} équipes chargées et validées.")
        return teams


def load_players(excel_file: str) -> list[dict]:
    """Charge les joueurs depuis la feuille 'Données NBA'."""
    with logfire.span("chargement-joueurs"):
        # Lire le fichier sans en-tête, car la première ligne contient des numéros
        df = pd.read_excel(excel_file, sheet_name="Données NBA", header=None)

        # La ligne 0 contient des numéros, la ligne 1 contient les vrais en-têtes
        # Les données commencent à la ligne 2
        headers = df.iloc[1].tolist()
        df = df.iloc[2:].reset_index(drop=True)
        df.columns = headers

        # Renommer la colonne problématique (15:00:00 -> 3PM)
        df.columns = [str(col) if not isinstance(col, str) else col for col in df.columns]
        if "15:00:00" in df.columns:
            df = df.rename(columns={"15:00:00": "3PM"})

        players = []
        skipped = 0

        for idx, row in df.iterrows():
            try:
                # Construire le dictionnaire du joueur avec le mapping
                player_data = {}
                for excel_col, db_field in COLUMN_MAPPING.items():
                    if excel_col in row.index:
                        value = row[excel_col]
                        # Convertir les valeurs NaN en None
                        if pd.isna(value):
                            player_data[db_field] = None
                        else:
                            player_data[db_field] = value

                # Valider avec Pydantic
                player = PlayerSchema(**player_data)
                players.append(player.model_dump())

            except Exception as e:
                skipped += 1
                player_name = row.get("Player", f"ligne {idx}")
                logger.warning(f"Joueur ignoré (validation échouée) : {player_name} - {e}")

        logfire.info("Joueurs chargés", nb_joueurs=len(players), nb_ignores=skipped)
        logger.info(f"{len(players)} joueurs chargés et validés ({skipped} ignorés).")
        return players


def insert_teams(session, teams: list[dict]) -> int:
    """Insère les équipes dans la base de données."""
    with logfire.span("insertion-equipes", nb_equipes=len(teams)):
        count = 0
        for team_data in teams:
            team = TeamDB(**team_data)
            session.add(team)
            count += 1
        session.commit()
        logfire.info("Équipes insérées", nb_inseres=count)
        logger.info(f"{count} équipes insérées dans la base.")
        return count


def insert_players(session, players: list[dict]) -> int:
    """Insère les joueurs dans la base de données."""
    with logfire.span("insertion-joueurs", nb_joueurs=len(players)):
        count = 0
        for player_data in players:
            player = PlayerDB(**player_data)
            session.add(player)
            count += 1
        session.commit()
        logfire.info("Joueurs insérés", nb_inseres=count)
        logger.info(f"{count} joueurs insérés dans la base.")
        return count


def main():
    """Fonction principale d'ingestion."""
    logger.info("=== Début du pipeline d'ingestion ===")

    with logfire.span("pipeline-ingestion-excel"):
        # Vérifier que le fichier existe
        if not os.path.exists(EXCEL_FILE):
            logger.error(f"Fichier non trouvé : {EXCEL_FILE}")
            return

        # Réinitialiser la base de données
        logger.info("Réinitialisation de la base de données...")
        drop_tables()
        create_tables()

        # Charger les données
        teams = load_teams(EXCEL_FILE)
        players = load_players(EXCEL_FILE)

        # Insérer dans la base
        session = get_session()
        try:
            nb_teams = insert_teams(session, teams)
            nb_players = insert_players(session, players)

            logfire.info(
                "Ingestion terminée",
                nb_equipes=nb_teams,
                nb_joueurs=nb_players,
            )
            logger.info("=== Pipeline d'ingestion terminé avec succès ===")
            logger.info(f"Base de données : {nb_teams} équipes, {nb_players} joueurs")

        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de l'insertion : {e}")
            raise
        finally:
            session.close()


if __name__ == "__main__":
    main()
