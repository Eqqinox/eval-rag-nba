"""
Modèles SQLAlchemy et schémas Pydantic pour la base de données NBA.
Définit les tables teams et players avec leurs validations.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from utils.config import DATABASE_URL

# --- Configuration SQLAlchemy ---

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


# --- Modèles SQLAlchemy (tables SQL) ---


class TeamDB(Base):
    """Table des équipes NBA."""

    __tablename__ = "teams"

    code = Column(String(3), primary_key=True)
    name = Column(String(100), nullable=False)

    # Relation avec les joueurs
    players = relationship("PlayerDB", back_populates="team")


class PlayerDB(Base):
    """Table des joueurs NBA avec leurs statistiques."""

    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    team_code = Column(String(3), ForeignKey("teams.code"), nullable=False)
    age = Column(Integer)

    # Statistiques de base
    games_played = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    minutes_per_game = Column(Float)
    points_per_game = Column(Float)

    # Tirs
    field_goals_made = Column(Float)
    field_goals_attempted = Column(Float)
    field_goal_pct = Column(Float)
    three_pointers_made = Column(Float)
    three_pointers_attempted = Column(Float)
    three_point_pct = Column(Float)
    free_throws_made = Column(Float)
    free_throws_attempted = Column(Float)
    free_throw_pct = Column(Float)

    # Rebonds
    offensive_rebounds = Column(Float)
    defensive_rebounds = Column(Float)
    total_rebounds = Column(Float)

    # Autres statistiques
    assists = Column(Float)
    turnovers = Column(Float)
    steals = Column(Float)
    blocks = Column(Float)
    personal_fouls = Column(Float)
    fantasy_points = Column(Float)
    double_doubles = Column(Integer)
    triple_doubles = Column(Integer)
    plus_minus = Column(Float)

    # Statistiques avancées
    offensive_rating = Column(Float)
    defensive_rating = Column(Float)
    net_rating = Column(Float)
    assist_pct = Column(Float)
    assist_to_turnover = Column(Float)
    assist_ratio = Column(Float)
    offensive_rebound_pct = Column(Float)
    defensive_rebound_pct = Column(Float)
    total_rebound_pct = Column(Float)
    turnover_ratio = Column(Float)
    effective_fg_pct = Column(Float)
    true_shooting_pct = Column(Float)
    usage_rate = Column(Float)
    pace = Column(Float)
    player_impact_estimate = Column(Float)
    possessions = Column(Float)

    # Relation avec l'équipe
    team = relationship("TeamDB", back_populates="players")


# --- Schémas Pydantic (validation des données) ---


class TeamSchema(BaseModel):
    """Schéma de validation pour une équipe."""

    code: str = Field(..., min_length=2, max_length=3, description="Code équipe (3 lettres)")
    name: str = Field(..., min_length=1, max_length=100, description="Nom complet de l'équipe")

    @field_validator("code")
    @classmethod
    def code_uppercase(cls, v: str) -> str:
        return v.upper()


class PlayerSchema(BaseModel):
    """Schéma de validation pour un joueur avec ses statistiques."""

    name: str = Field(..., min_length=1, max_length=100, description="Nom du joueur")
    team_code: str = Field(..., min_length=2, max_length=3, description="Code équipe")
    age: Optional[int] = Field(None, ge=18, le=50, description="Âge du joueur")

    # Statistiques de base
    games_played: Optional[int] = Field(None, ge=0, description="Matchs joués")
    wins: Optional[int] = Field(None, ge=0, description="Victoires")
    losses: Optional[int] = Field(None, ge=0, description="Défaites")
    minutes_per_game: Optional[float] = Field(None, ge=0, description="Minutes par match")
    points_per_game: Optional[float] = Field(None, ge=0, description="Points par match")

    # Tirs
    field_goals_made: Optional[float] = Field(None, ge=0, description="Tirs réussis")
    field_goals_attempted: Optional[float] = Field(None, ge=0, description="Tirs tentés")
    field_goal_pct: Optional[float] = Field(None, ge=0, le=100, description="% tirs réussis")
    three_pointers_made: Optional[float] = Field(None, ge=0, description="3 points réussis")
    three_pointers_attempted: Optional[float] = Field(None, ge=0, description="3 points tentés")
    three_point_pct: Optional[float] = Field(None, ge=0, le=100, description="% 3 points")
    free_throws_made: Optional[float] = Field(None, ge=0, description="LF réussis")
    free_throws_attempted: Optional[float] = Field(None, ge=0, description="LF tentés")
    free_throw_pct: Optional[float] = Field(None, ge=0, le=100, description="% LF")

    # Rebonds
    offensive_rebounds: Optional[float] = Field(None, ge=0, description="Rebonds offensifs")
    defensive_rebounds: Optional[float] = Field(None, ge=0, description="Rebonds défensifs")
    total_rebounds: Optional[float] = Field(None, ge=0, description="Rebonds totaux")

    # Autres statistiques
    assists: Optional[float] = Field(None, ge=0, description="Passes décisives")
    turnovers: Optional[float] = Field(None, ge=0, description="Balles perdues")
    steals: Optional[float] = Field(None, ge=0, description="Interceptions")
    blocks: Optional[float] = Field(None, ge=0, description="Contres")
    personal_fouls: Optional[float] = Field(None, ge=0, description="Fautes personnelles")
    fantasy_points: Optional[float] = Field(None, description="Fantasy Points")
    double_doubles: Optional[int] = Field(None, ge=0, description="Double-doubles")
    triple_doubles: Optional[int] = Field(None, ge=0, description="Triple-doubles")
    plus_minus: Optional[float] = Field(None, description="Plus-Minus")

    # Statistiques avancées
    offensive_rating: Optional[float] = Field(None, description="Offensive Rating")
    defensive_rating: Optional[float] = Field(None, description="Defensive Rating")
    net_rating: Optional[float] = Field(None, description="Net Rating")
    assist_pct: Optional[float] = Field(None, ge=0, le=100, description="% assists")
    assist_to_turnover: Optional[float] = Field(None, ge=0, description="Ratio AST/TO")
    assist_ratio: Optional[float] = Field(None, ge=0, description="Assist Ratio")
    offensive_rebound_pct: Optional[float] = Field(None, ge=0, le=100, description="% OREB")
    defensive_rebound_pct: Optional[float] = Field(None, ge=0, le=100, description="% DREB")
    total_rebound_pct: Optional[float] = Field(None, ge=0, le=100, description="% REB")
    turnover_ratio: Optional[float] = Field(None, ge=0, description="Turnover Ratio")
    effective_fg_pct: Optional[float] = Field(None, ge=0, le=100, description="EFG%")
    true_shooting_pct: Optional[float] = Field(None, ge=0, le=100, description="TS%")
    usage_rate: Optional[float] = Field(None, ge=0, le=100, description="Usage Rate")
    pace: Optional[float] = Field(None, ge=0, description="Pace")
    player_impact_estimate: Optional[float] = Field(None, description="PIE")
    possessions: Optional[float] = Field(None, ge=0, description="Possessions")

    @field_validator("team_code")
    @classmethod
    def team_code_uppercase(cls, v: str) -> str:
        return v.upper()

    class Config:
        from_attributes = True


# --- Fonctions utilitaires ---


def create_tables():
    """Crée toutes les tables dans la base de données."""
    Base.metadata.create_all(engine)


def drop_tables():
    """Supprime toutes les tables de la base de données."""
    Base.metadata.drop_all(engine)


def get_session():
    """Retourne une nouvelle session de base de données."""
    return SessionLocal()
