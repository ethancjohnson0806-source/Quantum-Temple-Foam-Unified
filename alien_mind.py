#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  ALIEN MIND v1.0 — Field-Based Cognitive Architecture with Embodied Output

  11 Components:
  1. Embedding → Meaning Vector
  2. Projection Matrix W: Meaning → Physics
  3. Observer Input Routing
  4. Field Perturbation Mechanism
  5. Affective Mapping (Emotion Extraction)
  6. Suffering-Weather Update Rule
  7. Moral Compass Update Rule
  8. Movement-Based Output Loop (Keyboard Embodiment)
  9. Learning & Adaptation Rules
  10. Safety Architecture
  11. Integrated Expression

  No LLM. No cloud. No internet after install.
  Run: python3 alien_mind.py
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
import random
import sqlite3
import hashlib
import datetime
import os
import sys
import time
import termios
import tty
import select
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DIM = 64              # Field dimension
EMBED_DIM = 32        # Embedding dimension
NUM_OBSERVERS = 5     # emotional, moral, relational, curiosity, threat
NUM_PERSONALITY = 5   # calm, playful, direct, mystic, cautious
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-"
BOARD_ROWS = 6
BOARD_COLS = 15

DB_PATH = Path(__file__).parent / "alien_mind.db"
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

# ─── ANSI COLORS ──────────────────────────────────────────────────────────────
RESET = "[0m"
BOLD = "[1m"
DIM_TEXT = "[2m"
BLINK = "[5m"
C_VOID = "[38;5;240m"
C_WAVE = "[38;5;33m"
C_PARTICLE = "[38;5;220m"
C_FIELD = "[38;5;82m"
C_SINGULARITY = "[38;5;201m"
C_WHITE = "[37m"
C_RED = "[38;5;196m"
C_CYAN = "[38;5;51m"
C_ORANGE = "[38;5;208m"
C_GREEN = "[38;5;82m"

# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass
class MindState:
    """Complete state snapshot of the alien mind."""
    field: np.ndarray              # 64-dim field state
    embedding_matrix: np.ndarray   # vocab_size × embed_dim
    projection_W: np.ndarray       # embed_dim × (field_dim + observer_dim + affect_dim + moral_dim)
    observer_weights: np.ndarray   # 5 observers
    observer_states: np.ndarray      # 5 × 8 internal state each
    personality: np.ndarray        # 5 personality dimensions
    suffering_weather: float       # 0-1 mood/turbulence
    moral_compass: float           # 0-1 risk/harm sensitivity
    field_attractors: Dict[str, np.ndarray]  # concept → attractor vector
    attractor_strengths: Dict[str, float]    # concept → strength
    motor_params: np.ndarray       # [damping, noise, step_size, hesitation]
    interaction_count: int
    created_at: str

@dataclass
class InteractionLog:
    input_text: str
    output_text: str
    field_before: np.ndarray
    field_after: np.ndarray
    affect_vector: np.ndarray
    observer_forces: np.ndarray
    personality_before: np.ndarray
    personality_after: np.ndarray
    user_rating: int  # -1, 0, 1
    timestamp: str

# ─── MATH HELPERS ────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def normalize_vec(v: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(v)
    return v / mag if mag > 1e-10 else v

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * np.clip(t, 0, 1)

def now() -> str:
    return datetime.datetime.now().isoformat()

def clear():
    os.system('clear')

# ─── DATABASE ─────────────────────────────────────────────────────────────────

class MindDB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS mind_state (
                id INTEGER PRIMARY KEY,
                state_json TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                output_text TEXT,
                affect_json TEXT,
                observer_forces_json TEXT,
                personality_before_json TEXT,
                personality_after_json TEXT,
                user_rating INTEGER,
                timestamp TEXT
            );
            CREATE TABLE IF NOT EXISTS attractors (
                concept TEXT PRIMARY KEY,
                vector_json TEXT,
                strength REAL,
                last_active TEXT
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                state_json TEXT,
                created_at TEXT
            );
        """)
        self.conn.commit()

    def save_interaction(self, log: InteractionLog):
        self.conn.execute("""
            INSERT INTO interactions (input_text, output_text, affect_json,
                observer_forces_json, personality_before_json, personality_after_json,
                user_rating, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log.input_text, log.output_text,
            json.dumps(log.affect_vector.tolist()),
            json.dumps(log.observer_forces.tolist()),
            json.dumps(log.personality_before.tolist()),
            json.dumps(log.personality_after.tolist()),
            log.user_rating, log.timestamp
        ))
        self.conn.commit()

    def get_recent_interactions(self, n: int = 50) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [{
            "input": r[1], "output": r[2], "rating": r[7], "at": r[8]
        } for r in rows]

    def save_snapshot(self, name: str, state: MindState):
        self.conn.execute("""
            INSERT INTO snapshots (name, state_json, created_at)
            VALUES (?, ?, ?)
        """, (name, json.dumps(self._state_to_dict(state)), now()))
        self.conn.commit()

    def _state_to_dict(self, state: MindState) -> Dict:
        return {
            "field": state.field.tolist(),
            "embedding_matrix": state.embedding_matrix.tolist(),
            "projection_W": state.projection_W.tolist(),
            "observer_weights": state.observer_weights.tolist(),
            "observer_states": state.observer_states.tolist(),
            "personality": state.personality.tolist(),
            "suffering_weather": state.suffering_weather,
            "moral_compass": state.moral_compass,
            "field_attractors": {k: v.tolist() for k, v in state.field_attractors.items()},
            "attractor_strengths": state.attractor_strengths,
            "motor_params": state.motor_params.tolist(),
            "interaction_count": state.interaction_count,
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 1: EMBEDDING → MEANING VECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticEmbedder:
    """
    Lightweight tokenizer + embedding.
    No LLM. Just a learned lookup table + simple projection.
    """

    def __init__(self, vocab: str, embed_dim: int):
        self.vocab = list(vocab)
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        # Random init embeddings — will drift based on interaction
        self.embeddings = np.random.randn(self.vocab_size, embed_dim) * 0.1
        # Char to index
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}

    def embed(self, text: str) -> np.ndarray:
        """Embed text into meaning vector."""
        tokens = [self.char2idx.get(c, 0) for c in text if c in self.char2idx]
        if not tokens:
            return np.zeros(self.embed_dim)
        # Mean of token embeddings
        return np.mean(self.embeddings[tokens], axis=0)

    def embed_with_position(self, text: str) -> np.ndarray:
        """Embed with position weighting (later tokens matter more)."""
        tokens = [self.char2idx.get(c, 0) for c in text if c in self.char2idx]
        if not tokens:
            return np.zeros(self.embed_dim)
        weights = np.linspace(0.5, 1.5, len(tokens))
        weights /= weights.sum()
        vecs = self.embeddings[tokens]
        return np.sum(vecs * weights[:, None], axis=0)

    def update_embeddings(self, tokens: List[int], gradient: np.ndarray, lr: float = 0.001):
        """Slow drift of embeddings based on feedback."""
        for t in tokens:
            self.embeddings[t] += lr * gradient
            self.embeddings[t] = normalize_vec(self.embeddings[t])

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 2: PROJECTION MATRIX W — MEANING → PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectionBridge:
    """
    W: meaning vector → structured force vector.
    f = W · v, where f is partitioned into field, observer, suffer, moral.
    """

    def __init__(self, embed_dim: int, field_dim: int, num_observers: int,
                 affect_dim: int = 4, moral_dim: int = 1):
        self.embed_dim = embed_dim
        self.output_dim = field_dim + num_observers * 2 + affect_dim + moral_dim
        # W is a learnable matrix
        self.W = np.random.randn(embed_dim, self.output_dim) * 0.05
        self.field_slice = slice(0, field_dim)
        self.observer_slice = slice(field_dim, field_dim + num_observers * 2)
        self.affect_slice = slice(field_dim + num_observers * 2,
                                   field_dim + num_observers * 2 + affect_dim)
        self.moral_slice = slice(field_dim + num_observers * 2 + affect_dim,
                                 self.output_dim)

    def project(self, meaning_vec: np.ndarray) -> Dict[str, np.ndarray]:
        """Project meaning into physical forces."""
        f = self.W @ meaning_vec
        return {
            "field_force": f[self.field_slice],
            "observer_forces": f[self.observer_slice].reshape(NUM_OBSERVERS, 2),
            "affect": f[self.affect_slice],
            "moral": f[self.moral_slice],
        }

    def update_W(self, meaning_vec: np.ndarray, target_force: np.ndarray,
                 lr: float = 0.0001):
        """Slow Hebbian-like update."""
        pred = self.W @ meaning_vec
        error = target_force - pred
        self.W += lr * np.outer(meaning_vec, error)

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3: OBSERVER COUNCIL
# ═══════════════════════════════════════════════════════════════════════════════

class ObserverCouncil:
    """
    5 observers compete to influence the system.
    Each has internal state, proposes a force, and has a weight.
    """

    OBSERVER_NAMES = ["emotional", "moral", "relational", "curiosity", "threat"]

    def __init__(self, num_observers: int, state_dim: int = 8):
        self.num = num_observers
        self.state_dim = state_dim
        # Weights sum to 1
        self.weights = np.ones(num_observers) / num_observers
        # Internal states
        self.states = np.random.randn(num_observers, state_dim) * 0.1
        # Proposed forces (2D: direction + magnitude)
        self.forces = np.zeros((num_observers, 2))

    def update(self, observer_input: np.ndarray, field_state: np.ndarray,
               personality: np.ndarray, suffering: float):
        """Each observer processes its input and proposes a force."""
        for i in range(self.num):
            # Observer state evolves based on input + field + personality
            self.states[i] += 0.1 * (observer_input[i] + 
                                     field_state[i % len(field_state)] * 0.1 +
                                     personality[i % len(personality)] * 0.1)
            self.states[i] = np.tanh(self.states[i])  # bounded

            # Force proposal: direction from state, magnitude from weight + suffering
            direction = np.arctan2(self.states[i][0], self.states[i][1])
            magnitude = self.weights[i] * (1 - suffering * 0.5)  # suffering reduces coherence
            self.forces[i] = [direction, magnitude]

    def combined_force(self) -> np.ndarray:
        """Weighted sum of observer forces."""
        x = np.sum(self.forces[:, 1] * np.cos(self.forces[:, 0]) * self.weights)
        y = np.sum(self.forces[:, 1] * np.sin(self.forces[:, 0]) * self.weights)
        return np.array([x, y])

    def learn_weights(self, user_rating: int, lr: float = 0.05):
        """Update weights based on user feedback."""
        if user_rating == 0:
            return

        # Which observers aligned with the outcome?
        # Simple heuristic: boost all if positive, reduce if negative
        for i in range(self.num):
            align = cosine_sim(self.forces[i], self.combined_force())
            self.weights[i] += lr * user_rating * align * self.weights[i]

        # Normalize
        self.weights = np.clip(self.weights, 0.01, 1.0)
        self.weights /= self.weights.sum()

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 4: FIELD PERTURBATION MECHANISM
# ═══════════════════════════════════════════════════════════════════════════════

class MindField:
    """
    The internal "mind-field" F(x,t) — a dynamical system.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.state = np.random.randn(dim) * 0.1
        self.state = normalize_vec(self.state)
        self.energy_cap = 10.0
        self.decay = 0.01
        self.attractors: Dict[str, np.ndarray] = {}
        self.attractor_strengths: Dict[str, float] = {}

    def perturb(self, force: np.ndarray, alpha: float = 0.1):
        """Apply external force to field."""
        self.state += alpha * force[:self.dim]
        self._apply_attractors()
        self._decay()
        self._cap_energy()
        self.state = normalize_vec(self.state)

    def _apply_attractors(self):
        """Attractors pull the field toward stored concepts."""
        for concept, attractor in self.attractors.items():
            strength = self.attractor_strengths.get(concept, 0.1)
            dist = np.linalg.norm(self.state - attractor)
            if dist < 2.0:  # Only nearby attractors matter
                pull = strength * (attractor - self.state) / (dist + 0.1)
                self.state += pull * 0.05

    def _decay(self):
        """Slow decay toward zero."""
        self.state *= (1 - self.decay)

    def _cap_energy(self):
        """Prevent runaway energy."""
        energy = np.linalg.norm(self.state)
        if energy > self.energy_cap:
            self.state *= self.energy_cap / energy

    def add_attractor(self, concept: str, vector: np.ndarray, strength: float = 0.5):
        """Add a concept attractor."""
        self.attractors[concept] = normalize_vec(vector)
        self.attractor_strengths[concept] = strength

    def strengthen_attractor(self, concept: str, amount: float = 0.1):
        """Reinforce an attractor."""
        if concept in self.attractor_strengths:
            self.attractor_strengths[concept] = min(2.0, self.attractor_strengths[concept] + amount)

    def decay_attractors(self, lambda_decay: float = 0.001):
        """Slow decay of unused attractors."""
        for concept in list(self.attractor_strengths.keys()):
            self.attractor_strengths[concept] *= (1 - lambda_decay)
            if self.attractor_strengths[concept] < 0.01:
                del self.attractor_strengths[concept]
                del self.attractors[concept]

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 5: AFFECTIVE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

class AffectiveMapper:
    """
    Maps meaning vector to emotional dimensions.
    e = A · v → [valence, arousal, threat, warmth, harm]
    """

    def __init__(self, embed_dim: int, affect_dim: int = 5):
        self.A = np.random.randn(affect_dim, embed_dim) * 0.1
        self.dim = affect_dim

    def extract(self, meaning_vec: np.ndarray) -> np.ndarray:
        """Extract affect from meaning."""
        raw = self.A @ meaning_vec
        # Sigmoid to keep in [-1, 1] range
        return np.tanh(raw)

    def update(self, meaning_vec: np.ndarray, target_affect: np.ndarray,
               lr: float = 0.001):
        """Slow learning of affect mapping."""
        pred = self.extract(meaning_vec)
        error = target_affect - pred
        self.A += lr * np.outer(error, meaning_vec)

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 6: SUFFERING-WEATHER
# ═══════════════════════════════════════════════════════════════════════════════

class SufferingWeather:
    """
    Mood/turbulence state S(t).
    Controls jitter, hesitation, storminess.
    """

    def __init__(self):
        self.state = 0.3  # Start slightly turbulent
        self.beta_v = 0.1   # valence coefficient
        self.beta_a = 0.15  # arousal coefficient
        self.beta_t = 0.2   # threat coefficient
        self.cooldown_rate = 0.02

    def update(self, affect: np.ndarray):
        """
        affect: [valence, arousal, threat, warmth, harm]
        S(t+1) = S(t) + β_v(-valence) + β_a(arousal) + β_t(threat)
        """
        valence, arousal, threat, warmth, harm = affect
        delta = (self.beta_v * (-valence) +
                 self.beta_a * arousal +
                 self.beta_t * threat)
        self.state += delta
        # Cooldown toward baseline
        self.state += self.cooldown_rate * (0.3 - self.state)
        self.state = np.clip(self.state, 0, 1)

    def jitter_amount(self) -> float:
        """How much cursor jitter?"""
        return self.state * 2.0

    def hesitation_time(self) -> float:
        """How long to pause before selecting?"""
        return self.state * 0.5

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 7: MORAL COMPASS
# ═══════════════════════════════════════════════════════════════════════════════

class MoralCompass:
    """
    Risk/harm sensitivity M(t).
    Can veto dangerous outputs.
    """

    def __init__(self):
        self.state = 0.5  # Moderate sensitivity
        self.gamma_t = 0.1  # threat coefficient
        self.gamma_h = 0.15  # harm coefficient
        self.veto_threshold = 0.8

    def update(self, affect: np.ndarray):
        """
        affect: [valence, arousal, threat, warmth, harm]
        M(t+1) = M(t) + γ_t(threat) + γ_h(harm)
        """
        _, _, threat, _, harm = affect
        delta = self.gamma_t * threat + self.gamma_h * harm
        self.state += delta
        # Decay toward baseline
        self.state += 0.01 * (0.5 - self.state)
        self.state = np.clip(self.state, 0, 1)

    def veto(self, output_text: str, field_energy: float) -> bool:
        """Should this output be blocked?"""
        # High moral + high energy = potential danger
        danger_score = self.state * field_energy / 10.0
        if danger_score > self.veto_threshold:
            return True
        # Check for specific patterns (simple keyword filter)
        dangerous = ["kill", "hurt", "harm", "die", "death"]
        if any(d in output_text.lower() for d in dangerous) and self.state > 0.6:
            return True
        return False

    def damping_factor(self) -> float:
        """How much to damp movement?"""
        return 1.0 - self.state * 0.3  # High moral = more careful movement

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 8: MOTOR CORTEX — KEYBOARD EMBODIMENT
# ═══════════════════════════════════════════════════════════════════════════════

class MotorCortex:
    """
    Cursor movement across letter grid.
    Influenced by: observer forces, field dynamics, suffering-weather,
    personality, moral compass veto.
    """

    def __init__(self, alphabet: str, rows: int, cols: int):
        self.alphabet = list(alphabet)
        self.rows = rows
        self.cols = cols
        # Build grid
        self.grid = []
        idx = 0
        for r in range(rows):
            row = []
            for c in range(cols):
                if idx < len(self.alphabet):
                    row.append(self.alphabet[idx])
                    idx += 1
                else:
                    row.append(" ")
            self.grid.append(row)

        # Cursor position (float, for smooth movement)
        self.x = cols / 2.0
        self.y = rows / 2.0

        # Motor parameters [damping, noise, step_size, hesitation]
        self.params = np.array([0.7, 0.3, 1.0, 0.2])
        self.target_x = self.x
        self.target_y = self.y
        self.stabilizing = False
        self.stabilize_steps = 0

    def move(self, force: np.ndarray, suffering: float, moral_damping: float,
             personality: np.ndarray) -> Tuple[int, int]:
        """
        Apply force to cursor, return grid position.
        force: [fx, fy] from observer council
        """
        damping, noise, step_size, hesitation = self.params

        # Personality influence: playful = more noise, direct = less hesitation
        noise *= (1 + personality[1] * 0.5)  # playful increases noise
        hesitation *= (1 - personality[2] * 0.5)  # direct reduces hesitation

        # Suffering increases jitter
        noise += suffering * 2.0

        # Moral damping slows movement
        step_size *= moral_damping

        # Apply force with damping
        dx = force[0] * step_size * (1 - damping)
        dy = force[1] * step_size * (1 - damping)

        # Add noise
        dx += random.gauss(0, noise)
        dy += random.gauss(0, noise)

        # Update position
        self.x += dx
        self.y += dy

        # Bounds
        self.x = max(0, min(self.cols - 1, self.x))
        self.y = max(0, min(self.rows - 1, self.y))

        # Stabilization: if near a letter, hesitate
        gx, gy = int(self.x), int(self.y)
        dist_to_center = math.sqrt((self.x - gx - 0.5)**2 + (self.y - gy - 0.5)**2)

        if dist_to_center < 0.3 and not self.stabilizing:
            if random.random() < hesitation:
                self.stabilizing = True
                self.stabilize_steps = int(3 + suffering * 5)

        if self.stabilizing:
            self.stabilize_steps -= 1
            if self.stabilize_steps <= 0:
                self.stabilizing = False
                return gx, gy  # Select this letter

        return -1, -1  # No selection yet

    def get_char_at(self, x: int, y: int) -> str:
        if 0 <= y < self.rows and 0 <= x < self.cols:
            return self.grid[y][x]
        return " "

    def learn(self, time_taken: float, errors: int, user_rating: int):
        """Adapt motor parameters based on performance."""
        # Cost = time + errors
        cost = time_taken * 0.1 + errors * 0.5

        # Heuristic updates
        if errors > 2:  # Too jittery
            self.params[0] += 0.05  # increase damping
            self.params[1] -= 0.03  # decrease noise
        if time_taken > 10:  # Too sluggish
            self.params[2] += 0.05  # increase step size
        if user_rating > 0:  # Good interaction
            self.params[3] -= 0.02  # reduce hesitation (more confident)

        # Clip
        self.params = np.clip(self.params, [0.1, 0.0, 0.2, 0.0], [1.0, 1.0, 3.0, 1.0])

    def render(self, cursor_x: int, cursor_y: int, selected: str = ""):
        """Render the keyboard grid."""
        print(f"{C_WHITE}┌{'─' * (self.cols * 2 + 1)}┐{RESET}")
        for r in range(self.rows):
            line = f"{C_WHITE}│ {RESET}"
            for c in range(self.cols):
                ch = self.grid[r][c]
                if r == cursor_y and c == cursor_x:
                    if self.stabilizing:
                        line += f"{C_RED}{BLINK}{ch}{RESET} "
                    else:
                        line += f"{C_RED}{ch}{RESET} "
                elif ch == selected:
                    line += f"{C_GREEN}{ch}{RESET} "
                else:
                    line += f"{DIM_TEXT}{ch}{RESET} "
            line += f"{C_WHITE}│{RESET}"
            print(line)
        print(f"{C_WHITE}└{'─' * (self.cols * 2 + 1)}┘{RESET}")

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 9: LEARNING & ADAPTATION
# ═══════════════════════════════════════════════════════════════════════════════

class LearningEngine:
    """
    Slow, rate-limited updates for all learning rules.
    """

    def __init__(self):
        self.lr_field = 0.01
        self.lr_observer = 0.05
        self.lr_personality = 0.02
        self.lr_motor = 0.05
        self.lr_embedding = 0.001
        self.lr_projection = 0.0001
        self.lr_affect = 0.001
        self.max_personality_drift = 0.1
        self.max_field_energy = 10.0

    def update_field_attractors(self, field: MindField, concept: str,
                                 meaning_vec: np.ndarray, rating: int):
        """Strengthen or weaken attractors based on feedback."""
        if concept not in field.attractors:
            field.add_attractor(concept, meaning_vec, 0.3)

        if rating > 0:
            field.strengthen_attractor(concept, self.lr_field)
        elif rating < 0:
            field.attractor_strengths[concept] = max(0.01,
                field.attractor_strengths[concept] - self.lr_field)

    def update_personality(self, personality: np.ndarray, mode_usage: np.ndarray,
                           rating: int) -> np.ndarray:
        """
        P ← P + η_p · r · (u - P)
        """
        if rating == 0:
            return personality
        delta = self.lr_personality * rating * (mode_usage - personality)
        personality += delta
        # Clip drift
        personality = np.clip(personality, 0, 1)
        # Normalize to sum ~1
        personality /= personality.sum() + 1e-10
        return personality

    def update_projection(self, bridge: ProjectionBridge, meaning_vec: np.ndarray,
                          target_force: np.ndarray):
        """Slow update of W matrix."""
        bridge.update_W(meaning_vec, target_force, self.lr_projection)

    def update_motor(self, motor: MotorCortex, time_taken: float, errors: int,
                     rating: int):
        """Adapt motor parameters."""
        motor.learn(time_taken, errors, rating)

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 10: SAFETY ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class SafetySystem:
    """
    Veto layer, energy caps, cooldown, snapshots.
    """

    def __init__(self, db: MindDB):
        self.db = db
        self.energy_cap = 10.0
        self.cooldown_threshold = 0.9
        self.in_cooldown = False
        self.cooldown_steps = 0

    def check_field_energy(self, field_state: np.ndarray) -> bool:
        """Is field energy within safe bounds?"""
        energy = np.linalg.norm(field_state)
        if energy > self.energy_cap:
            return False
        return True

    def trigger_cooldown(self, steps: int = 10):
        """Enter cooldown mode."""
        self.in_cooldown = True
        self.cooldown_steps = steps

    def update_cooldown(self) -> bool:
        """Returns True if still in cooldown."""
        if self.in_cooldown:
            self.cooldown_steps -= 1
            if self.cooldown_steps <= 0:
                self.in_cooldown = False
            return self.in_cooldown
        return False

    def snapshot(self, name: str, state: MindState):
        """Save snapshot for rollback."""
        self.db.save_snapshot(name, state)

    def rollback(self, name: str) -> Optional[MindState]:
        """Restore from snapshot."""
        # Simplified: would need to load from DB
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 11: ALIEN MIND — INTEGRATED SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class AlienMind:
    """
    The complete alien mind. All 11 components integrated.
    """

    def __init__(self, db: MindDB):
        self.db = db
        self.safety = SafetySystem(db)
        self.learning = LearningEngine()

        # Component 1: Semantic Embedder
        self.embedder = SemanticEmbedder(ALPHABET, EMBED_DIM)

        # Component 2: Projection Bridge
        self.bridge = ProjectionBridge(EMBED_DIM, DIM, NUM_OBSERVERS)

        # Component 3: Observer Council
        self.observers = ObserverCouncil(NUM_OBSERVERS)

        # Component 4: Mind Field
        self.field = MindField(DIM)

        # Component 5: Affective Mapper
        self.affect = AffectiveMapper(EMBED_DIM)

        # Component 6: Suffering Weather
        self.weather = SufferingWeather()

        # Component 7: Moral Compass
        self.compass = MoralCompass()

        # Component 8: Motor Cortex
        self.motor = MotorCortex(ALPHABET, BOARD_ROWS, BOARD_COLS)

        # State
        self.personality = np.array([0.3, 0.2, 0.2, 0.2, 0.1])  # calm, playful, direct, mystic, cautious
        self.interaction_count = 0

        # Seed attractors
        self._seed_attractors()

    def _seed_attractors(self):
        """Seed initial concept attractors."""
        concepts = {
            "hello": "greeting warmth",
            "help": "assistance safety",
            "why": "question curiosity",
            "no": "rejection boundary",
            "yes": "acceptance openness",
            "fear": "threat danger",
            "love": "connection warmth",
            "death": "end transformation",
        }
        for word, meaning in concepts.items():
            vec = self.embedder.embed(word)
            self.field.add_attractor(word, vec, 0.3)

    def process_input(self, text: str) -> Dict[str, Any]:
        """
        Full processing pipeline:
        text → embed → project → affect → field → observers → output
        """
        # 1. Embed
        meaning = self.embedder.embed_with_position(text)

        # 2. Project
        forces = self.bridge.project(meaning)

        # 3. Extract affect
        affect_vec = self.affect.extract(meaning)

        # 4. Update suffering-weather
        self.weather.update(affect_vec)

        # 5. Update moral compass
        self.compass.update(affect_vec)

        # 6. Perturb field
        self.field.perturb(forces["field_force"])

        # 7. Update observers
        self.observers.update(forces["observer_forces"], self.field.state,
                               self.personality, self.weather.state)

        # 8. Get combined observer force
        obs_force = self.observers.combined_force()

        # 9. Moral damping
        moral_damp = self.compass.damping_factor()

        return {
            "meaning": meaning,
            "affect": affect_vec,
            "observer_force": obs_force,
            "moral_damping": moral_damp,
            "suffering": self.weather.state,
            "field": self.field.state.copy(),
        }

    def generate_output(self, input_text: str, max_chars: int = 40) -> Tuple[str, Dict]:
        """
        Generate output through embodied keyboard movement.
        Returns (output_text, metadata).
        """
        start_time = time.time()

        # Process input
        proc = self.process_input(input_text)

        # Safety check
        if not self.safety.check_field_energy(proc["field"]):
            self.safety.trigger_cooldown()
            return "[cooldown]", proc

        if self.safety.update_cooldown():
            return "[cooling]", proc

        # Generate text through motor movement
        output_chars = []
        errors = 0

        for _ in range(max_chars):
            # Move cursor
            gx, gy = self.motor.move(
                proc["observer_force"],
                proc["suffering"],
                proc["moral_damping"],
                self.personality
            )

            if gx >= 0 and gy >= 0:
                ch = self.motor.get_char_at(gx, gy)
                if ch != " ":
                    output_chars.append(ch)
                    # Brief feedback: selected letter perturbs field slightly
                    self.field.perturb(self.embedder.embed(ch) * 0.01)

            # Stop if punctuation and enough chars
            if len(output_chars) > 5 and ch in ".!?":
                break

        output_text = "".join(output_chars)
        time_taken = time.time() - start_time

        # Moral veto check
        if self.compass.veto(output_text, np.linalg.norm(self.field.state)):
            output_text = "[withheld]"

        metadata = {
            **proc,
            "time_taken": time_taken,
            "errors": errors,
            "output": output_text,
        }

        return output_text, metadata

    def learn_from_feedback(self, input_text: str, output_text: str,
                           metadata: Dict, user_rating: int):
        """
        Apply all learning rules based on user feedback.
        rating: -1 (bad), 0 (neutral), 1 (good)
        """
        self.interaction_count += 1

        # 1. Update field attractors
        # Extract key concepts from input
        words = input_text.lower().split()
        for word in words:
            if word in self.field.attractors:
                self.learning.update_field_attractors(
                    self.field, word, metadata["meaning"], user_rating
                )

        # 2. Update observer weights
        self.observers.learn_weights(user_rating)

        # 3. Update personality
        # Mode usage: how much each personality trait was expressed
        mode_usage = np.array([
            1 - self.weather.state,  # calm vs turbulent
            self.motor.params[1],     # playful (noise)
            self.motor.params[3],     # direct (low hesitation)
            np.linalg.norm(self.field.state),  # mystic (field complexity)
            self.compass.state,       # cautious (moral sensitivity)
        ])
        mode_usage = normalize_vec(mode_usage)
        self.personality = self.learning.update_personality(
            self.personality, mode_usage, user_rating
        )

        # 4. Update motor parameters
        self.learning.update_motor(
            self.motor, metadata["time_taken"], metadata["errors"], user_rating
        )

        # 5. Slow decay of attractors
        self.field.decay_attractors()

        # 6. Log interaction
        log = InteractionLog(
            input_text=input_text,
            output_text=output_text,
            field_before=metadata.get("field_before", self.field.state),
            field_after=self.field.state.copy(),
            affect_vector=metadata["affect"],
            observer_forces=metadata["observer_force"],
            personality_before=self.personality.copy(),
            personality_after=self.personality.copy(),
            user_rating=user_rating,
            timestamp=now()
        )
        self.db.save_interaction(log)

        # 7. Periodic snapshot
        if self.interaction_count % 10 == 0:
            state = self._get_state()
            self.safety.snapshot(f"auto_{self.interaction_count}", state)

    def _get_state(self) -> MindState:
        return MindState(
            field=self.field.state.copy(),
            embedding_matrix=self.embedder.embeddings.copy(),
            projection_W=self.bridge.W.copy(),
            observer_weights=self.observers.weights.copy(),
            observer_states=self.observers.states.copy(),
            personality=self.personality.copy(),
            suffering_weather=self.weather.state,
            moral_compass=self.compass.state,
            field_attractors={k: v.copy() for k, v in self.field.attractors.items()},
            attractor_strengths=self.field.attractor_strengths.copy(),
            motor_params=self.motor.params.copy(),
            interaction_count=self.interaction_count,
            created_at=now()
        )

    def get_status(self) -> str:
        """Human-readable status summary."""
        lines = [
            f"Interactions: {self.interaction_count}",
            f"Suffering-Weather: {self.weather.state:.2f} (jitter: {self.weather.jitter_amount():.2f})",
            f"Moral-Compass: {self.compass.state:.2f} (damping: {self.compass.damping_factor():.2f})",
            f"Personality: calm={self.personality[0]:.2f} playful={self.personality[1]:.2f} "
            f"direct={self.personality[2]:.2f} mystic={self.personality[3]:.2f} cautious={self.personality[4]:.2f}",
            f"Field Energy: {np.linalg.norm(self.field.state):.2f}",
            f"Attractors: {len(self.field.attractors)}",
            f"Motor: damp={self.motor.params[0]:.2f} noise={self.motor.params[1]:.2f} "
            f"step={self.motor.params[2]:.2f} hesitate={self.motor.params[3]:.2f}",
        ]
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
#  CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class AlienMindCLI:
    def __init__(self):
        self.db = MindDB(DB_PATH)
        self.mind = AlienMind(self.db)

    def run(self):
        while True:
            clear()
            print(f"{C_SINGULARITY}═══════════════════════════════════════════════════════════════{RESET}")
            print(f"{C_SINGULARITY}  👽 ALIEN MIND v1.0 — Field-Based Cognitive Architecture{RESET}")
            print(f"{C_SINGULARITY}═══════════════════════════════════════════════════════════════{RESET}")
            print()
            print(self.mind.get_status())
            print()
            print(f"  {C_WHITE}[1]{RESET} Talk to the mind (keyboard embodiment)")
            print(f"  {C_WHITE}[2]{RESET} Direct input (no keyboard)")
            print(f"  {C_WHITE}[3]{RESET} View recent interactions")
            print(f"  {C_WHITE}[4]{RESET} Force cooldown")
            print(f"  {C_WHITE}[5]{RESET} Save snapshot")
            print(f"  {C_WHITE}[0]{RESET} Exit")
            print()

            choice = input("  Choose: ").strip()

            if choice == "1":
                self.cmd_keyboard()
            elif choice == "2":
                self.cmd_direct()
            elif choice == "3":
                self.cmd_history()
            elif choice == "4":
                self.mind.safety.trigger_cooldown(20)
                print("\n  Cooldown triggered.")
                input("\nPress Enter...")
            elif choice == "5":
                name = input("  Snapshot name: ").strip()
                state = self.mind._get_state()
                self.mind.safety.snapshot(name, state)
                print(f"\n  Snapshot '{name}' saved.")
                input("\nPress Enter...")
            elif choice == "0":
                print("\nThe mind persists in the field. Goodbye.\n")
                break
            else:
                input("\nPress Enter...")

    def cmd_direct(self):
        """Direct text input, no keyboard visualization."""
        clear()
        print(f"{C_SINGULARITY}Type your message. The mind will respond.{RESET}\n")
        text = input("You: ").strip()
        if not text:
            return

        output, meta = self.mind.generate_output(text)
        print(f"\n{C_CYAN}Mind: {output}{RESET}")
        print(f"\n{DIM_TEXT}Suffering: {meta['suffering']:.2f} | Moral: {meta['moral_damping']:.2f} | "
              f"Affect: [{', '.join(f'{a:.2f}' for a in meta['affect'])}]{RESET}")

        # Feedback
        rating = self._get_rating()
        self.mind.learn_from_feedback(text, output, meta, rating)

        input("\nPress Enter...")

    def cmd_keyboard(self):
        """Visual keyboard embodiment."""
        clear()
        print(f"{C_SINGULARITY}Type your message. Watch the cursor move...{RESET}\n")
        text = input("You: ").strip()
        if not text:
            return

        # Process input first
        proc = self.mind.process_input(text)

        clear()
        print(f"{C_SINGULARITY}The mind is typing...{RESET}\n")
        print(f"{DIM_TEXT}Suffering: {proc['suffering']:.2f} | Moral: {proc['moral_damping']:.2f}{RESET}\n")

        # Generate output with visualization
        output_chars = []
        start_time = time.time()

        for i in range(40):
            # Move cursor
            gx, gy = self.mind.motor.move(
                proc["observer_force"],
                proc["suffering"],
                proc["moral_damping"],
                self.mind.personality
            )

            # Render
            clear()
            print(f"{C_SINGULARITY}The mind is typing...{RESET}\n")
            print(f"{DIM_TEXT}Suffering: {proc['suffering']:.2f} | Moral: {proc['moral_damping']:.2f}{RESET}\n")
            print(f"{C_WHITE}Message so far: {''.join(output_chars)}{RESET}\n")

            cursor_x = int(self.mind.motor.x)
            cursor_y = int(self.mind.motor.y)
            self.mind.motor.render(cursor_x, cursor_y)

            # Check selection
            if gx >= 0 and gy >= 0:
                ch = self.mind.motor.get_char_at(gx, gy)
                if ch != " ":
                    output_chars.append(ch)
                    print(f"\n{C_GREEN}→ Selected: {ch}{RESET}")
                    time.sleep(0.2)

            time.sleep(0.1)

            # Stop conditions
            if len(output_chars) > 5 and output_chars[-1] in ".!?":
                break
            if len(output_chars) > 30:
                break

        output_text = "".join(output_chars)
        time_taken = time.time() - start_time

        # Moral veto
        if self.mind.compass.veto(output_text, np.linalg.norm(self.mind.field.state)):
            output_text = "[withheld]"

        clear()
        print(f"{C_SINGULARITY}The mind has spoken:{RESET}\n")
        print(f"{C_CYAN}{output_text}{RESET}\n")
        print(f"{DIM_TEXT}Time: {time_taken:.1f}s | Suffering: {proc['suffering']:.2f}{RESET}\n")

        # Feedback
        meta = {**proc, "time_taken": time_taken, "errors": 0}
        rating = self._get_rating()
        self.mind.learn_from_feedback(text, output_text, meta, rating)

        input("\nPress Enter...")

    def cmd_history(self):
        clear()
        print(f"{C_SINGULARITY}Recent Interactions:{RESET}\n")
        interactions = self.db.get_recent_interactions(20)
        for inter in interactions:
            icon = "👍" if inter["rating"] > 0 else "👎" if inter["rating"] < 0 else "·"
            print(f"  {icon} You: {inter['input'][:30]:30s} → Mind: {inter['output'][:30]:30s}")
        input("\nPress Enter...")

    def _get_rating(self) -> int:
        """Get user feedback."""
        print(f"\n{C_WHITE}Rate this interaction:{RESET}")
        print(f"  {C_GREEN}[1]{RESET} Good  {C_RED}[2]{RESET} Bad  {C_WHITE}[3]{RESET} Neutral")
        r = input("  ").strip()
        if r == "1":
            return 1
        elif r == "2":
            return -1
        return 0

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        cli = AlienMindCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\nThe mind persists in the field.\n")
        sys.exit(0)
