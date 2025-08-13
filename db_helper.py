import sqlite3
import hashlib
import json
from typing import List, Dict, Any
from datetime import datetime

DB_FILE = "app_data.db"


def _get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_db():
    """Initialize database with proper tables"""
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()

        # Users table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        # Workouts table with foreign key
        cur.execute("""
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            date TEXT NOT NULL,
            body_part TEXT,
            exercise TEXT,
            equipment TEXT,
            health_condition TEXT,
            sets_json TEXT NOT NULL,
            target_reps_json TEXT NOT NULL,
            recommendation_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE
        )""")

        # Create indexes for performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_workouts_user ON workouts(user_email)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_workouts_date ON workouts(date)")

        conn.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def hash_password(password: str) -> str:
    """Secure password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(name: str, email: str, password: str) -> bool:
    """Register new user with error handling"""
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists
    except Exception as e:
        print(f"Registration error: {e}")
        return False
    finally:
        if conn:
            conn.close()


def verify_user(email: str, password: str) -> bool:
    """Verify user credentials"""
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT password_hash FROM users WHERE email = ?",
            (email,))
        row = cur.fetchone()
        return row and hash_password(password) == row["password_hash"]
    except Exception as e:
        print(f"Login error: {e}")
        return False
    finally:
        if conn:
            conn.close()


def save_workout(user_email: str, date: str, body_part: str, exercise: str,
                 equipment: str, health_condition: str, sets: List[Dict[str, Any]],
                 target_reps: Dict[str, Any], recommendation: Dict[str, Any]):
    """Save workout with transaction"""
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO workouts (
                user_email, date, body_part, exercise, equipment, 
                health_condition, sets_json, target_reps_json, recommendation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            user_email,
            date,
            body_part,
            exercise,
            equipment,
            health_condition,
            json.dumps(sets),
            json.dumps(target_reps),
            json.dumps(recommendation)
        ))
        conn.commit()
    except Exception as e:
        print(f"Error saving workout: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_user_workouts(user_email: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get user's workout history with error handling"""
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT date, body_part, exercise, equipment, health_condition, 
                   sets_json, target_reps_json, recommendation_json
            FROM workouts
            WHERE user_email = ?
            ORDER BY date DESC, id DESC
            LIMIT ?
            """, (user_email, limit))
        return [
            {
                "date": row["date"],
                "body_part": row["body_part"],
                "exercise": row["exercise"],
                "equipment": row["equipment"],
                "health_condition": row["health_condition"],
                "sets": json.loads(row["sets_json"]),
                "target_reps": json.loads(row["target_reps_json"]),
                "recommendation": json.loads(row["recommendation_json"])
            }
            for row in cur.fetchall()
        ]
    except Exception as e:
        print(f"Error fetching workouts: {e}")
        return []
    finally:
        if conn:
            conn.close()


# Initialize database on import
init_db()