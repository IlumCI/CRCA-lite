"""Event-sourced ledger for policy engine.

This module provides a SQLite-based append-only event store for the policy engine.
The ledger serves as:
- Dataset for online learning
- Audit trail for decisions
- Replay engine for deterministic execution

All events are hashed and stored immutably (append-only).
"""

import sqlite3
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from schemas.policy import LedgerEvent


class Ledger:
    """SQLite-based append-only event ledger.
    
    Schema:
        events(
            id INTEGER PRIMARY KEY,
            type TEXT,
            epoch INTEGER,
            hash TEXT UNIQUE,
            payload_json TEXT
        )
    
    Indexed on (epoch, type) for fast window queries.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize ledger with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                hash TEXT UNIQUE NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_epoch_type 
            ON events(epoch, type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_epoch 
            ON events(epoch)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def append(self, event: LedgerEvent) -> int:
        """
        Append an event to the ledger.
        
        Args:
            event: LedgerEvent to append
            
        Returns:
            int: Event ID
            
        Raises:
            sqlite3.IntegrityError: If hash already exists (duplicate event)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO events (type, epoch, hash, payload_json)
                VALUES (?, ?, ?, ?)
            """, (
                event.type,
                event.epoch,
                event.hash,
                json.dumps(event.payload)
            ))
            conn.commit()
            event_id = cursor.lastrowid
            logger.debug(f"Appended event {event_id} (type={event.type}, epoch={event.epoch})")
            return event_id
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.warning(f"Duplicate event hash {event.hash} - skipping")
                # Return existing event ID
                cursor.execute("SELECT id FROM events WHERE hash = ?", (event.hash,))
                row = cursor.fetchone()
                return row["id"] if row else -1
            raise
    
    def window(self, epoch: int, k: int, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get events from the last k epochs ending at epoch.
        
        Args:
            epoch: End epoch (inclusive)
            k: Number of epochs to look back
            event_type: Optional event type filter
            
        Returns:
            List[Dict[str, Any]]: List of event dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        start_epoch = max(0, epoch - k + 1)
        
        if event_type:
            cursor.execute("""
                SELECT * FROM events
                WHERE epoch >= ? AND epoch <= ? AND type = ?
                ORDER BY epoch ASC, id ASC
            """, (start_epoch, epoch, event_type))
        else:
            cursor.execute("""
                SELECT * FROM events
                WHERE epoch >= ? AND epoch <= ?
                ORDER BY epoch ASC, id ASC
            """, (start_epoch, epoch))
        
        rows = cursor.fetchall()
        events = []
        for row in rows:
            events.append({
                "id": row["id"],
                "type": row["type"],
                "epoch": row["epoch"],
                "hash": row["hash"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"]
            })
        
        return events
    
    def latest(self, event_type: Optional[str] = None, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest event (optionally filtered by type and epoch).
        
        Args:
            event_type: Optional event type filter
            epoch: Optional epoch filter
            
        Returns:
            Optional[Dict[str, Any]]: Latest event or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if event_type and epoch is not None:
            cursor.execute("""
                SELECT * FROM events
                WHERE type = ? AND epoch = ?
                ORDER BY id DESC
                LIMIT 1
            """, (event_type, epoch))
        elif event_type:
            cursor.execute("""
                SELECT * FROM events
                WHERE type = ?
                ORDER BY epoch DESC, id DESC
                LIMIT 1
            """, (event_type,))
        elif epoch is not None:
            cursor.execute("""
                SELECT * FROM events
                WHERE epoch = ?
                ORDER BY id DESC
                LIMIT 1
            """, (epoch,))
        else:
            cursor.execute("""
                SELECT * FROM events
                ORDER BY epoch DESC, id DESC
                LIMIT 1
            """)
        
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "type": row["type"],
                "epoch": row["epoch"],
                "hash": row["hash"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"]
            }
        return None
    
    def iter_range(
        self,
        t0: int,
        t1: int,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Iterate over events in epoch range [t0, t1] (inclusive).
        
        Args:
            t0: Start epoch (inclusive)
            t1: End epoch (inclusive)
            event_type: Optional event type filter
            
        Returns:
            List[Dict[str, Any]]: List of event dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if event_type:
            cursor.execute("""
                SELECT * FROM events
                WHERE epoch >= ? AND epoch <= ? AND type = ?
                ORDER BY epoch ASC, id ASC
            """, (t0, t1, event_type))
        else:
            cursor.execute("""
                SELECT * FROM events
                WHERE epoch >= ? AND epoch <= ?
                ORDER BY epoch ASC, id ASC
            """, (t0, t1))
        
        rows = cursor.fetchall()
        events = []
        for row in rows:
            events.append({
                "id": row["id"],
                "type": row["type"],
                "epoch": row["epoch"],
                "hash": row["hash"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"]
            })
        
        return events
    
    def compute_deltas(self, epoch: int) -> Dict[str, float]:
        """
        Compute feature deltas between observation and outcome at epoch t.
        
        Args:
            epoch: Epoch to compute deltas for
            
        Returns:
            Dict[str, float]: Dictionary mapping metric names to deltas
        """
        obs = self.latest(event_type="observation", epoch=epoch)
        outcome = self.latest(event_type="outcome", epoch=epoch)
        
        if not obs or not outcome:
            return {}
        
        obs_metrics = obs["payload"].get("metrics", {})
        outcome_metrics = outcome["payload"].get("metrics", {})
        
        deltas = {}
        all_metrics = set(obs_metrics.keys()) | set(outcome_metrics.keys())
        
        for metric in all_metrics:
            obs_val = obs_metrics.get(metric, 0.0)
            outcome_val = outcome_metrics.get(metric, 0.0)
            deltas[metric] = outcome_val - obs_val
        
        return deltas
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

