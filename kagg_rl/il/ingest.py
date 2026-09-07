"""Ingest Kaggriculture episode dumps for imitation learning.

Supports:
  1) Local flat/bucketed directories of ``*.json`` / ``*.json.gz`` episode files
  2) HF-style store with ``index.csv`` + ``seats.csv`` + ``episodes/{00-99}/``
  3) Filtering to top agents via Elo (from index) and/or final cash / winners
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SeatRef:
    episode_id: str
    seat: int
    agent: str = ""
    reward: float = 0.0
    won: bool = False
    elo: float = 0.0
    path: Optional[Path] = None


def _open_json(path: Path) -> Any:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_episode_files(root: Path) -> List[Path]:
    """Recursively find episode JSON files under root."""
    root = Path(root)
    files = sorted(root.rglob("*.json.gz")) + sorted(root.rglob("*.json"))
    # Prefer gzip when both exist for same stem.
    seen = set()
    out = []
    for p in files:
        key = p.name.replace(".json.gz", "").replace(".json", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def episode_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return path.stem


def load_index_elo(index_csv: Path) -> Dict[str, float]:
    """Map episode_id → elo_max (or elo_avg) from IL store index.csv."""
    out: Dict[str, float] = {}
    with open(index_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = str(row["episode_id"])
            for key in ("elo_max", "elo_avg", "elo_min"):
                if key in row and row[key] not in (None, ""):
                    out[eid] = float(row[key])
                    break
    return out


def load_seat_table(seats_csv: Path) -> List[SeatRef]:
    seats: List[SeatRef] = []
    with open(seats_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seats.append(
                SeatRef(
                    episode_id=str(row["episode_id"]),
                    seat=int(row["seat"]),
                    agent=row.get("agent", ""),
                    reward=float(row.get("reward") or 0),
                    won=bool(int(float(row.get("won") or 0))),
                )
            )
    return seats


def select_top_seats(
    seats: List[SeatRef],
    *,
    min_reward: float = 0.0,
    min_elo: float = 0.0,
    elo_by_episode: Optional[Dict[str, float]] = None,
    winners_only: bool = False,
    top_agents: Optional[Set[str]] = None,
    top_k_by_reward: Optional[int] = None,
) -> List[SeatRef]:
    elo_by_episode = elo_by_episode or {}
    filtered: List[SeatRef] = []
    for s in seats:
        s.elo = elo_by_episode.get(s.episode_id, 0.0)
        if s.reward < min_reward:
            continue
        if min_elo and s.elo < min_elo:
            continue
        if winners_only and not s.won:
            continue
        if top_agents and s.agent not in top_agents:
            continue
        filtered.append(s)

    filtered.sort(key=lambda x: (x.reward, x.elo), reverse=True)
    if top_k_by_reward is not None:
        filtered = filtered[:top_k_by_reward]
    return filtered


def attach_paths(seats: List[SeatRef], episodes_root: Path) -> List[SeatRef]:
    """Resolve episode file paths for seat refs.

    Looks for ``{id}.json.gz`` / ``{id}.json`` under root or
    ``episodes/{id%100:02d}/`` buckets.
    """
    episodes_root = Path(episodes_root)
    resolved: List[SeatRef] = []
    for s in seats:
        candidates = [
            episodes_root / f"{s.episode_id}.json.gz",
            episodes_root / f"{s.episode_id}.json",
            episodes_root / "episodes" / f"{int(s.episode_id) % 100:02d}" / f"{s.episode_id}.json.gz",
            episodes_root / f"{int(s.episode_id) % 100:02d}" / f"{s.episode_id}.json.gz",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            # Fall back to rglob once (slow); cache not needed for small sets.
            matches = list(episodes_root.rglob(f"{s.episode_id}.json*"))
            path = matches[0] if matches else None
        if path is None:
            logger.debug("missing episode file for %s", s.episode_id)
            continue
        s.path = path
        resolved.append(s)
    return resolved


def seats_from_local_episodes(
    episodes_root: Path,
    *,
    min_reward: float = 0.0,
    winners_only: bool = False,
) -> List[SeatRef]:
    """Build seat refs directly from episode files (no seats.csv required)."""
    seats: List[SeatRef] = []
    for path in discover_episode_files(episodes_root):
        try:
            ep = _open_json(path)
        except Exception as e:
            logger.warning("skip unreadable %s: %s", path, e)
            continue
        eid = str(ep.get("info", {}).get("EpisodeId") or episode_id_from_path(path))
        rewards = ep.get("rewards") or [0, 0]
        agents = (
            ep.get("info", {}).get("TeamNames")
            or ep.get("info", {}).get("Agents")
            or ["", ""]
        )
        for seat in (0, 1):
            reward = float(rewards[seat]) if seat < len(rewards) else 0.0
            if reward < min_reward:
                continue
            won = False
            if len(rewards) >= 2:
                won = reward > float(rewards[1 - seat])
            if winners_only and not won:
                continue
            seats.append(
                SeatRef(
                    episode_id=eid,
                    seat=seat,
                    agent=str(agents[seat]) if seat < len(agents) else "",
                    reward=reward,
                    won=won,
                    path=path,
                )
            )
    seats.sort(key=lambda x: x.reward, reverse=True)
    return seats


def iter_seat_transitions(seat: SeatRef) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield (observation, action) pairs for one seat across an episode.

    Uses step[t] observation with step[t+1] action when available; otherwise
    pairs step[t].observation with step[t].action (Kaggle tape convention
    varies; we prefer action recorded on the same step as observation).
    """
    assert seat.path is not None
    ep = _open_json(seat.path)
    steps = ep.get("steps") or []
    for step in steps:
        if not isinstance(step, list) or seat.seat >= len(step):
            continue
        rec = step[seat.seat] or {}
        obs = rec.get("observation")
        action = rec.get("action")
        if obs is None or action is None:
            continue
        # Ensure step index present for features.
        if "step" not in obs:
            obs = dict(obs)
            obs["step"] = obs.get("day", 0) * 24 + obs.get("hour", 0)
        yield obs, action
