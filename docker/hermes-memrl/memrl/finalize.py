"""The MemRL mini-batch update, run once per batch by the host.

A batched provider parks each task's session in memrl/pending.json; the host
(ARIES) collects them into memrl/pending/, records each task's verdict in
memrl/rewards.json, and runs this module over the run's store:

    python -m plugins.memory.memrl.finalize

Sessions are applied in execution order (the -NNN suffix of the task ID),
each exactly as the sequential path applies one: utility updates for the
memories it recalled, then a new experience or reflection. This matches
MemRL's buffered flush (memrl/run/bcb_runner.py _flush_memory_updates). A
null reward drops the session, as it does sequentially. A missing reward is
an error: the host records every task of a batch before its update.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from . import REWARDS_FILE, MemRLConfig, MemRLMemoryProvider
from .procedure import ChatFn, hermes_chat
from .store import Store, read_pending

PENDING_DIR = "pending"


def _order(task_id: str) -> int:
    suffix = task_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def finalize_batch(home: Path, chat: Optional[ChatFn] = None, config: Optional[MemRLConfig] = None) -> Dict[str, int]:
    memrl_dir = home / "memrl"
    rewards = json.loads((memrl_dir / REWARDS_FILE).read_text())
    parked = sorted((read_pending(p) for p in (memrl_dir / PENDING_DIR).glob("*.json")),
                    key=lambda p: (_order(p.task_id), p.task_id))
    missing = [p.task_id for p in parked if p.task_id not in rewards]
    if missing:
        raise ValueError(f"no recorded reward for {', '.join(missing)}")
    provider = MemRLMemoryProvider(config=config or MemRLConfig.from_env(), chat=chat)
    provider._home = home
    provider._chat = chat or hermes_chat(home)
    provider._store = Store(memrl_dir / "memrl.db")
    counts = {"applied": 0, "dropped": 0}
    try:
        for pending in parked:
            reward = rewards[pending.task_id]
            if reward is None:
                counts["dropped"] += 1
                continue
            provider._finalize(pending, float(reward))
            counts["applied"] += 1
    finally:
        provider._store.close()
    return counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    print(json.dumps(finalize_batch(home)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
