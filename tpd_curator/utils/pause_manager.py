

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class PauseState:

    paper_doi: str
    round: int
    results_dir: str
    paper_dir: str
    round_dir: str
    config_path: str
    input_prompt_path: str
    cache_file_path: str
    phase: str
    paper_index: int
    timestamp: str
    batch_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PauseState':
        return cls(**data)


class PauseManager:


    PAUSE_STATE_FILENAME = "pause_state.json"

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.state_file_path = os.path.join(results_dir, self.PAUSE_STATE_FILENAME)

    def save_pause_state(self, state: PauseState) -> None:
        os.makedirs(self.results_dir, exist_ok=True)

        with open(self.state_file_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"\n[PAUSE STATE SAVED] {self.state_file_path}")

    def load_pause_state(self) -> Optional[PauseState]:
        if not os.path.exists(self.state_file_path):
            return None

        try:
            with open(self.state_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            state = PauseState.from_dict(data)
            print(f"\n[PAUSE STATE LOADED] {self.state_file_path}")
            return state
        except Exception as e:
            print(f"[ERROR] Failed to load pause state: {e}")
            return None

    def clear_pause_state(self) -> None:
        if os.path.exists(self.state_file_path):
            os.remove(self.state_file_path)
            print(f"\n[PAUSE STATE CLEARED] {self.state_file_path}")

    def has_pause_state(self) -> bool:
        return os.path.exists(self.state_file_path)

    def get_state_file_path(self) -> str:
        return self.state_file_path


def create_pause_state(
    paper_doi: str,
    round: int,
    results_dir: str,
    paper_dir: str,
    round_dir: str,
    config_path: str,
    input_prompt_path: str,
    cache_file_path: str,
    phase: str,
    paper_index: int
) -> PauseState:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return PauseState(
        paper_doi=paper_doi,
        round=round,
        results_dir=results_dir,
        paper_dir=paper_dir,
        round_dir=round_dir,
        config_path=config_path,
        input_prompt_path=input_prompt_path,
        cache_file_path=cache_file_path,
        phase=phase,
        paper_index=paper_index,
        timestamp=timestamp
    )


def create_pause_state_for_batch(
    paper_doi: str,
    round: int,
    paper_dir: str,
    round_dir: str,
    config_path: str,
    cache_file_path: str,
    paper_index_in_batch: int,
    completed_papers: List[Dict],
    batch_papers: List[Dict],
    prompt_version: int
) -> PauseState:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt_path = str(Path(round_dir).parent / f"{prompt_version:02d}_prompt.md")

    batch_info = {
        "paper_index_in_batch": paper_index_in_batch,
        "completed_papers": completed_papers,
        "batch_papers": batch_papers,
        "prompt_version": prompt_version,
    }

    return PauseState(
        paper_doi=paper_doi,
        round=round,
        results_dir=str(Path(round_dir).parent),
        paper_dir=paper_dir,
        round_dir=round_dir,
        config_path=config_path,
        input_prompt_path=prompt_path,
        cache_file_path=cache_file_path,
        phase="batch_active",
        paper_index=paper_index_in_batch,
        timestamp=timestamp,
        batch_info=batch_info
    )
