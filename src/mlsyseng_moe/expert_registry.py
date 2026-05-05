"""Expert registry for managing chapter-based ML experts."""

import json
import logging
from typing import Any, Dict, List, Optional

from .database import Expert, MoEDatabase

logger = logging.getLogger(__name__)


class ExpertRegistry:
    """Manages expert definitions and queries against specific experts."""

    def __init__(self, db: MoEDatabase):
        self.db = db

    def register_expert(
        self,
        expert_name: str,
        slug: str,
        chapter_id: str,
        capabilities: List[str],
        skills: List[str],
        strategy: str = "Baseline → EDA → Feature Engineering → Model Selection → Submit",
        formula: Optional[Dict[str, Any]] = None,
        loop_config: Optional[Dict[str, Any]] = None,
    ) -> Expert:
        """Register or update an expert in the database."""
        import time

        if formula is None:
            formula = {
                "objective": "minimize_validation_loss",
                "function": "L = f(X, θ, α)",
                "metrics": ["accuracy"],
            }
        if loop_config is None:
            loop_config = {
                "objective": "minimize_validation_loss",
                "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
                "epsilon": 0.001,
                "max_iterations": 10,
                "patience": 3,
            }

        expert = Expert(
            expert_name=expert_name,
            slug=slug,
            chapter_id=chapter_id,
            capabilities=json.dumps(capabilities),
            skills=json.dumps(skills),
            strategy=strategy,
            formula=json.dumps(formula),
            loop_config=json.dumps(loop_config),
            created_at=time.time(),
        )
        self.db.upsert_expert(expert)
        logger.info(f"Registered expert: {expert_name} (slug={slug})")
        return expert

    def get_expert(self, name_or_slug: str) -> Optional[Expert]:
        """Look up an expert by name or slug."""
        expert = self.db.get_expert(name_or_slug)
        if expert:
            return expert
        return self.db.get_expert_by_slug(name_or_slug)

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all experts as dicts."""
        return [e.to_dict() for e in self.db.list_experts()]

    def ask_expert(self, name_or_slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert - returns the expert's knowledge context for the question."""
        expert = self.get_expert(name_or_slug)
        if not expert:
            return {"error": f"Expert '{name_or_slug}' not found"}

        chapter = self.db.get_chapter(expert.chapter_id)
        chapter_content = chapter.content_md[:2000] if chapter else ""

        expert_dict = expert.to_dict()
        return {
            "expert": expert_dict,
            "chapter_summary": chapter_content,
            "question": question,
            "guidance": _build_expert_guidance(expert_dict, question),
        }

    def select_experts_for_task(self, task_description: str, max_experts: int = 3) -> List[Dict[str, Any]]:
        """Select the most relevant experts for a given task using keyword matching."""
        all_experts = self.db.list_experts()
        scored = []

        task_lower = task_description.lower()
        for expert in all_experts:
            score = 0
            ed = expert.to_dict()
            for cap in ed.get("capabilities", []):
                if any(w in task_lower for w in cap.lower().split()):
                    score += 1
            for skill in ed.get("skills", []):
                skill_name = skill.split("/")[-1] if "/" in skill else skill
                if any(w in task_lower for w in skill_name.replace("-", " ").split()):
                    score += 1

            name_words = expert.expert_name.lower().split()
            if any(w in task_lower for w in name_words if len(w) > 3):
                score += 2

            if score > 0:
                scored.append((score, ed))

        scored.sort(key=lambda x: -x[0])
        return [item[1] for item in scored[:max_experts]]


def _build_expert_guidance(expert: Dict[str, Any], question: str) -> str:
    """Build a guidance string from expert knowledge for a given question."""
    parts = [f"# Expert: {expert['expert_name']}"]
    parts.append("")

    caps = expert.get("capabilities", [])
    if caps:
        parts.append("## Capabilities")
        for c in caps:
            parts.append(f"- {c}")
        parts.append("")

    parts.append(f"## Strategy: {expert.get('strategy', 'N/A')}")
    parts.append("")

    formula = expert.get("formula", {})
    if formula:
        parts.append("## Objective")
        parts.append(f"- Function: {formula.get('function', 'N/A')}")
        parts.append(f"- Metrics: {', '.join(formula.get('metrics', []))}")
        parts.append("")

    loop = expert.get("loop_config", {})
    if loop:
        parts.append("## Convergence Loop")
        parts.append(f"- Exit: {loop.get('exit_condition', 'N/A')}")
        parts.append(f"- ε = {loop.get('epsilon', 0.001)}")
        parts.append(f"- Max iterations: {loop.get('max_iterations', 10)}")
        parts.append("")

    skills = expert.get("skills", [])
    if skills:
        parts.append("## Recommended Skills")
        for s in skills:
            parts.append(f"- {s}")
        parts.append("")

    parts.append(f"## Question: {question}")
    return "\n".join(parts)
