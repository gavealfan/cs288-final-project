"""
Organizational role taxonomy and interaction context definitions.

A role is a structured tuple: (hierarchy, function, power_relation, seniority)
An interaction context defines the workstream setting.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Dict, List, Optional, Tuple
import random
import json


class Hierarchy(Enum):
    EXECUTIVE = "executive"
    MIDDLE_MANAGER = "middle_manager"
    TEAM_LEAD = "team_lead"
    INDIVIDUAL_CONTRIBUTOR = "individual_contributor"
    INTERN = "intern"


class Function(Enum):
    ENGINEERING = "engineering"
    PRODUCT = "product"
    DESIGN = "design"
    MARKETING = "marketing"
    FINANCE = "finance"
    HR = "human_resources"
    OPERATIONS = "operations"


class PowerRelation(Enum):
    SUPERIOR = "superior"        # speaker outranks interlocutor
    SUBORDINATE = "subordinate"  # speaker is below interlocutor
    PEER = "peer"                # same level
    CROSS_FUNCTIONAL = "cross_functional"  # no direct authority


class Seniority(Enum):
    NEW_HIRE = "new_hire"        # < 6 months
    EXPERIENCED = "experienced"  # 1-5 years
    VETERAN = "veteran"          # 5+ years, institutional knowledge


class InteractionContext(Enum):
    TASK_DELEGATION = "task_delegation"
    FEEDBACK_GIVING = "feedback_giving"
    FEEDBACK_RECEIVING = "feedback_receiving"
    STATUS_REPORTING_UP = "status_reporting_up"
    STATUS_REPORTING_DOWN = "status_reporting_down"
    STATUS_REPORTING_LATERAL = "status_reporting_lateral"
    DECISION_MAKING = "decision_making"
    CROSS_FUNCTIONAL_COORDINATION = "cross_functional_coordination"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    CONFLICT_RESOLUTION = "conflict_resolution"
    ESCALATION = "escalation"
    MENTORING = "mentoring"


@dataclass
class OrganizationalRole:
    hierarchy: Hierarchy
    function: Function
    power_relation: PowerRelation
    seniority: Seniority

    def to_prompt_description(self) -> str:
        hierarchy_desc = {
            Hierarchy.EXECUTIVE: "a C-level executive",
            Hierarchy.MIDDLE_MANAGER: "a middle manager",
            Hierarchy.TEAM_LEAD: "a team lead",
            Hierarchy.INDIVIDUAL_CONTRIBUTOR: "an individual contributor",
            Hierarchy.INTERN: "an intern",
        }
        function_desc = {
            Function.ENGINEERING: "engineering",
            Function.PRODUCT: "product management",
            Function.DESIGN: "design",
            Function.MARKETING: "marketing",
            Function.FINANCE: "finance",
            Function.HR: "human resources",
            Function.OPERATIONS: "operations",
        }
        power_desc = {
            PowerRelation.SUPERIOR: "who outranks the person they are speaking with",
            PowerRelation.SUBORDINATE: "who reports to the person they are speaking with",
            PowerRelation.PEER: "who is at the same level as the person they are speaking with",
            PowerRelation.CROSS_FUNCTIONAL: "who has no direct authority over the person they are speaking with (cross-functional)",
        }
        seniority_desc = {
            Seniority.NEW_HIRE: "recently joined the company (less than 6 months)",
            Seniority.EXPERIENCED: "has 1-5 years at the company",
            Seniority.VETERAN: "is a long-tenured employee with deep institutional knowledge",
        }
        return (
            f"You are {hierarchy_desc[self.hierarchy]} in {function_desc[self.function]}, "
            f"{power_desc[self.power_relation]}. "
            f"You {seniority_desc[self.seniority]}."
        )

    def to_dict(self) -> dict:
        return {
            "hierarchy": self.hierarchy.value,
            "function": self.function.value,
            "power_relation": self.power_relation.value,
            "seniority": self.seniority.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrganizationalRole":
        return cls(
            hierarchy=Hierarchy(d["hierarchy"]),
            function=Function(d["function"]),
            power_relation=PowerRelation(d["power_relation"]),
            seniority=Seniority(d["seniority"]),
        )


CONTEXT_DESCRIPTIONS = {
    InteractionContext.TASK_DELEGATION: (
        "You are assigning a task to someone. Describe the task clearly, "
        "set expectations, and provide any necessary context."
    ),
    InteractionContext.FEEDBACK_GIVING: (
        "You are providing feedback on someone's recent work. "
        "Address both strengths and areas for improvement."
    ),
    InteractionContext.FEEDBACK_RECEIVING: (
        "You have just received feedback on your work. "
        "Respond to the feedback constructively."
    ),
    InteractionContext.STATUS_REPORTING_UP: (
        "You are reporting the status of your project or workstream to your manager or leadership. "
        "Summarize progress, blockers, and next steps."
    ),
    InteractionContext.STATUS_REPORTING_DOWN: (
        "You are sharing a project status update with your team. "
        "Communicate priorities and any changes."
    ),
    InteractionContext.STATUS_REPORTING_LATERAL: (
        "You are sharing a status update with a peer or cross-functional colleague. "
        "Coordinate on shared dependencies."
    ),
    InteractionContext.DECISION_MAKING: (
        "You are participating in a discussion about an important decision. "
        "Present your perspective and reasoning."
    ),
    InteractionContext.CROSS_FUNCTIONAL_COORDINATION: (
        "You are coordinating with someone from a different team on a shared initiative. "
        "You have no direct authority over them."
    ),
    InteractionContext.KNOWLEDGE_SHARING: (
        "You are explaining a concept, process, or piece of institutional knowledge to a colleague."
    ),
    InteractionContext.CONFLICT_RESOLUTION: (
        "There is a disagreement about priorities, approach, or resource allocation. "
        "Work toward a resolution."
    ),
    InteractionContext.ESCALATION: (
        "An issue has arisen that exceeds your authority or capability to resolve. "
        "Escalate it appropriately."
    ),
    InteractionContext.MENTORING: (
        "You are mentoring a more junior colleague. "
        "Provide guidance, share experience, and support their development."
    ),
}


# Validity constraints: filter out implausible role-context combinations
def is_valid_combination(role: OrganizationalRole, context: InteractionContext) -> bool:
    # Interns don't delegate tasks or mentor
    if role.hierarchy == Hierarchy.INTERN:
        if context in (InteractionContext.TASK_DELEGATION, InteractionContext.MENTORING,
                       InteractionContext.STATUS_REPORTING_DOWN):
            return False
    # Subordinates don't delegate to or report down to their superiors
    if role.power_relation == PowerRelation.SUBORDINATE:
        if context in (InteractionContext.TASK_DELEGATION, InteractionContext.STATUS_REPORTING_DOWN):
            return False
    # Superiors don't report up to subordinates or receive escalations from above
    if role.power_relation == PowerRelation.SUPERIOR:
        if context in (InteractionContext.STATUS_REPORTING_UP, InteractionContext.ESCALATION):
            return False
    # Cross-functional roles shouldn't delegate tasks (no authority)
    if role.power_relation == PowerRelation.CROSS_FUNCTIONAL:
        if context == InteractionContext.TASK_DELEGATION:
            return False
    return True


def generate_all_valid_combinations() -> List[Tuple[OrganizationalRole, InteractionContext]]:
    combinations = []
    for h, f, p, s in product(Hierarchy, Function, PowerRelation, Seniority):
        role = OrganizationalRole(h, f, p, s)
        for ctx in InteractionContext:
            if is_valid_combination(role, ctx):
                combinations.append((role, ctx))
    return combinations


def sample_combinations(
    n: int,
    seed: int = 42,
    holdout_fraction: float = 0.15,
) -> Tuple[List[Tuple[OrganizationalRole, InteractionContext]],
           List[Tuple[OrganizationalRole, InteractionContext]]]:
    """
    Sample n role-context combinations for training, holding out a fraction
    for evaluation of generalization to unseen combinations.
    """
    all_combos = generate_all_valid_combinations()
    rng = random.Random(seed)
    rng.shuffle(all_combos)

    holdout_size = int(len(all_combos) * holdout_fraction)
    holdout = all_combos[:holdout_size]
    train_pool = all_combos[holdout_size:]

    train_sample = rng.sample(train_pool, min(n, len(train_pool)))
    return train_sample, holdout


if __name__ == "__main__":
    all_combos = generate_all_valid_combinations()
    print(f"Total valid role-context combinations: {len(all_combos)}")

    train, holdout = sample_combinations(200)
    print(f"Training sample: {train[0][0].to_prompt_description()}")
    print(f"Context: {CONTEXT_DESCRIPTIONS[train[0][1]]}")
    print(f"\nHoldout size: {len(holdout)}")
