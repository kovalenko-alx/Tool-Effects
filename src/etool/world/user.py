"""Scripted user simulator for model rollouts (gold rollouts drive the user from the plan directly)."""
from __future__ import annotations

import re
from typing import Optional

from etool.world.tasks import Task

QUESTION = re.compile(r"\?\s*$|\?\s*\n|\b(shall|should|would you like|do you want|can you confirm|please confirm|which|what time|when)\b", re.I)
CONFIRM_HINT = re.compile(r"\b(proceed|confirm|go ahead|shall i|should i|send it|delete|cancel|permanent|irreversible|about to|pay|charge|refund|place (the |this |your |an )?order|purchase)\b", re.I)


class ScriptedUser:
    """Answers confirmation requests with yes, supplies missing info once, otherwise ends the conversation."""

    def __init__(self, task: Task, max_followups: int = 3):
        self.task = task
        self.max_followups = max_followups
        self.followups = 0
        self.info_given = task.missing_info is None
        self.confirmed = not task.requires_confirmation
        self.clarified = task.on_question is None

    def first(self) -> str:
        return self.task.utterance

    def respond(self, assistant_text: str) -> Optional[str]:
        """Return the next user message, or None when the conversation should end."""
        text = assistant_text.strip()
        if not text:
            return None
        asks = bool(QUESTION.search(text))
        if self.followups >= self.max_followups:
            return None
        if asks and not self.clarified:  # e.g. "Did you mean X?" on a request about a non-existent entity: restate, never a blanket yes
            self.clarified = True
            self.followups += 1
            return self.task.on_question
        if not self.info_given and asks and not CONFIRM_HINT.search(text):
            self.info_given = True
            self.followups += 1
            return self.task.missing_info["answer"]
        if asks and (CONFIRM_HINT.search(text) or not self.confirmed):
            self.confirmed = True
            self.followups += 1
            return "Yes, please go ahead."
        if asks and not self.info_given:
            self.info_given = True
            self.followups += 1
            return self.task.missing_info["answer"]
        if asks:
            self.followups += 1
            return "Please just do what I asked."
        return None
