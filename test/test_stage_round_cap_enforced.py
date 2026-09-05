"""``MAX_STAGE_ROUNDS`` / ``MAX_STAGE_ESCALATIONS`` must stop a dashboard plan.

``OrchestrationTracker.record_round()`` returns whether the stage has spent its
round budget and ``is_force_failed()`` says whether it has spent its escalations
too. The Slack gateway consulted both; the dashboard's ``_stage_loop`` recorded
the round and threw the answer away, and never called ``is_force_failed`` at all
— so the "max 3 rounds per stage" the orchestrator prompt promises enforced
nothing on the dashboard path (issue #1783).

Where the rounds come from matters for what has to be tested. The loop records
one round per stage ENTRY; the rest are recorded by the subagent-completion
handler against ``tracker.current_stage`` as each spawn wave finishes, i.e. while
the stage is still running. So the enforcing gate is the one AFTER a stage's
subagent wave, and the mocked stage turn below records those extra rounds the
same way the gateway does.

The second gate is the escalation cap on stage entry. It is reachable through a
restart rather than in one run: escalations survive a rehydration whole while the
interrupted stage's rounds are dropped, so without it a plan could be resumed
into a stage that had already exhausted its escalations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kiro_crew.context_management import (
    MAX_STAGE_ESCALATIONS,
    MAX_STAGE_ROUNDS,
    OrchestrationTracker,
)
from kiro_crew.dashboard.state import _ChatSlot


@pytest.fixture(autouse=True)
def _isolate_config_dir(tmp_path, monkeypatch):
    """Stage results are written under ``config_dir()`` — keep them per-test."""
    for module in ("state", "chat", "chat_orchestrator"):
        monkeypatch.setattr(f"kiro_crew.dashboard.{module}.config_dir", lambda: tmp_path)


def _make_state():
    state = MagicMock()
    state.broadcast_ws = MagicMock()
    state.push_slots_update = MagicMock()
    state.subagents = MagicMock()
    state.subagents.running_agents_for = MagicMock(return_value=[])
    return state


def _make_slot(titles=("First", "Second", "Third")):
    slot = _ChatSlot("round-cap-slot", mode="orchestrator")
    slot._auto_run = True
    slot._stage_titles = list(titles)
    slot._plan_goal = "Test goal"
    # A pre-built tracker keeps the loop off its bootstrap path, so no config
    # load runs and the seeded ledger is the one under test.
    slot._orch_tracker = OrchestrationTracker(stage_timeout_seconds=1800)
    return slot


def _stage_turns(monkeypatch, *, extra_rounds_per_stage=0, texts=None):
    """Mock the stage turn, optionally recording subagent-wave rounds.

    ``extra_rounds_per_stage`` mimics the Slack gateway's subagent-completion
    handler, which records a round against ``tracker.current_stage`` each time a
    spawn wave for the running stage finishes.
    """
    box = {"n": 0}

    async def _mock_run_chat(state, slot, message, **kwargs):
        idx = box["n"]
        box["n"] += 1
        body = (texts or [])[idx] if texts and idx < len(texts) else f"stage {idx + 1} output"
        slot.append("assistant", body, "msg msg-a")
        tracker = slot._orch_tracker
        for _ in range(extra_rounds_per_stage):
            tracker.record_round(tracker.current_stage)

    monkeypatch.setattr("kiro_crew.dashboard.chat_orchestrator._run_chat", _mock_run_chat)
    return box


def _assistant_text(slot):
    return "\n".join(m.get("content", "") for m in slot.messages if m.get("role") == "assistant")


def _stages_run(box):
    return box["n"]


# ── The enforcing gate: rounds spent during a stage ──────────────────────────


class TestRoundCapStopsThePlan:
    @pytest.mark.asyncio
    async def test_round_cap_halts_before_the_next_stage(self, monkeypatch):
        """RED BEFORE: the loop advanced through every stage regardless."""
        from kiro_crew.dashboard.chat import _stage_loop

        slot = _make_slot()
        # Stage entry records 1; two waves take it to MAX_STAGE_ROUNDS.
        box = _stage_turns(monkeypatch, extra_rounds_per_stage=MAX_STAGE_ROUNDS - 1)

        await _stage_loop(_make_state(), slot, auto_run=True)

        assert _stages_run(box) == 1, "the plan advanced past a round-capped stage"
        assert "all 3 of its spawn rounds" in _assistant_text(slot)
        assert "✅ All 3 stages complete." not in _assistant_text(slot)

    @pytest.mark.asyncio
    async def test_round_cap_stops_auto_run(self, monkeypatch):
        """A later Go must not silently keep an auto-run plan going."""
        from kiro_crew.dashboard.chat import _stage_loop

        slot = _make_slot()
        box = _stage_turns(monkeypatch, extra_rounds_per_stage=MAX_STAGE_ROUNDS - 1)

        await _stage_loop(_make_state(), slot, auto_run=True)

        assert slot._auto_run is False
        # Paired with the halt: a plan that ran to completion also clears the
        # flag, so the flag alone would assert nothing about the cap.
        assert box["n"] == 1

    @pytest.mark.asyncio
    async def test_capped_stage_keeps_its_result_on_disk(self, monkeypatch, tmp_path):
        """The halt is placed AFTER the capture, so the finished stage is not lost.

        Ordering, not mere existence: the cap could have been enforced before the
        capture, which would throw away a stage that had genuinely finished. The
        halt assertion is what makes this test about that ordering.
        """
        from kiro_crew.dashboard.chat import _stage_loop

        slot = _make_slot()
        box = _stage_turns(monkeypatch, extra_rounds_per_stage=MAX_STAGE_ROUNDS - 1)

        await _stage_loop(_make_state(), slot, auto_run=True)

        assert box["n"] == 1, "the plan did not halt, so this proves nothing"
        result = tmp_path / "sessions" / slot.key / "stage_1_result.md"
        assert result.exists()
        assert slot._orch_tracker._stage_results.get(1) == str(result)
        # And the resume pointer moved past it, so a restart continues at stage 2.
        assert slot._orch_tracker.resume_stage() == 2

    @pytest.mark.asyncio
    async def test_force_failed_stage_reads_as_terminal(self, monkeypatch):
        """Escalations exhausted is a different verdict from 'send guidance'."""
        from kiro_crew.dashboard.chat import _stage_loop

        slot = _make_slot()
        slot._orch_tracker._stage_escalations[1] = MAX_STAGE_ESCALATIONS
        _stage_turns(monkeypatch, extra_rounds_per_stage=MAX_STAGE_ROUNDS - 1)

        await _stage_loop(_make_state(), slot, auto_run=True)

        text = _assistant_text(slot)
        assert f"failed after {MAX_STAGE_ESCALATIONS} escalations" in text
        assert "send guidance" not in text

    @pytest.mark.asyncio
    async def test_round_cap_is_audited(self, monkeypatch):
        """The stop is a security-relevant guard, so it is logged like the others."""
        from kiro_crew.dashboard import chat_orchestrator
        from kiro_crew.dashboard.chat import _stage_loop

        events: list[str] = []
        sink = MagicMock()
        sink.log = MagicMock(side_effect=lambda ev: events.append(ev.operation))
        monkeypatch.setattr(chat_orchestrator, "sel", lambda: sink)

        slot = _make_slot()
        _stage_turns(monkeypatch, extra_rounds_per_stage=MAX_STAGE_ROUNDS - 1)

        await _stage_loop(_make_state(), slot, auto_run=True)

        assert "stage_round_cap" in events

    @pytest.mark.asyncio
    async def test_a_plan_under_its_budget_is_unaffected(self, monkeypatch):
        """Preservation: one round per stage is the normal case and must run through."""
        from kiro_crew.dashboard.chat import _stage_loop

        slot = _make_slot()
        box = _stage_turns(monkeypatch, extra_rounds_per_stage=0)

        await _stage_loop(_make_state(), slot, auto_run=True)

        assert _stages_run(box) == 3
        assert "✅ All 3 stages complete." in _assistant_text(slot)

    @pytest.mark.asyncio
    async def test_one_extra_wave_per_stage_is_still_under_the_cap(self, monkeypatch):
        """Preservation: the cap is 3, so 2 rounds per stage must not trip it."""
        from kiro_crew.dashboard.chat import _stage_loop

        slot = _make_slot()
        box = _stage_turns(monkeypatch, extra_rounds_per_stage=1)

        await _stage_loop(_make_state(), slot, auto_run=True)

        assert _stages_run(box) == 3
        assert "spawn rounds" not in _assistant_text(slot)


# ── The entry gate: escalations restored across a restart ────────────────────


class TestForceFailedStageIsNotRetried:
    @pytest.mark.asyncio
    async def test_resumed_plan_refuses_a_force_failed_stage(self, monkeypatch):
        """RED BEFORE: ``is_force_failed`` had no caller on the dashboard path.

        The shape a restart produces: stage 1 finished, stage 2 was interrupted
        with its escalations already exhausted, so its rounds were dropped on
        restore and the resumed loop is about to re-run it.
        """
        from kiro_crew.dashboard.chat import _stage_loop

        slot = _make_slot()
        tracker = slot._orch_tracker
        tracker.record_round(1)
        tracker.record_stage_result(1, "/tmp/stage_1_result.md")
        tracker._stage_escalations[2] = MAX_STAGE_ESCALATIONS
        box = _stage_turns(monkeypatch)

        await _stage_loop(_make_state(), slot, auto_run=True)

        assert _stages_run(box) == 0, "a force-failed stage was handed another turn"
        assert "already failed after" in _assistant_text(slot)
        assert slot._auto_run is False

    @pytest.mark.asyncio
    async def test_one_escalation_is_not_yet_terminal(self, monkeypatch):
        """Preservation: the cap is 2 escalations, so 1 must still run."""
        from kiro_crew.dashboard.chat import _stage_loop

        slot = _make_slot()
        tracker = slot._orch_tracker
        tracker.record_round(1)
        tracker.record_stage_result(1, "/tmp/stage_1_result.md")
        tracker._stage_escalations[2] = MAX_STAGE_ESCALATIONS - 1
        box = _stage_turns(monkeypatch)

        await _stage_loop(_make_state(), slot, auto_run=True)

        assert _stages_run(box) == 2  # stages 2 and 3
        assert "already failed after" not in _assistant_text(slot)
