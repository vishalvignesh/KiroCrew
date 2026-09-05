"""An orchestrator plan must survive a gateway restart with a resume path.

The autopilot's execution pointer lived only in memory: ``_stage_titles``,
``_plan_goal``, ``_auto_run`` and the ``OrchestrationTracker``'s round /
escalation / stage-result ledger all hang off ``_ChatSlot``, and the dashboard's
slot save serialised none of them. The per-stage result FILES survived a restart,
so the work was on disk, but nothing recorded which stage was next -- a restart
mid-plan lost the run outright, with no way to pick it back up (issue #1783).

Three properties are asserted here, and they are separable:

* the tracker's ledger round-trips through a plain JSON snapshot, and the
  interrupted stage's rounds are dropped so a resumed loop re-runs that stage
  instead of stepping past it;
* the slot save writes the plan record, and both rehydration paths read it back;
* a restored, unfinished plan surfaces a resume offer -- and a plan that
  finished, was cancelled, or never started surfaces nothing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from chat_test_helpers import _make_ready_kiro_prerequisite

from kiro_crew.context_management import OrchestrationTracker
from kiro_crew.dashboard.chat_persistence import (
    _PLAN_RESUME_MARKER,
    _append_plan_resume_offer,
    _confine_restored_result_paths,
    _plan_state_for_save,
    _rehydrate_slot_from_history,
    _restore_plan_state,
    _save_slot_to_history,
    restore_recent_sessions,
)
from kiro_crew.dashboard.state import DashboardState, _ChatSlot
from kiro_crew.history import SLOT_OWNED_META_KEYS, ConversationLog


def _patch_config_dir(monkeypatch, tmp_path):
    """Point every ``config_dir`` the plan round-trip reads at *tmp_path*.

    ``chat_persistence`` is included because the restore allow-lists a stage-result
    path against ``config_dir() / "sessions" / <slot>``; leaving it on the live data
    home would reject the fixture's paths and quietly move each resume assertion.
    """
    for module in ("state", "chat_persistence"):
        monkeypatch.setattr(f"kiro_crew.dashboard.{module}.config_dir", lambda: tmp_path)


def _make_state(tmp_path) -> DashboardState:
    sessions = MagicMock(count=0)
    sessions.remove = AsyncMock()
    sessions.recycle_background = AsyncMock()
    sessions.get_pid = MagicMock(return_value=None)
    state = DashboardState(
        sessions=sessions,
        crons=MagicMock(list_jobs=MagicMock(return_value=[]), status=MagicMock(return_value={})),
        lessons=MagicMock(load_all=MagicMock(return_value=[])),
        start_time=0.0,
        conversation_log=ConversationLog(base_dir=tmp_path),
    )
    state.kiro_prerequisite_service = _make_ready_kiro_prerequisite()
    return state


def _stage_result_path(root, slot_name, stage):
    """The path ``_write_stage_result`` would produce, as a string.

    Built rather than invented: a restored result path is allow-listed against
    exactly this, so a fixture using an arbitrary ``/tmp`` path would exercise the
    REJECTED branch and every resume assertion below would silently shift by a
    stage.
    """
    return str(root / "sessions" / slot_name / f"stage_{stage}_result.md")


def _plan_slot(
    state,
    root,
    name="plan-slot",
    *,
    titles=("Recon", "Implement", "Verify"),
    results=(1,),
    rounds=None,
    escalations=None,
    auto_run=True,
):
    """A live orchestrator slot mid-plan: *results* stages have finished."""
    slot = state.get_or_create_slot(name)
    slot.mode = "orchestrator"
    slot._stage_titles = list(titles)
    slot._stage_descriptions = [[f"- do {t.lower()}"] for t in titles]
    slot._plan_goal = "Ship the thing"
    slot._auto_run = auto_run
    tracker = OrchestrationTracker()
    for stage in rounds if rounds is not None else range(1, len(results) + 2):
        tracker.record_round(stage)
    for stage in results:
        tracker.record_stage_result(stage, _stage_result_path(root, name, stage))
    for stage in escalations or ():
        tracker._stage_escalations[stage] = tracker._stage_escalations.get(stage, 0) + 1
    slot._orch_tracker = tracker
    slot.append("user", "plan it")
    slot.drain()
    return slot


# ── The tracker's own snapshot contract ──────────────────────────────────────


class TestTrackerSnapshot:
    def test_snapshot_is_json_shaped(self):
        """Keys are strings: this record round-trips through the history JSONL."""
        tracker = OrchestrationTracker()
        tracker.record_round(1)
        tracker.record_stage_result(1, "/tmp/one.md")

        snap = tracker.snapshot()

        assert snap == {
            "stage_rounds": {"1": 1},
            "stage_escalations": {},
            "stage_results": {"1": "/tmp/one.md"},
            # Carried explicitly: the ledgers cannot always answer it, because the
            # resume filter empties them for a stage-1 interruption.
            "started": True,
        }

    def test_snapshot_survives_concurrent_mutation(self):
        """RED BEFORE: the slot save runs off-loop and iterated the live ledger.

        ``_plan_state_for_save`` is reached from ``_save_slot_to_history`` on a
        worker thread, so a plan that finished a stage during its own save could
        raise "dictionary changed size during iteration" and lose the record.
        """
        import threading

        tracker = OrchestrationTracker()
        for stage in range(1, 200):
            tracker.record_round(stage)
            tracker.record_stage_result(stage, f"/tmp/stage_{stage}.md")

        errors: list[BaseException] = []
        stop = threading.Event()

        def _mutate():
            n = 1000
            while not stop.is_set():
                tracker.record_round(n)
                tracker.record_stage_result(n, f"/tmp/stage_{n}.md")
                tracker._stage_rounds.pop(n, None)
                tracker._stage_results.pop(n, None)
                n += 1

        def _snapshot():
            try:
                for _ in range(2000):
                    tracker.snapshot()
            except BaseException as exc:  # pragma: no cover - the defect path
                errors.append(exc)

        writer = threading.Thread(target=_mutate, daemon=True)
        writer.start()
        try:
            _snapshot()
        finally:
            stop.set()
            writer.join(timeout=5)

        assert errors == [], f"snapshot raced tracker mutation: {errors[:1]}"

    def test_resume_stage_is_the_first_stage_with_no_result(self):
        tracker = OrchestrationTracker()
        tracker.record_stage_result(1, "/tmp/one.md")
        tracker.record_stage_result(2, "/tmp/two.md")

        assert tracker.resume_stage() == 3

    def test_resume_stage_is_one_when_nothing_finished(self):
        assert OrchestrationTracker().resume_stage() == 1

    def test_interrupted_stage_rounds_are_dropped_on_restore(self):
        """RED BEFORE: the resumed loop must re-run the interrupted stage.

        ``_stage_loop`` derives its starting index from ``current_stage`` -- the
        highest stage with a recorded round -- so a restore that carried stage
        3's round through would start the resumed plan at stage 4 and silently
        skip the work that was interrupted.
        """
        live = OrchestrationTracker()
        for stage in (1, 2, 3):
            live.record_round(stage)
        live.record_stage_result(1, "/tmp/one.md")
        live.record_stage_result(2, "/tmp/two.md")

        restored = OrchestrationTracker.from_snapshot(live.snapshot())

        assert restored.resume_stage() == 3
        # current_stage is what the loop turns into its 0-based start index.
        assert restored.current_stage == 2
        assert restored.round_count(3) == 0

    def test_escalations_survive_restore_whole(self):
        """The harder cap must not be launderable by restarting the gateway."""
        live = OrchestrationTracker()
        live.record_round(2)
        live._stage_escalations[2] = 2

        restored = OrchestrationTracker.from_snapshot(live.snapshot())

        assert restored.is_force_failed(2) is True

    def test_a_force_failed_stage_counts_as_started(self):
        """RED BEFORE: escalations alone read as 'never started'.

        This is the shape a force-failed stage actually reaches a restore in, and
        it carries NEITHER of the other two signals: ``reset_after_guidance``
        zeroes the stage's rounds as it increments the escalation, and
        ``from_snapshot`` then drops the rounds of a stage that produced no
        result. The escalation ledger is the only trace left, so a ``started``
        that ignores it makes the restore discard the tracker — handing the plan a
        clean ledger and letting the exhausted stage run again.
        """
        live = OrchestrationTracker()
        live.record_round(1)
        live._stage_escalations[1] = 2
        live.reset_after_guidance()

        restored = OrchestrationTracker.from_snapshot(live.snapshot())

        assert restored._stage_rounds == {}
        assert restored._stage_results == {}
        assert restored.started is True
        assert restored.is_force_failed(1) is True

    def test_an_armed_but_unrun_plan_is_not_started(self):
        """Preservation: the ARMED/RUNNING distinction still holds."""
        assert OrchestrationTracker().started is False
        assert OrchestrationTracker.from_snapshot({}).started is False

    def test_a_first_stage_restart_still_counts_as_started(self):
        """RED BEFORE: the most ordinary restart of all read as never-started.

        A plan interrupted during stage 1 has one recorded round, no result and no
        escalation. ``resume_stage()`` is 1, so the ``stage < resume`` filter drops
        that round and every ledger ends up empty -- and a tracker that reads as
        never-started is DISCARDED by the restore, losing the plan.
        """
        live = OrchestrationTracker()
        live.record_round(1)

        restored = OrchestrationTracker.from_snapshot(live.snapshot())

        assert restored._stage_rounds == {}, "the resume filter should still drop it"
        assert restored._stage_results == {}
        assert restored._stage_escalations == {}
        assert restored.started is True
        # And the resume point is stage 1, so that stage re-runs from the start.
        assert restored.resume_stage() == 1

    def test_a_record_with_no_rounds_at_all_is_still_not_started(self):
        """The fact is taken from the record, not invented by passing through it."""
        restored = OrchestrationTracker.from_snapshot(
            {"stage_rounds": {}, "stage_escalations": {}, "stage_results": {}}
        )

        assert restored.started is False

    @pytest.mark.parametrize(
        "not_true", ["false", "true", "yes", "0", 1, -1, [], {}, "  ", 0.0, None]
    )
    def test_only_a_literal_true_started_flag_offers_a_resume(self, not_true):
        """RED BEFORE: any truthy value offered a resume for a plan that never ran.

        This key decides whether a restored plan is OFFERED, and the string
        ``"false"`` is non-empty and therefore truthy — so a bare truthiness test
        let a hand-edited or older record smuggle a resume through, on empty
        ledgers, for a plan with no run behind it. Every other field in this record
        is validated strictly; this one was the exception.
        """
        restored = OrchestrationTracker.from_snapshot(
            {
                "stage_rounds": {},
                "stage_escalations": {},
                "stage_results": {},
                "started": not_true,
            }
        )

        assert restored.started is False

    def test_a_literal_true_started_flag_is_honoured(self):
        """Preservation: the real writer's value still restores the fact."""
        restored = OrchestrationTracker.from_snapshot(
            {"stage_rounds": {}, "stage_escalations": {}, "stage_results": {}, "started": True}
        )

        assert restored.started is True

    def test_a_second_restart_keeps_the_started_fact(self):
        """A second restart must not lose what the first one preserved.

        The flag is re-derived from the record on every load, so a plan interrupted
        during stage 1 twice running has to survive both hops -- otherwise the
        second restart reintroduces exactly the bug the first one fixed.
        """
        live = OrchestrationTracker()
        live.record_round(1)
        first = OrchestrationTracker.from_snapshot(live.snapshot())

        second = OrchestrationTracker.from_snapshot(first.snapshot())

        assert first.started is True
        assert second.started is True

    def test_stage_results_survive_restore(self):
        live = OrchestrationTracker()
        live.record_stage_result(1, "/tmp/one.md")

        restored = OrchestrationTracker.from_snapshot(live.snapshot())

        assert restored._stage_results == {1: "/tmp/one.md"}

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"stage_rounds": "not-a-dict"},
            {"stage_results": {"abc": "/tmp/x.md"}},
            {"stage_results": {"0": "/tmp/x.md"}},
            {"stage_rounds": {"1": "many"}},
        ],
    )
    def test_malformed_snapshot_does_not_raise(self, payload):
        """The record is a file on disk: a hand-edit must degrade, not crash."""
        restored = OrchestrationTracker.from_snapshot(payload)

        assert restored.resume_stage() == 1

    @pytest.mark.parametrize(
        "bad_value",
        [None, 0, 1, "", "   ", True, False, [], {}, {"path": "/tmp/x.md"}, 1.5],
    )
    def test_an_unusable_result_value_does_not_mark_a_stage_complete(self, bad_value):
        """RED BEFORE: any value at all counted as a finished stage.

        ``resume_stage()`` reads the mere PRESENCE of a stage key as "this stage
        produced a result", and the restore used to coerce values with ``str``,
        which accepts everything -- ``None`` became ``"None"``, an empty string
        stayed a key. A corrupted or hand-edited record therefore made a resumed
        plan step straight over a stage that had never run, losing that work with
        no trace. Rejecting the entry re-runs the stage instead, which is the safe
        direction.
        """
        restored = OrchestrationTracker.from_snapshot({"stage_results": {"1": bad_value}})

        assert restored.resume_stage() == 1
        assert 1 not in restored._stage_results

    def test_a_valid_result_beyond_an_unusable_one_is_not_credited_forward(self):
        """The gap is what bounds the resume, so stage 1 is still re-run."""
        restored = OrchestrationTracker.from_snapshot(
            {"stage_results": {"1": None, "2": "/tmp/two.md"}}
        )

        assert restored.resume_stage() == 1

    @pytest.mark.parametrize("bad_count", [-1, "3", None, True, 2.0, [], {}])
    def test_an_unusable_counter_is_dropped(self, bad_count):
        """A negative or non-integer count would corrupt the cap arithmetic.

        ``bool`` is rejected even though it is an ``int`` subclass: a JSON ``true``
        here is a malformed record, not a count of one.
        """
        restored = OrchestrationTracker.from_snapshot(
            {"stage_results": {"1": "/tmp/one.md"}, "stage_escalations": {"1": bad_count}}
        )

        assert restored._stage_escalations == {}
        assert restored.is_force_failed(1) is False

    def test_a_zero_counter_is_kept(self):
        """Preservation: 0 is a legitimate count, not a rejected value."""
        restored = OrchestrationTracker.from_snapshot({"stage_escalations": {"1": 0}})

        assert restored._stage_escalations == {1: 0}


# ── What the save writes ─────────────────────────────────────────────────────


class TestPlanStateIsPersisted:
    def test_plan_is_a_slot_owned_metadata_key(self):
        """Absence must mean CLEARED, or a finished plan is re-offered forever."""
        assert "plan" in SLOT_OWNED_META_KEYS

    def test_save_writes_the_plan_record(self, tmp_path, monkeypatch):
        """RED BEFORE: chat_persistence serialised no plan state at all."""
        _patch_config_dir(monkeypatch, tmp_path)
        state = _make_state(tmp_path)
        slot = _plan_slot(state, tmp_path)

        _save_slot_to_history(state, slot, closed=False)

        meta = state.conversation_log._read_metadata("dashboard:plan-slot")
        plan = meta.get("plan")
        assert plan, "the plan record was not persisted"
        assert plan["goal"] == "Ship the thing"
        assert plan["stage_titles"] == ["Recon", "Implement", "Verify"]
        assert plan["stage_descriptions"][0] == ["- do recon"]
        assert plan["tracker"]["stage_results"] == {
            "1": _stage_result_path(tmp_path, "plan-slot", 1)
        }
        # No ``auto_run``: the restore never re-arms it and the offer draws no
        # distinction between Go and Go All, so a persisted flag would have no
        # reader. Asserted rather than merely omitted, so re-adding it fails here.
        assert "auto_run" not in plan

    def test_non_orchestrator_slot_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("plain")
        slot.append("user", "hi")
        slot.drain()

        _save_slot_to_history(state, slot, closed=False)

        assert "plan" not in state.conversation_log._read_metadata("dashboard:plain")

    def test_completed_plan_is_not_persisted(self, tmp_path, monkeypatch):
        """Every stage has a result: re-entering the loop would re-emit the summary."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = _plan_slot(state, tmp_path, titles=("One", "Two"), results=(1, 2), rounds=(1, 2))

        assert _plan_state_for_save(slot) == {}

    def test_cancelled_plan_is_not_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = _plan_slot(state, tmp_path)
        slot._plan_cancelled = True

        assert _plan_state_for_save(slot) == {}

    def test_stopped_tracker_is_not_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = _plan_slot(state, tmp_path)
        slot._orch_tracker.stop()

        assert _plan_state_for_save(slot) == {}

    def test_armed_but_unstarted_plan_still_persists_its_titles(self, tmp_path, monkeypatch):
        """The transcript's own Go buttons need the stage list to mean anything."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("armed")
        slot.mode = "orchestrator"
        slot._stage_titles = ["Only"]
        slot._orch_tracker = None

        record = _plan_state_for_save(slot)

        assert record["stage_titles"] == ["Only"]
        assert record["tracker"] == {}

    def test_a_later_save_clears_a_finished_plan(self, tmp_path, monkeypatch):
        """The record must not outlive the plan it describes."""
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        slot = _plan_slot(state, tmp_path, titles=("One", "Two"), results=(1,), rounds=(1, 2))
        _save_slot_to_history(state, slot, closed=False)
        assert state.conversation_log._read_metadata("dashboard:plan-slot").get("plan")

        slot._orch_tracker.record_stage_result(2, _stage_result_path(tmp_path, "plan-slot", 2))
        slot.append("assistant", "done")
        slot.drain()
        _save_slot_to_history(state, slot, closed=False)

        assert "plan" not in state.conversation_log._read_metadata("dashboard:plan-slot")


# ── What the restore reads back ──────────────────────────────────────────────


class TestPlanStateIsRestored:
    def test_rehydrate_restores_the_execution_pointer(self, tmp_path, monkeypatch):
        """RED BEFORE: a restarted gateway came back with no plan at all."""
        _patch_config_dir(monkeypatch, tmp_path)
        state = _make_state(tmp_path)
        _plan_slot(state, tmp_path)
        _save_slot_to_history(state, state._slots["plan-slot"], closed=False)
        del state._slots["plan-slot"]

        restored = _rehydrate_slot_from_history(state, "plan-slot")

        assert restored is not None
        assert restored._stage_titles == ["Recon", "Implement", "Verify"]
        assert restored._plan_goal == "Ship the thing"
        assert restored._stage_descriptions[1] == ["- do implement"]
        assert restored._orch_tracker is not None
        assert restored._orch_tracker.resume_stage() == 2

    def test_bulk_restore_restores_the_execution_pointer(self, tmp_path, monkeypatch):
        """The recent-sessions path is a second, independent reader."""
        _patch_config_dir(monkeypatch, tmp_path)
        state = _make_state(tmp_path)
        _plan_slot(state, tmp_path)
        _save_slot_to_history(state, state._slots["plan-slot"], closed=False)
        del state._slots["plan-slot"]

        restore_recent_sessions(state, window_minutes=10_000)

        restored = state._slots.get("plan-slot")
        assert restored is not None
        assert restored._orch_tracker is not None
        assert restored._orch_tracker.resume_stage() == 2

    def test_restore_does_not_rearm_auto_run(self, tmp_path, monkeypatch):
        """A restart must not silently resume unattended execution."""
        _patch_config_dir(monkeypatch, tmp_path)
        state = _make_state(tmp_path)
        _plan_slot(state, tmp_path, auto_run=True)
        _save_slot_to_history(state, state._slots["plan-slot"], closed=False)
        del state._slots["plan-slot"]

        restored = _rehydrate_slot_from_history(state, "plan-slot")

        assert restored._auto_run is False

    def test_restore_surfaces_a_resume_offer(self, tmp_path, monkeypatch):
        """RED BEFORE: there was no resume path, so nothing was offered."""
        _patch_config_dir(monkeypatch, tmp_path)
        state = _make_state(tmp_path)
        _plan_slot(state, tmp_path)
        _save_slot_to_history(state, state._slots["plan-slot"], closed=False)
        del state._slots["plan-slot"]

        restored = _rehydrate_slot_from_history(state, "plan-slot")

        tail = restored.messages[-1]
        assert _PLAN_RESUME_MARKER in tail["content"]
        # Names the stage that did NOT finish, not the one after it.
        assert "Stage 2: Implement" in tail["content"]
        # The same control the stage gates use, so Go re-enters the stage loop.
        assert "[OPTION: Go | Go All | Cancel]" in tail["content"]
        assert restored._dirty is True

    def test_resume_offer_is_not_stacked_twice(self, tmp_path, monkeypatch):
        """A second restart on the same unfinished plan must not re-offer."""
        _patch_config_dir(monkeypatch, tmp_path)
        state = _make_state(tmp_path)
        _plan_slot(state, tmp_path)
        _save_slot_to_history(state, state._slots["plan-slot"], closed=False)
        del state._slots["plan-slot"]
        first = _rehydrate_slot_from_history(state, "plan-slot")
        _save_slot_to_history(state, first, closed=False)
        del state._slots["plan-slot"]

        second = _rehydrate_slot_from_history(state, "plan-slot")

        offers = [m for m in second.messages if _PLAN_RESUME_MARKER in m.get("content", "")]
        assert len(offers) == 1

    def test_a_force_failed_stage_keeps_its_ledger_through_the_restore(self):
        """RED BEFORE: the escalation ledger was dropped, laundering the cap.

        ``_restore_plan_state`` only publishes the tracker once the plan counts as
        started, so a force-failed stage whose rounds were zeroed and dropped left
        ``slot._orch_tracker`` unset. ``_stage_loop`` then bootstrapped a clean
        tracker and the force-fail gate this PR adds could never fire — the plan
        re-ran a stage that had exhausted its escalations.
        """
        slot = _ChatSlot("force-failed-restore", mode="orchestrator")

        resume = _restore_plan_state(
            slot,
            {
                "stage_titles": ["One", "Two"],
                "goal": "g",
                "tracker": {
                    "stage_rounds": {"1": 0},
                    "stage_escalations": {"1": 2},
                    "stage_results": {},
                },
            },
        )

        assert resume == 1, "the exhausted stage must be the one offered, not skipped"
        assert slot._orch_tracker is not None, "the tracker was discarded, clearing the cap"
        assert slot._orch_tracker.is_force_failed(1) is True

    def test_a_first_stage_restart_is_offered_for_resume(self, tmp_path, monkeypatch):
        """RED BEFORE: a restart during stage 1 silently dropped the whole plan.

        Nothing had completed, so the tracker restored with empty ledgers, read as
        never-started, and ``_restore_plan_state`` returned None -- no tracker
        published and no resume offer, for the single most likely restart there is.
        """
        _patch_config_dir(monkeypatch, tmp_path)
        slot = _ChatSlot("first-stage-restart", mode="orchestrator")

        resume = _restore_plan_state(
            slot,
            {
                "stage_titles": ["Recon", "Implement"],
                "goal": "Ship the thing",
                "tracker": {"stage_rounds": {"1": 1}, "stage_escalations": {}, "stage_results": {}},
            },
        )

        assert resume == 1, "the interrupted first stage is the one to resume from"
        assert slot._orch_tracker is not None, "the plan was discarded"

        _append_plan_resume_offer(slot, resume)
        assert "Stage 1: Recon" in slot.messages[-1]["content"]

    def test_armed_but_unstarted_plan_gets_no_resume_offer(self):
        """Nothing was interrupted -- the plan message's own buttons still stand."""
        slot = _ChatSlot("unstarted", mode="orchestrator")

        resume = _restore_plan_state(slot, {"stage_titles": ["Only"], "goal": "g", "tracker": {}})

        assert resume is None
        assert slot._stage_titles == ["Only"]

    @pytest.mark.parametrize(
        "hostile",
        [
            "/etc/passwd",
            "/etc/shadow",
            "~/.ssh/id_rsa",
            "../../../../etc/passwd",
            "stage_1_result.md",
            "/tmp/stage_1_result.md",
            "",
            None,
            0,
        ],
    )
    def test_a_result_path_outside_the_slots_own_directory_is_dropped(
        self, tmp_path, monkeypatch, hostile
    ):
        """RED BEFORE: any non-empty string was accepted as a result path.

        The restored path is opened by ``_read_previous_results`` and its contents
        are inlined into the NEXT stage's prompt, so accepting a free-form string
        let the transcript's own metadata line name any readable file on the host
        and hand its bytes to the model. The history JSONL is a plain file, so a
        hand-edit reaches this.
        """
        _patch_config_dir(monkeypatch, tmp_path)

        confined = _confine_restored_result_paths(
            {"stage_rounds": {"1": 1}, "stage_results": {"1": hostile}}, "hostile-path-slot"
        )

        assert confined["stage_results"] == {}
        # And the rest of the record is carried through untouched.
        assert confined["stage_rounds"] == {"1": 1}

    def test_the_writers_own_path_is_kept(self, tmp_path, monkeypatch):
        """Preservation: the real path a completed stage recorded still restores."""
        _patch_config_dir(monkeypatch, tmp_path)
        slot = _ChatSlot("kept-path-slot", mode="orchestrator")
        real = _stage_result_path(tmp_path, "kept-path-slot", 1)

        resume = _restore_plan_state(
            slot,
            {
                "stage_titles": ["One", "Two"],
                "tracker": {"stage_rounds": {"1": 1}, "stage_results": {"1": real}},
            },
        )

        assert slot._orch_tracker._stage_results == {1: real}
        assert resume == 2

    def test_a_path_belonging_to_another_slot_is_dropped(self, tmp_path, monkeypatch):
        """The allowlist is per-slot, not merely 'somewhere under sessions/'."""
        _patch_config_dir(monkeypatch, tmp_path)

        confined = _confine_restored_result_paths(
            {"stage_results": {"1": _stage_result_path(tmp_path, "someone-else", 1)}}, "mine"
        )

        assert confined["stage_results"] == {}

    def test_a_hostile_path_is_dropped_from_a_published_tracker(self, tmp_path, monkeypatch):
        """The end-to-end shape, with a ledger that survives to be inspected.

        An escalation keeps the plan ``started`` after the rejection, so the
        published tracker can be asserted on directly — with the path gone, the
        stage is re-run rather than skipped, and the force-fail gate still holds.
        """
        _patch_config_dir(monkeypatch, tmp_path)
        slot = _ChatSlot("published-hostile", mode="orchestrator")

        resume = _restore_plan_state(
            slot,
            {
                "stage_titles": ["One", "Two"],
                "tracker": {
                    "stage_rounds": {"1": 1},
                    "stage_escalations": {"1": 2},
                    "stage_results": {"1": "/etc/passwd"},
                },
            },
        )

        assert slot._orch_tracker is not None
        assert slot._orch_tracker._stage_results == {}
        assert resume == 1
        assert slot._orch_tracker.is_force_failed(1) is True

    def test_the_resume_offer_is_redacted(self, tmp_path, monkeypatch):
        """RED BEFORE: the one new orchestrator row could print a credential.

        Stage titles are model-authored and round-trip through the metadata line,
        which the load path does not redact (meta is deferred to its emit sites).
        Every other row the orchestrator writes redacts at its emit site; this one
        did not.
        """
        monkeypatch.setattr("kiro_crew.dashboard.chat_persistence.config_dir", lambda: tmp_path)
        slot = _ChatSlot("redact-offer-slot", mode="orchestrator")
        _restore_plan_state(
            slot,
            {
                "stage_titles": ["Ship it with AKIAIOSFODNN7EXAMPLE", "Two"],
                "tracker": {"stage_rounds": {"1": 1}},
            },
        )

        _append_plan_resume_offer(slot, 1)

        assert "AKIAIOSFODNN7EXAMPLE" not in slot.messages[-1]["content"]
        assert _PLAN_RESUME_MARKER in slot.messages[-1]["content"]

    @pytest.mark.parametrize("payload", [None, {}, {"stage_titles": []}, "nonsense"])
    def test_absent_or_malformed_record_restores_nothing(self, payload):
        slot = _ChatSlot("empty", mode="orchestrator")

        assert _restore_plan_state(slot, payload) is None
        assert slot._orch_tracker is None
