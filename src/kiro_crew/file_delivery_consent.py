"""Explicit owner consent before Kiro Crew delivers a file whose contents the
credential scanner flags.

An agent can legitimately generate secret material the owner needs delivered --
a VPN device private key inside a compose stack the owner will deploy on another
machine is the reported case. Every delivery surface refuses it today, and the
refusal is CORRECT rather than over-eager: that file matches the PEM private key
branch of ``security._CREDENTIAL_PATTERNS``, the highest-confidence detector in
the catalogue. The detector is right, so no amount of tuning is the remedy --
what is missing is a way for the owner to say "yes, that is mine, hand it over."

Selecting the destination class IS the consent point, not the delivery
-----------------------------------------------------------------------
The delivery cannot be the confirmation point, for the same reason
:mod:`kiro_crew.aws_consent` gives for a paid AWS call: ``file_send`` fires from
surfaces with nobody watching. A cron job exports a report, a subagent hands back
an artifact, a Slack thread reply attaches a file. A per-invocation "Deliver /
Cancel" card has no one to answer it there, and "no confirmation available means
no delivery" would leave the feature exactly as broken as the hard wall it
replaced.

There is a second, sharper reason here. The same scanner rule is enforced at FOUR
independent points, and only one of them is the tool call:

* ``mcp_tools.messaging.file_send``      -- the MCP tool, before any byte is copied
* ``dashboard.handlers.files``           -- ``POST /api/outbox/notify``
* ``dashboard.handlers.files``           -- ``GET  /api/outbox/{filename}``
* ``dashboard.handlers.files._gate_upload_file`` -- shared by the Slack and
  channel upload legs

A card shown at the tool call can only speak for the first. The other three
re-scan at serve time and know nothing about a click that happened earlier, so a
per-invocation grant would report "delivered", render a card, and then refuse the
download -- worse than today's clean refusal. A durable record is readable at
every gate, which is why the grant is configuration-time and lives on disk.

Which destinations a grant can EVER cover
-----------------------------------------
``GRANTABLE_CLASSES`` has exactly one member, and that is a security property
rather than a starting point.

* ``owner_dashboard`` -- the outbox file on the owner's own disk, the chat file
  card, and the authenticated ``GET /api/outbox/{filename}`` download. The
  audience is the owner's own machine and their own authenticated browser (no
  entry in any ``dashboard.token_auth`` bypass list reaches that route). An owner
  seeing their own secret is not a leak.

Deliberately absent, and named in :data:`NEVER_GRANTABLE_CLASSES` so a reader can
see the omission is a decision:

* the Slack upload leg -- the one destination with a genuine third-party audience
  AND the one an agent aims by argument, since ``file_send``'s schema exposes an
  optional ``channel`` id. A grant reachable by a tool argument is not owner
  consent; it is agent-chosen disclosure wearing consent's name.
* the channel (Telegram / Discord) upload leg -- the destination comes from the
  caller's session map rather than an argument, but a linked conversation is not
  demonstrably 1:1, and an audience that cannot be proved is a reason to refuse
  rather than to assume.

Both of those legs pass through ``_gate_upload_file``, which exists (by its own
docstring) "so the Slack and channel legs cannot drift apart gate by gate". That
function does not read this module, and nothing in this module can be reached
from it. The guarantee is therefore structural: there is no code path by which a
grant arrives at a third-party destination, so the property cannot be undone by
inverting a check -- only by editing that gate, which is a separate decision.

What this does NOT change
-------------------------
No detector, pattern, or threshold moves. ``security.redact`` and its catalogue
are untouched; a grant changes what a gate DOES with a positive result, never
whether the scanner finds it. Note also that ``security.redact`` runs only the
exfiltration-URL and credential passes -- it does not call ``redact_local_paths``
-- so the scope a grant can affect is credentials and exfil URLs, not "anything
sensitive".

Where the grant lives, and why not ``config.json``
--------------------------------------------------
``file_delivery_consent.json`` sits on the read+write KEYSTONE floor
(``security._CREW_SECRET_LEAVES``), the same placement as
``aws_service_consent.json`` and ``computer_use.json``, and for the same reason:
this is an authorization record, not a preference. ``config.json`` is writable by
any auto-approved agent shell, so a grant stored there could be minted by a
prompt-injected agent -- consenting, on the owner's behalf, to shipping the
owner's secrets. The platform's own ``CredentialPolicy.exempt_exact_hosts``
docstring states the rule this file obeys: such a set is "NEVER sourced from
``config.json`` -- an agent-writable exemption would be a hole in the redaction
ceiling."

The authenticated, OWNER-gated dashboard handler opens the path directly and is
the only writer. There is deliberately no CLI verb: a terminal command that
records a grant on request is a grant an automated caller can take, and its guard
would have to key on an env var an in-process agent can unset.

Known limit, stated rather than papered over
--------------------------------------------
A grant is durable and coarse. Once ``owner_dashboard`` is confirmed, every later
flagged file reaches the owner's dashboard without asking again -- that is the
point (an unattended cron must be able to deliver), and it is also the cost. The
grant does not distinguish one secret from another, so an agent that generates a
credential the owner did NOT ask for will also be able to put it in the owner's
outbox. What that buys an attacker is bounded by the audience: the file lands on
the owner's own disk and in the owner's own authenticated browser, which is where
the agent could already write it with ordinary file tools. Every delivery under a
grant is SEL-audited as ``sensitive_content_delivered_with_consent`` so the
record exists even though the refusal does not.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from kiro_crew import platform_compat
from kiro_crew.atomic_write import atomic_write
from kiro_crew.config.loader import file_delivery_consent_path

logger = logging.getLogger(__name__)

#: The one destination class a grant can cover: the owner's own disk plus their
#: own authenticated dashboard (outbox file, chat file card, download route).
#: The id is the stored grant key, so renaming it invalidates existing grants
#: (fail-closed: the owner is asked again) rather than silently authorizing a
#: different destination.
CLASS_OWNER_DASHBOARD = "owner_dashboard"

#: Destination classes a grant may EVER cover. Exactly one member, deliberately.
GRANTABLE_CLASSES: frozenset[str] = frozenset({CLASS_OWNER_DASHBOARD})

#: Destination classes that must never appear in :data:`GRANTABLE_CLASSES`,
#: recorded so the omission reads as a decision rather than an oversight. Both
#: route through ``dashboard.handlers.files._gate_upload_file``, which does not
#: read this module; these ids exist for documentation and for the ratchet test
#: that asserts the two sets stay disjoint.
NEVER_GRANTABLE_CLASSES: frozenset[str] = frozenset({"slack_upload", "channel_upload"})

#: Human-facing labels for the confirmation surface and the log lines.
CLASS_LABELS: dict[str, str] = {
    CLASS_OWNER_DASHBOARD: "This machine's outbox and my own dashboard",
}

#: Lock filename beside the consent file -- NOT the file itself, because
#: ``atomic_write`` renames a new inode over it and a lock on the old inode
#: protects nothing. Same placement and reasoning as ``aws_consent._ConsentLock``.
_LOCK_FILENAME = ".file_delivery_consent.lock"


@dataclass(frozen=True)
class Grant:
    """A recorded consent to deliver scanner-flagged files to one destination class."""

    destination_class: str
    granted_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination_class": self.destination_class,
            "granted_at": self.granted_at,
        }


def _read_all() -> dict[str, Any]:
    """The whole store, or ``{}`` when it is missing or unreadable.

    Failing soft is the right READ behaviour -- an authorization record that
    cannot be parsed is not an authorization, so every gate keeps refusing. See
    :func:`_preserve_if_unreadable` for what happens before a write, where
    failing soft would otherwise discard the unreadable bytes.
    """
    try:
        raw = json.loads(file_delivery_consent_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "file-delivery consent store is unreadable; treating every destination as unconfirmed"
        )
        return {}
    return raw if isinstance(raw, dict) else {}


def _preserve_if_unreadable() -> None:
    """Copy an unreadable store aside before a write replaces it.

    ``_read_all`` fails soft to ``{}``, so a write built on it would replace an
    unparseable file wholesale and the old bytes would be gone. What is lost is
    not a working authorization -- an unreadable store already grants nothing --
    but discarding it silently is not this function's call to make. Preserved
    rather than refused: refusing would leave an owner with a corrupt file unable
    to re-confirm from the dashboard at all. Mirrors
    ``aws_consent._preserve_if_unreadable``.
    """
    path = file_delivery_consent_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "could not read the file-delivery consent store to preserve it", exc_info=True
        )
        return
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return  # Readable; the write is a normal read-modify-write.
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    sidecar = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        # restrict_to_owner=True locks the temp file down BEFORE the preserved
        # contents reach it and implies the owner-only POSIX mode. The default
        # restrict_on_error="raise" surfaces a lockdown failure into this except,
        # where the whole preservation attempt is already warn-only -- and
        # because the failure happens before the rename, a sidecar that could not
        # be protected never exists at the final path at all.
        atomic_write(sidecar, raw, restrict_to_owner=True)
        logger.warning(
            "file-delivery consent store was unreadable; preserved the previous contents at %s "
            "before recording a new confirmation",
            sidecar.name,
        )
    except OSError:
        logger.warning(
            "could not preserve the unreadable file-delivery consent store", exc_info=True
        )


class _ConsentLock:
    """Exclusive lock around a read-modify-write of the consent file.

    Every writer here is a read-modify-write over a file ``atomic_write``
    REPLACES wholesale, so two concurrent writes would each apply their change to
    a stale snapshot and the later one would silently drop the other. On an
    authorization record that is a correctness defect, not a lost-update
    annoyance.
    """

    def __init__(self) -> None:
        self._fd: int | None = None

    def __enter__(self) -> "_ConsentLock":
        lock_file = file_delivery_consent_path().parent / _LOCK_FILENAME
        self._fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o600)
        platform_compat.acquire_lock(self._fd, exclusive=True)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None:
            try:
                platform_compat.release_lock(self._fd)
            finally:
                os.close(self._fd)
                self._fd = None


def _write_all(data: dict[str, Any]) -> None:
    # Fail-loud lockdown BEFORE any content lands, same as the sibling keystone
    # stores: restrict_to_owner=True applies the owner-only DACL to the temp file
    # before the payload reaches it and implies the owner-only POSIX mode. The
    # default restrict_on_error="raise" refuses to write a record it cannot
    # protect. Every failure inside atomic_write happens before the final path is
    # touched, so no cleanup is needed here and an unlink would instead delete the
    # previous, healthy, already-locked-down store on a transient failure.
    atomic_write(
        file_delivery_consent_path(),
        json.dumps(data, indent=2, sort_keys=True),
        restrict_to_owner=True,
    )


def read_grant(destination_class: str) -> Grant | None:
    """The stored grant for ``destination_class``, or ``None`` when there is none.

    Fails soft to ``None`` (no consent) on a missing, unreadable, or malformed
    file: an authorization record that cannot be read is not an authorization.
    """
    row = _read_all().get(destination_class)
    if not isinstance(row, dict):
        return None
    stored = str(row.get("destination_class", ""))
    # A row filed under one key but naming another destination is not a grant for
    # either: refuse rather than trust the key, so a hand-edited or partially
    # written store cannot widen a grant by disagreeing with itself.
    if stored != destination_class:
        logger.warning(
            "file-delivery consent record under %r names %r; treating as absent",
            destination_class,
            stored,
        )
        return None
    return Grant(destination_class=stored, granted_at=str(row.get("granted_at", "")))


def is_granted(destination_class: str) -> bool:
    """Whether the owner has confirmed delivery to ``destination_class``.

    Fail-closed on every unexpected input: a class outside
    :data:`GRANTABLE_CLASSES` is refused before the store is even read, so a
    caller cannot consult this module about a third-party destination and get a
    True. LOCAL only -- no network, no probe.
    """
    if destination_class not in GRANTABLE_CLASSES:
        return False
    return read_grant(destination_class) is not None


def record_grant(destination_class: str, *, granted_at: str) -> Grant:
    """Persist the owner's consent for ``destination_class``."""
    if destination_class not in GRANTABLE_CLASSES:
        raise ValueError(f"destination class {destination_class!r} can never be granted")
    grant = Grant(destination_class=destination_class, granted_at=granted_at)
    with _ConsentLock():
        # Inside the lock, before the read: a concurrent writer must not be able
        # to slip between preserving the old bytes and replacing them.
        _preserve_if_unreadable()
        data = _read_all()
        data[destination_class] = grant.to_dict()
        _write_all(data)
    audit_decision(destination_class, outcome="granted")
    return grant


def revoke(destination_class: str) -> bool:
    """Drop consent for ``destination_class``. True when a grant was removed."""
    with _ConsentLock():
        data = _read_all()
        if destination_class not in data:
            return False
        del data[destination_class]
        _write_all(data)
    audit_decision(destination_class, outcome="revoked")
    return True


def audit_decision(destination_class: str, *, outcome: str, detail: str = "") -> None:
    """Record a consent state change, a denial, or a consented delivery in the SEL.

    Grants, revocations, denials AND deliveries made under a grant are recorded.
    The delivery entry is the point: the refusal it replaces was self-evident in
    the tool's error string, whereas a successful consented delivery would
    otherwise leave no trace that a flagged file left the gate at all. Every
    entry answers a question an incident review actually asks -- who authorized
    delivery, when was it withdrawn, and which flagged files went out under it.

    Never raises: an audit failure must not be what stops a refusal from being
    enforced. Imported lazily because this module is reached from the MCP stdio
    servers, whose stray writes would corrupt the JSON-RPC stream, and because
    the security-event layer pulls the redaction stack the read path never needs.
    """
    try:
        from kiro_crew.sel import sel

        sel().log_api_access(
            caller="owner" if outcome in ("granted", "revoked") else "gateway",
            operation=f"file_delivery_consent.{outcome}",
            outcome=outcome,
            source="file-delivery-consent",
            resources=(f"{destination_class}: {detail[:200]}" if detail else destination_class),
        )
    except Exception:  # pragma: no cover - audit must never break the gate
        logger.debug("could not write the file-delivery consent audit event", exc_info=True)
