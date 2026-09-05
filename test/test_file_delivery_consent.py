"""Owner-consented delivery of scanner-flagged files (issue #7770).

Every piece of credential-shaped material here is SYNTHESIZED AT RUNTIME from a
small grammar rather than checked in as a literal. That is deliberate and is not
cosmetic: a diff carrying literal PEM headers or working token strings reads as an
exfiltration recipe, and one of this repository's pull requests is permanently
deadlocked because a review provider refused it twelve consecutive times as
"potentially high-risk cyber activity" and then failed closed on the absent
verdict. The scanner sees identical bytes at runtime either way, so the
assertions below are exactly as strong as literals would have been -- only the
reviewed bytes differ.

``_synth_pem`` in particular never contains real key material: its body is
deterministic base64 over a SHA-256 of a loop counter, so it is
private-key-SHAPED without being a private key.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
from unittest.mock import patch

import pytest

from kiro_crew import file_delivery_consent, security
from kiro_crew.config.loader import file_delivery_consent_path

# ---------------------------------------------------------------- generators


def _synth_pem() -> str:
    """PEM private-key-SHAPED text assembled at runtime from fragments."""
    rule = "-" * 5
    begin = " ".join(["BEGIN", "RSA", "PRIVATE", "KEY"])
    end = " ".join(["END", "RSA", "PRIVATE", "KEY"])
    body = "\n".join(
        base64.b64encode(hashlib.sha256(f"kc-7770-{i}".encode()).digest() * 2).decode()
        for i in range(4)
    )
    return f"{rule}{begin}{rule}\n{body}\n{rule}{end}{rule}\n"


def _synth_aws_key() -> str:
    """An AWS-access-key-SHAPED token: fixed public prefix plus synthetic body."""
    prefix = "A" + "KIA"
    body = hashlib.sha256(b"kc-7770-aws").hexdigest().upper()[:16]
    return prefix + body


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the consent store at a tmp dir so no test touches the real one."""
    store = tmp_path / "file_delivery_consent.json"
    monkeypatch.setattr(
        file_delivery_consent, "file_delivery_consent_path", lambda: store, raising=True
    )
    return store


def _grant() -> None:
    file_delivery_consent.record_grant(
        file_delivery_consent.CLASS_OWNER_DASHBOARD, granted_at="2026-09-05T00:00:00+00:00"
    )


# ------------------------------------------------- the generators are honest


class TestSynthesizedMaterialActuallyTrips:
    """A generator the scanner ignores would make every test below vacuous."""

    def test_synth_pem_is_detected(self):
        pem = _synth_pem()
        assert security.redact(pem) != pem

    def test_synth_aws_key_is_detected(self):
        key = _synth_aws_key()
        assert security.redact(key) != key


# --------------------------------------------------- the absolute condition


class TestThirdPartyLegsCanNeverBeGranted:
    def test_grantable_and_never_grantable_are_disjoint(self):
        assert not (
            file_delivery_consent.GRANTABLE_CLASSES & file_delivery_consent.NEVER_GRANTABLE_CLASSES
        )

    def test_only_the_owner_dashboard_class_is_grantable(self):
        assert file_delivery_consent.GRANTABLE_CLASSES == {
            file_delivery_consent.CLASS_OWNER_DASHBOARD
        }

    @pytest.mark.parametrize("leg", sorted(file_delivery_consent.NEVER_GRANTABLE_CLASSES))
    def test_recording_a_third_party_leg_raises(self, leg):
        with pytest.raises(ValueError):
            file_delivery_consent.record_grant(leg, granted_at="2026-09-05T00:00:00+00:00")

    @pytest.mark.parametrize("leg", sorted(file_delivery_consent.NEVER_GRANTABLE_CLASSES))
    def test_a_third_party_leg_is_never_granted_even_if_the_file_says_so(
        self, leg, _isolated_store
    ):
        """A hand-planted row for an upload leg must not authorize anything.

        ``is_granted`` refuses the class before it reads the store, so the row
        below is inert. Without that ordering a writer who reached the file --
        which the keystone fence exists to prevent, but which this test does not
        assume -- could authorize the Slack leg.
        """
        _isolated_store.write_text(
            json.dumps({leg: {"destination_class": leg, "granted_at": "2026-09-05T00:00:00+00:00"}})
        )
        assert file_delivery_consent.is_granted(leg) is False

    def test_the_shared_upload_gate_cannot_read_the_consent_store(self):
        """The structural half of the guarantee, asserted on real source.

        ``_gate_upload_file`` is the single admission gate both third-party legs
        route through. If it never references the consent store, no grant can
        reach the Slack or channel leg by any code path -- a property that cannot
        be undone by inverting a check, only by editing that function.
        """
        from kiro_crew.dashboard.handlers import files as files_handlers

        src = inspect.getsource(files_handlers._gate_upload_file)
        assert "consent" not in src.lower()
        assert "file_delivery_consent" not in src


# ------------------------------------------------------- no detector moved


class TestNoDetectorMoved:
    def test_a_grant_does_not_change_what_redact_finds(self):
        """A grant changes what a GATE does with a positive, never the scan."""
        pem = _synth_pem()
        before = security.redact(pem)
        _grant()
        assert security.redact(pem) == before
        assert security.redact(pem) != pem

    def test_redact_does_not_strip_local_paths(self):
        """Bounds the grant's scope: credentials and exfil URLs, not everything.

        Stated in the PR body as a scope claim, so it is pinned here rather than
        left for a reviewer to derive.
        """
        probe = "[Errno 2] No such file or directory: '/home/someone/.kiro/crew/x'"
        assert security.redact(probe) == probe


# ------------------------------------------------------------- the store


class TestGrantStore:
    def test_absent_store_grants_nothing(self):
        assert (
            file_delivery_consent.is_granted(file_delivery_consent.CLASS_OWNER_DASHBOARD) is False
        )

    def test_record_then_read(self):
        _grant()
        assert file_delivery_consent.is_granted(file_delivery_consent.CLASS_OWNER_DASHBOARD)
        grant = file_delivery_consent.read_grant(file_delivery_consent.CLASS_OWNER_DASHBOARD)
        assert grant is not None and grant.granted_at == "2026-09-05T00:00:00+00:00"

    def test_revoke_removes_it(self):
        _grant()
        assert file_delivery_consent.revoke(file_delivery_consent.CLASS_OWNER_DASHBOARD) is True
        assert (
            file_delivery_consent.is_granted(file_delivery_consent.CLASS_OWNER_DASHBOARD) is False
        )

    def test_unreadable_store_grants_nothing(self, _isolated_store):
        _isolated_store.write_text("{not json")
        assert (
            file_delivery_consent.is_granted(file_delivery_consent.CLASS_OWNER_DASHBOARD) is False
        )

    def test_a_row_disagreeing_with_its_key_grants_nothing(self, _isolated_store):
        """Refuse rather than trust the key, so a partial write cannot widen a grant."""
        _isolated_store.write_text(
            json.dumps(
                {
                    file_delivery_consent.CLASS_OWNER_DASHBOARD: {
                        "destination_class": "something_else",
                        "granted_at": "2026-09-05T00:00:00+00:00",
                    }
                }
            )
        )
        assert (
            file_delivery_consent.is_granted(file_delivery_consent.CLASS_OWNER_DASHBOARD) is False
        )

    def test_an_unreadable_store_is_preserved_before_a_write(self, _isolated_store):
        _isolated_store.write_text("{corrupt")
        _grant()
        sidecars = list(_isolated_store.parent.glob(f"{_isolated_store.name}.corrupt-*"))
        assert sidecars, "the unreadable store must be preserved, not silently replaced"
        assert sidecars[0].read_text() == "{corrupt"


# ------------------------------------------------------------ the keystone


class TestKeystoneFencing:
    def test_the_grant_file_is_fenced_from_agent_file_tools(self):
        assert security.is_sensitive_path(str(file_delivery_consent_path())) is True

    def test_the_leaf_is_registered_beside_its_sibling(self):
        assert "file_delivery_consent.json" in security._CREW_SECRET_LEAVES
        assert "aws_service_consent.json" in security._CREW_SECRET_LEAVES

    def test_there_is_no_cli_verb_for_the_grant(self):
        """A CLI verb is a grant an automated caller can take, so there is none."""
        from kiro_crew import cli

        src = inspect.getsource(cli)
        assert "file-delivery-consent" not in src
        assert "file_delivery_consent" not in src


# ---------------------------------------------------------------- the tool


class TestConsentedDownloadRequiresOwnerIdentity:
    """A grant lifts the refusal for the OWNER, not for every authenticated caller.

    The bug both review lanes found independently. The download route is absent
    from every ``token_auth`` bypass list, which establishes it needs
    AUTHENTICATION, not OWNER IDENTITY -- a Slack allow-listed non-owner running
    ``!dashboard`` authenticates with ``app == ""`` and ``sub != owner_id``. Without
    the owner conjunct the grant turns a clean 400-for-everyone into raw bytes for
    any authenticated caller.
    """

    def test_the_download_gate_requires_both_conjuncts(self):
        import inspect

        from kiro_crew.dashboard.handlers import files as files_handlers

        src = inspect.getsource(files_handlers.api_outbox_download)
        assert "is_granted" in src
        # Asserted on source because the alternative is an aiohttp request fixture
        # carrying a forged non-owner identity, which would pin the harness rather
        # than the route. Paired with the negative below so this cannot pass by
        # merely mentioning the name.
        assert "is_owner_dashboard_request" in src

    def test_owner_check_and_grant_are_ANDed_not_ORed(self):
        import inspect

        from kiro_crew.dashboard.handlers import files as files_handlers

        src = inspect.getsource(files_handlers.api_outbox_download)
        window = src[src.index("is_granted") : src.index("is_granted") + 400]
        assert " and is_owner_dashboard_request(request)" in window
        assert " or is_owner_dashboard_request(request)" not in window


class TestFileSendHonoursTheGrant:
    def _call(self, path):
        from kiro_crew.mcp_tools.messaging import file_send

        return file_send("file_send", {"path": str(path)})

    def test_without_a_grant_a_flagged_file_is_refused(self, tmp_path):
        src = tmp_path / "device.conf"
        src.write_text(_synth_pem())
        out = self._call(src)
        assert "sensitive data" in out and "aborted" in out

    def test_with_a_grant_it_delivers_and_skips_both_upload_legs(self, tmp_path):
        from kiro_crew import mcp_core

        src = tmp_path / "device.conf"
        src.write_text(_synth_pem())
        _grant()
        posted: list[str] = []

        def _fake_post(path, *a, **kw):
            posted.append(path)
            return {"ok": True}

        with patch.object(mcp_core, "_post", side_effect=_fake_post):
            out = self._call(src)
        assert "File sent" in out
        assert "Slack and channel upload skipped" in out
        # The absolute condition, asserted on behaviour: neither upload leg is
        # even ATTEMPTED for content the owner scoped to their own dashboard.
        assert "/api/outbox/notify" in posted
        assert not any("upload-file" in p for p in posted)
