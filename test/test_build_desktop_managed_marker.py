"""Regression guard for the baked ``EXTERNALLY-MANAGED`` marker step in
``packaging/build-desktop.sh``.

Background
----------
An edition whose installs are owned by an external package manager (a Toolbox,
a corporate installer) names its marker through ``KIROCREW_MANAGED_INSTALL_MARKER``.
The script copies it to ``website/electron/EXTERNALLY-MANAGED`` so electron-builder
packs it INTO ``app.asar`` next to ``main.js``, where ``readExternallyManaged``
(``website/electron/auto-update.js``) trusts it as the application's own code on
every platform -- unlike a marker dropped beside the app after the build, which
is gated on file provenance and refused on every user-owned install.

Why this test exists
--------------------
The reader treats a malformed marker as "managed, nothing to run": the updater
is OFF and the About panel offers no command. So a typo in an edition's marker
would ship an app that can neither self-update nor be updated from the UI, and
nothing in the running app would say why. The build step is where that must
fail loudly. These tests extract the step from the shipped script (not a copy,
so a revert is what runs here) and drive it through the accepted shape and each
rejected one, plus the "unset removes a stale leftover" contract that keeps a
previous local build's declaration from riding along.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="build-desktop.sh is a bash script"),
    pytest.mark.skipif(shutil.which("node") is None, reason="the step validates the marker with node"),
]

SCRIPT = Path(__file__).parent.parent / "packaging" / "build-desktop.sh"


def _extract_step() -> str:
    """Pull step 3b out of the shipped script, up to the step-4 header."""
    text = SCRIPT.read_text()
    m = re.search(
        r"(# --- 3b\. Baked EXTERNALLY-MANAGED marker.*?)\n# --- 4\. Package the desktop app",
        text,
        re.DOTALL,
    )
    assert m, "step 3b (baked EXTERNALLY-MANAGED marker) not found in packaging/build-desktop.sh"
    return m.group(1)


def _run(tmp_path: Path, marker_env: str | None) -> tuple[subprocess.CompletedProcess, Path]:
    electron_dir = tmp_path / "electron"
    electron_dir.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "KIROCREW_MANAGED_INSTALL_MARKER"}
    if marker_env is not None:
        env["KIROCREW_MANAGED_INSTALL_MARKER"] = marker_env
    env["ELECTRON_DIR"] = str(electron_dir)
    # The same strict mode the real script runs under, and its `log` helper.
    script = "set -euo pipefail\nlog() { echo \"$@\"; }\n" + _extract_step() + "\n"
    proc = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    return proc, electron_dir / "EXTERNALLY-MANAGED"


VALID = {
    "managedBy": "Builder Toolbox",
    "updateCommand": "/opt/toolbox/bin/toolbox update kirocrew",
    "checkCommand": "/opt/toolbox/bin/kirocrew-update-check",
}


def test_valid_marker_is_copied_verbatim(tmp_path: Path) -> None:
    src = tmp_path / "marker.json"
    src.write_text(json.dumps(VALID))
    proc, dest = _run(tmp_path, str(src))
    assert proc.returncode == 0, proc.stderr
    assert dest.read_text() == src.read_text()
    assert "Baking EXTERNALLY-MANAGED marker" in proc.stdout


def test_unset_removes_a_stale_leftover(tmp_path: Path) -> None:
    # A previous local build with the variable set must not leak its
    # declaration into a build without it.
    electron_dir = tmp_path / "electron"
    electron_dir.mkdir()
    (electron_dir / "EXTERNALLY-MANAGED").write_text(json.dumps(VALID))
    proc, dest = _run(tmp_path, None)
    assert proc.returncode == 0, proc.stderr
    assert not dest.exists()


def test_empty_value_means_unset(tmp_path: Path) -> None:
    proc, dest = _run(tmp_path, "")
    assert proc.returncode == 0, proc.stderr
    assert not dest.exists()


@pytest.mark.parametrize(
    "body, reason",
    [
        ("not json", "not JSON"),
        (json.dumps([1, 2]), "must be a JSON object"),
        (json.dumps({**VALID, "extra": "x"}), "unknown field"),
        (json.dumps({**VALID, "updateCommand": 7}), "must be a string"),
        (json.dumps({"managedBy": "x"}), "no updateCommand"),
        (json.dumps({**VALID, "managedBy": "x" * 9000}), "caps at 8192"),
    ],
    ids=["not-json", "array", "unknown-field", "non-string", "no-updateCommand", "over-cap"],
)
def test_malformed_marker_fails_the_build(tmp_path: Path, body: str, reason: str) -> None:
    # Each rejection names its cause: a lane operator reading the build log must
    # not have to diff a JSON file against the reader to find the typo.
    src = tmp_path / "marker.json"
    src.write_text(body)
    proc, dest = _run(tmp_path, str(src))
    assert proc.returncode != 0
    assert reason in proc.stderr, proc.stderr
    assert "KIROCREW_MANAGED_INSTALL_MARKER rejected" in proc.stderr
    assert not dest.exists(), "a rejected marker must not be placed"


def test_missing_file_fails_the_build(tmp_path: Path) -> None:
    proc, dest = _run(tmp_path, str(tmp_path / "nope.json"))
    assert proc.returncode != 0
    assert "does not name a file" in proc.stderr
    assert not dest.exists()


def test_package_json_packs_the_marker_into_the_asar() -> None:
    # The copy is pointless unless electron-builder ships it: build.files is an
    # explicit allowlist, and readExternallyManaged looks next to main.js.
    pkg = json.loads((SCRIPT.parent.parent / "website" / "electron" / "package.json").read_text())
    assert "EXTERNALLY-MANAGED" in pkg["build"]["files"]
    assert "main.js" in pkg["build"]["files"]
