"""Fail-closed contract tests for macOS assets on GitHub Releases.

The release job has ``contents: write`` and is the final trust-boundary hop
before files become public.  These tests execute its actual shell step so an
unsigned fallback, a broad ``find | head`` selector, or a superficial presence
check cannot silently reappear.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml

from kiro_crew.subprocess_utf8 import UTF8_TEXT

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="the GitHub Release assembly step runs under bash on ubuntu-latest",
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
STEP_NAME = "Assemble release assets (require gated macOS artifacts)"
CHANNEL = "stable"
VERSION = "1.2.3"
ARTIFACT_NAME = f"KiroCrew-notarized-{CHANNEL}-{VERSION}"
# The default harness models a byte-for-byte PROMOTION (stable with
# ``rebuild != 'true'``): the one stable mode that builds no desktop leg and so
# carries no manifest of its own. That exemption is what lets the DMG/zip
# fail-closed tests below reach the validation code without first tripping the
# symbols-manifest guard. The requirement path -- insider, and an ordinary
# stable REBUILD -- is exercised by the two tests that override ``REBUILD``.
REBUILD = "false"


def _resolve(text: str) -> str:
    resolved = text.replace("${{ needs.version.outputs.channel }}", CHANNEL)
    resolved = resolved.replace("${{ needs.version.outputs.version }}", VERSION)
    resolved = resolved.replace("${{ needs.version.outputs.rebuild }}", REBUILD)
    assert "${{" not in resolved, "test harness left an unresolved GitHub expression"
    return resolved


def _assembly_step() -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["github-release"]["steps"]
    step = next((item for item in steps if item.get("name") == STEP_NAME), None)
    assert step is not None, f"release workflow step {STEP_NAME!r} not found"
    return step


def _assembly_script() -> str:
    return _resolve(_assembly_step()["run"])


def _assembly_env() -> dict[str, str]:
    """The step's own ``env:`` block, which Actions exports before the script runs.

    Read from the workflow rather than hardcoded here: the script runs under
    ``set -u``, so a value the step declares and this harness omits does not
    quietly default -- every test in this file dies with ``unbound variable``
    instead of exercising the fail-closed paths it exists to pin. Reading the
    block keeps the harness faithful to what the runner actually provides, and
    still fails loudly if the step starts reading a variable nobody sets.
    """
    return {
        name: _resolve(str(value)) for name, value in (_assembly_step().get("env") or {}).items()
    }


def _artifact_dir(root: Path, name: str = ARTIFACT_NAME) -> Path:
    path = root / "artifacts" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_valid_zip(path: Path, app_name: str = "KiroCrew.app") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{app_name}/Contents/Info.plist", "<plist/>")


def _write_valid_dmg(path: Path) -> None:
    # hdiutil's UDIF output ends in a 512-byte trailer beginning with "koly".
    path.write_bytes(b"test payload" + b"koly" + bytes(508))


def _write_valid_handoff(root: Path, name: str = ARTIFACT_NAME) -> Path:
    artifact = _artifact_dir(root, name)
    _write_valid_zip(artifact / "notarized.zip")
    _write_valid_dmg(artifact / "KiroCrew.dmg")
    return artifact


def _run_assembly(root: Path) -> subprocess.CompletedProcess[str]:
    (root / "artifacts").mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", "-c", _assembly_script()],
        cwd=root,
        capture_output=True,
        check=False,
        env={**os.environ, **_assembly_env()},
        **UTF8_TEXT,
    )


def test_missing_exact_gated_artifact_does_not_fall_back_to_unsigned(tmp_path: Path) -> None:
    """A valid-looking unsigned build or stale notarized run cannot be selected."""
    unsigned = _artifact_dir(tmp_path, "unsigned-build-darwin-universal")
    _write_valid_zip(unsigned / "KiroCrew-universal-mac.zip")
    _write_valid_dmg(unsigned / "KiroCrew.dmg")
    _write_valid_handoff(tmp_path, "KiroCrew-notarized-stable-1.2.2")

    result = _run_assembly(tmp_path)

    assert result.returncode != 0
    assert "Required gated macOS ZIP is missing or empty" in result.stderr + result.stdout
    assert not list((tmp_path / "release").glob("*mac.zip"))
    assert not list((tmp_path / "release").glob("*.dmg"))


@pytest.mark.parametrize(
    ("missing_name", "expected_error"),
    (
        ("notarized.zip", "Required gated macOS ZIP is missing or empty"),
        ("KiroCrew.dmg", "Required gated macOS DMG is missing or empty"),
    ),
)
def test_incomplete_gated_handoff_fails(
    tmp_path: Path, missing_name: str, expected_error: str
) -> None:
    artifact = _write_valid_handoff(tmp_path)
    (artifact / missing_name).unlink()

    result = _run_assembly(tmp_path)

    assert result.returncode != 0
    assert expected_error in result.stderr + result.stdout


def test_corrupt_notarized_zip_fails(tmp_path: Path) -> None:
    artifact = _artifact_dir(tmp_path)
    (artifact / "notarized.zip").write_bytes(b"not a zip")
    _write_valid_dmg(artifact / "KiroCrew.dmg")

    result = _run_assembly(tmp_path)

    assert result.returncode != 0
    assert "is not a valid ZIP archive" in result.stderr + result.stdout


def test_non_udif_dmg_fails(tmp_path: Path) -> None:
    artifact = _artifact_dir(tmp_path)
    _write_valid_zip(artifact / "notarized.zip")
    (artifact / "KiroCrew.dmg").write_bytes(b"not a UDIF image")

    result = _run_assembly(tmp_path)

    assert result.returncode != 0
    assert "is not a valid UDIF DMG" in result.stderr + result.stdout


def test_symbols_manifest_is_required_on_the_channel_that_builds(tmp_path: Path) -> None:
    """Insider builds the desktop legs, so a missing Electron pin is a lost artifact.

    The exempt direction is pinned by the passing promotion cases above, which carry
    no manifest at all: a byte promotion (``rebuild != 'true'``) builds no desktop
    leg, so demanding one would fail this job and publish no Release page -- the
    incident release.yml's own comment records. Executed rather than grepped because
    only running the script proves which branch each condition takes.
    """
    _write_valid_handoff(tmp_path)

    result = subprocess.run(
        ["bash", "-c", _assembly_script()],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        env={**os.environ, **_assembly_env(), "CHANNEL": "insider"},
        **UTF8_TEXT,
    )

    assert result.returncode != 0
    assert "No symbols-manifest.json found" in result.stderr + result.stdout


def test_symbols_manifest_is_required_on_a_stable_rebuild(tmp_path: Path) -> None:
    """A stable release now REBUILDS from source, so it too must carry its pin.

    Stable ships a bare version and rebuilds the desktop legs (``rebuild == 'true'``)
    unless the byte-promotion escape hatch is armed. That fresh build produces its
    own manifest, so an emit step that broke on a stable tag would otherwise ship a
    green release whose crash reports no one can decode -- the exact loss the whole
    feature exists to prevent. Scoping the guard to insider alone, as it was before
    stable became a rebuild, left this case silently undecodable.
    """
    _write_valid_handoff(tmp_path)

    result = subprocess.run(
        ["bash", "-c", _assembly_script()],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        env={**os.environ, **_assembly_env(), "CHANNEL": "stable", "REBUILD": "true"},
        **UTF8_TEXT,
    )

    assert result.returncode != 0
    assert "No symbols-manifest.json found" in result.stderr + result.stdout


def test_exact_gated_handoff_is_renamed_for_the_release(tmp_path: Path) -> None:
    gated = _write_valid_handoff(tmp_path)
    unsigned = _artifact_dir(tmp_path, "unsigned-build-darwin-universal")
    (unsigned / "unsigned-mac.zip").write_bytes(b"unsigned zip")
    (unsigned / "unsigned.dmg").write_bytes(b"unsigned dmg")
    (tmp_path / "artifacts" / "cli.whl").write_bytes(b"wheel")

    result = _run_assembly(tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    release = tmp_path / "release"
    release_zip = release / f"KiroCrew-{VERSION}-universal-mac.zip"
    release_dmg = release / f"KiroCrew-{VERSION}-universal.dmg"
    assert release_zip.read_bytes() == (gated / "notarized.zip").read_bytes()
    assert release_dmg.read_bytes() == (gated / "KiroCrew.dmg").read_bytes()
    assert not (release / "unsigned-mac.zip").exists()
    assert not (release / "unsigned.dmg").exists()
