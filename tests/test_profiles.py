"""Smoke tests: profile loading and validation."""

import pytest

from model_reconciler.profiles import load_all_profiles, load_profile


def test_valid_profile_loads(fixtures_dir):
    """Valid YAML loads with slug derived from filename."""
    p = load_profile(fixtures_dir / "valid.yaml")
    assert p.name == "Test Profile"
    assert p.slug == "valid"
    assert len(p.types) == 1
    assert p.temperature == 0.1
    assert p.cache_ttl == 3600
    assert p.use_dspy is False


def test_missing_file_raises(tmp_path):
    """FileNotFoundError for nonexistent YAML."""
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "ghost.yaml")


def test_missing_required_field_raises(tmp_path):
    """ValueError when required field (types) is absent."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: Broken\nprompt: No types here\n")
    with pytest.raises(ValueError, match="Invalid profile"):
        load_profile(bad)


def test_bad_yaml_raises(tmp_path):
    """ValueError when file isn't a YAML mapping."""
    bad = tmp_path / "list.yaml"
    bad.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_profile(bad)


def test_load_all_profiles(fixtures_dir):
    """load_all_profiles returns validated list."""
    profiles = load_all_profiles(fixtures_dir)
    assert len(profiles) >= 1
    assert all(p.slug for p in profiles)


def test_duplicate_slugs_rejected(tmp_path):
    """ValueError when two profiles share a slug."""
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(
            "name: Dupe\nprompt: Test\nslug: same\ntypes:\n  - id: x\n    name: X\n"
        )
    with pytest.raises(ValueError, match="Duplicate"):
        load_all_profiles(tmp_path)
