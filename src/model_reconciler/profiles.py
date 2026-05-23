"""Load YAML profile files into validated ProfileConfig objects."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from model_reconciler.models import ProfileConfig

logger = logging.getLogger(__name__)


def load_profile(path: Path) -> ProfileConfig:
    """Load one YAML file, derive slug from filename if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a YAML mapping: {path}")

    if not data.get("slug"):
        data["slug"] = path.stem

    try:
        return ProfileConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid profile {path}: {e}") from e


def load_all_profiles(profiles_dir: Path) -> list[ProfileConfig]:
    """Load all *.yaml files from a directory. Rejects duplicate slugs."""
    profiles = []
    for path in sorted(profiles_dir.glob("*.yaml")):
        profile = load_profile(path)
        profiles.append(profile)
        logger.info(f"Loaded profile: {profile.slug} ({profile.name})")

    slugs = [p.slug for p in profiles]
    dupes = [s for s in slugs if slugs.count(s) > 1]
    if dupes:
        raise ValueError(f"Duplicate profile slugs: {set(dupes)}")

    return profiles
