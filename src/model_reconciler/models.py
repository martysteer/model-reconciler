"""Pydantic models for profiles, W3C types, and reconciliation results."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ProfileConfig(BaseModel):
    """A reconciliation profile loaded from YAML."""

    name: str
    prompt: str
    types: list[dict[str, str]]
    slug: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=800, ge=1)
    cache_ttl: int = Field(default=3600, ge=0)
    description: Optional[str] = None
    use_dspy: bool = False


class ReconciliationQuery(BaseModel):
    """A single reconciliation query from OpenRefine."""

    query: str
    type: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=25)
    properties: list[dict[str, Any]] = Field(default_factory=list)


class ReconciliationCandidate(BaseModel):
    """A single match result returned to OpenRefine."""

    id: str
    name: str
    score: float = Field(ge=0, le=100)
    match: bool = False
    type: list[dict[str, str]] = Field(default_factory=list)
    description: Optional[str] = None


class ServiceManifest(BaseModel):
    """W3C Reconciliation Service Manifest (v0.2)."""

    versions: list[str] = ["0.2"]
    name: str
    identifierSpace: str
    schemaSpace: str
    defaultTypes: list[dict[str, str]] = Field(default_factory=list)
    view: Optional[dict[str, str]] = None
    preview: Optional[dict[str, Any]] = None
    suggest: Optional[dict[str, Any]] = None
    extend: Optional[dict[str, Any]] = None
