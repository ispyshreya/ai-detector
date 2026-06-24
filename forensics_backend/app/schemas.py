from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ModuleStatus = Literal["ok", "warning", "skipped", "error"]


class ModuleBase(BaseModel):
    status: ModuleStatus
    summary: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_ms: int


class ExifSignal(BaseModel):
    field: str
    value: str


class ExifResult(ModuleBase):
    metadata_present: bool
    gps_present: bool
    software_tag: str | None = None
    anomaly_flags: list[str] = Field(default_factory=list)
    extracted_fields: list[ExifSignal] = Field(default_factory=list)
    raw_field_count: int = 0


class ElaRegion(BaseModel):
    label: str
    left: int
    top: int
    right: int
    bottom: int
    score: float


class ElaResult(ModuleBase):
    heatmap_url: str | None = None
    max_intensity: float
    mean_intensity: float
    suspicious_regions: list[ElaRegion] = Field(default_factory=list)


class ReverseSearchMatch(BaseModel):
    title: str
    link: str
    source: str | None = None
    first_seen: str | None = None
    confidence_note: str | None = None


class ReverseSearchResult(ModuleBase):
    provider: str | None = None
    query_mode: str | None = None
    earliest_known_appearance: str | None = None
    matches: list[ReverseSearchMatch] = Field(default_factory=list)


class C2PAAssertion(BaseModel):
    label: str
    value: str


class C2PAResult(ModuleBase):
    manifest_present: bool
    validation_status: str
    signer: str | None = None
    issuer: str | None = None
    assertions: list[C2PAAssertion] = Field(default_factory=list)


class ForensicsEnvelope(BaseModel):
    exif: ExifResult
    ela: ElaResult
    reverse_search: ReverseSearchResult
    c2pa: C2PAResult


class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    content_type: str | None = None
    sha256: str
    analyzed_at: datetime
    ready_for_triangulation: bool
    completed_modules: list[str] = Field(default_factory=list)
    partial_modules: list[str] = Field(default_factory=list)
    failed_modules: list[str] = Field(default_factory=list)
    forensics: ForensicsEnvelope
