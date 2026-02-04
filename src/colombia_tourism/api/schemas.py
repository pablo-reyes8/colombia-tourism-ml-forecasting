"""Pydantic schemas for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    model_uri: Optional[str] = None
    records: List[Dict[str, Any]]
    target: Optional[str] = None


class ExplainRequest(BaseModel):
    model_uri: Optional[str] = None
    records: List[Dict[str, Any]]
    target: Optional[str] = None
    max_samples: int = 200
