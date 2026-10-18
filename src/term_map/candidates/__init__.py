"""Candidate generation modules (per coding system, independent)."""
from .base import Candidate, CandidateSet, merge_candidates

__all__ = ["Candidate", "CandidateSet", "merge_candidates"]
