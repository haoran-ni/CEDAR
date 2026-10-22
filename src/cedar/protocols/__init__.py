"""Versioned protocols that examine inputs and check runs for each supported calculation type."""

from cedar.protocols.base import Protocol, ProtocolRegistry, Reading, RunReport
from cedar.protocols.fhiaims_run import (
    TOTAL_ENERGY,
    ZERO_BROADENING_ENERGY,
    FhiAimsRun,
    FhiAimsRunJobChecked,
    FhiAimsRunReviewed,
    FhiAimsRunSlurmOnly,
)
from cedar.protocols.vasp_run import FREE_ENERGY, SIGMA0_ENERGY, VaspRun

CODES = {"fhiaims.run": "fhi-aims", "vasp.run": "vasp"}
"""The code each protocol runs, by protocol name: its key in a compute profile's ``codes``."""

__all__ = ["CODES", "FhiAimsRun", "FhiAimsRunJobChecked", "FhiAimsRunReviewed", "FhiAimsRunSlurmOnly", "Protocol",
           "ProtocolRegistry", "Reading", "RunReport", "VaspRun",
           "FREE_ENERGY", "SIGMA0_ENERGY", "TOTAL_ENERGY", "ZERO_BROADENING_ENERGY"]
