"""Evidence from outside the calculations: measured values and facts about the experiment.

Since the investigation is about improving the calculation, what is known about the measurement
is often no more than a sentence from the researcher ("the measured adsorption height is
2.5 Å"). An agent records such a fact as evidence, citing where it came from:

- a researcher's message, by its record id: the quote is checked against the recorded text;
- the investigation's recorded measurement, by its label: the value must be the recorded one;
- an attached material or a readable file: stored by hash, and a quote from its text, or from
  the text of a PDF, is checked against it;
- a figure or a web page: nothing can be checked, so the evidence stays PROVISIONAL.

The validator records the evidence, never the agent: quality comes from the checks made here. A
quoted value must appear in the quote. Such evidence existed before any prediction, so it can
establish the discrepancy or challenge a hypothesis, never support it.
"""

import io
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from cedar.agents.media import is_image, is_pdf
from cedar.codes.fhiaims.output import numbers_in
from cedar.contracts.base import Actor, ExternalSource, Observable, Quantity, new_id
from cedar.contracts.events import EventDraft, EvidenceRecorded
from cedar.contracts.evidence import EvidenceKind, EvidenceObject, QualityCheck
from cedar.contracts.status import EvidenceQuality
from cedar.engine.results import VALIDATOR, ValidationOutcome
from cedar.state.ledger import ResearchLedger
from cedar.store.artifacts import ArtifactStore

SourceKind = Literal["researcher_message", "investigation", "material", "file", "url"]
KINDS = {"measurement": EvidenceKind.PUBLISHED_EXPERIMENT, "constraint": EvidenceKind.INDIRECT_CONSTRAINT,
         "inference": EvidenceKind.INFERENCE}
MAX_FILE_BYTES = 100_000_000
UNCHECKED = "Its source could not be checked by CEDAR (a figure or a web page); the value is as the agent read it."
LIMITATION = ("Reported outside CEDAR's calculations and recorded as the source states it; it existed before any "
              "prediction, so it can establish the discrepancy or challenge a hypothesis, not support it.")


@dataclass(frozen=True)
class ExternalFact:
    kind: Literal["measurement", "constraint", "inference"]
    observable: Observable
    summary: str
    source_kind: SourceKind
    source: str
    """A message id, a measurement label, a material id, an absolute path or a URL."""
    quote: str
    """The source's words, copied exactly; for a figure, what was read off it and where."""
    value: Quantity | None = None
    locator: str | None = None
    """Page, figure or table within the source."""
    title: str | None = None
    conditions: dict | None = None
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


def _same(text: str) -> str:
    return " ".join(text.split())


def _text_of(data: bytes) -> str | None:
    """The text of a stored file: itself if it is text, the extracted text of a PDF; None for an image."""
    if is_pdf(data):
        try:
            return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
        except (PdfReadError, ValueError, OSError):
            return None
    if is_image(data):
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


class ExternalEvidenceRecorder:
    def __init__(self, ledger: ResearchLedger, artifacts: ArtifactStore, *, readable=None,
                 actor: Actor = VALIDATOR) -> None:
        """``readable`` resolves a path the agent may read, raising if it may not."""
        self._ledger = ledger
        self._artifacts = artifacts
        self._readable = readable
        self._actor = actor

    def record(self, investigation_id: str, fact: ExternalFact, *, reader: Actor) -> ValidationOutcome:
        state = self._ledger.load(investigation_id)
        checks: list[QualityCheck] = []
        sources: list = []
        limitations = [LIMITATION, *fact.limitations]
        text: str | None = None
        if fact.source_kind == "researcher_message":
            message = state.interventions.get(fact.source)
            if message is None:
                return ValidationOutcome(None, (f"no researcher message {fact.source} in the record",))
            text = message.text
            sources.append(ExternalSource(citation=f"the researcher's message {fact.source}", locator=fact.locator))
        elif fact.source_kind == "investigation":
            measured = next((e for e in state.spec.experiments if e.label == fact.source), None)
            if measured is None:
                labels = ", ".join(e.label for e in state.spec.experiments) or "none"
                return ValidationOutcome(None, (f"no recorded measurement {fact.source!r} (recorded: {labels})",))
            if fact.value is not None:
                same = measured.quantity is not None and measured.quantity.unit == fact.value.unit and math.isclose(
                    measured.quantity.value, fact.value.value, rel_tol=1e-12, abs_tol=1e-12)
                if not same:
                    return ValidationOutcome(None, (f"the recorded value of {fact.source!r} is "
                                                    + (f"{measured.quantity.value!r} {measured.quantity.unit}"
                                                       if measured.quantity else "not given"),))
                checks.append(QualityCheck(name="matches_recorded_measurement", passed=True,
                                           detail=f"as recorded when the investigation was opened: {fact.source}"))
            sources += [ExternalSource(citation=f"the investigation's recorded measurement {fact.source!r}"),
                        *measured.sources]
        elif fact.source_kind in ("material", "file"):
            if fact.source_kind == "material":
                material = state.materials.get(fact.source)
                if material is None:
                    return ValidationOutcome(None, (f"no material {fact.source} in this investigation",))
                ref, name = material.artifact, f"material {fact.source}: {material.name}"
            else:
                path = Path(fact.source)
                resolved = self._readable(path) if self._readable is not None else path.resolve()
                if not resolved.is_file():
                    return ValidationOutcome(None, (f"{fact.source} is not a readable file",))
                if resolved.stat().st_size > MAX_FILE_BYTES:
                    return ValidationOutcome(None, (f"{fact.source} is larger than {MAX_FILE_BYTES} bytes",))
                ref, name = self._artifacts.put_file(resolved), str(resolved)
            text = _text_of(self._artifacts.read(ref))
            sources += [ExternalSource(citation=name, locator=fact.locator), ref]
            if text is None:
                limitations.append(UNCHECKED)
        else:
            sources.append(ExternalSource(citation=fact.title or fact.source, url=fact.source, locator=fact.locator,
                                          accessed_on=date.today()))
            limitations.append(UNCHECKED)

        if text is not None:
            if _same(fact.quote) not in _same(text):
                return ValidationOutcome(None, ("the quote is not found, word for word, in the source",))
            checks.append(QualityCheck(name="quote_verified", passed=True,
                                       detail=f"found in {fact.source_kind.replace('_', ' ')} {fact.source}"))
        if fact.value is not None and fact.source_kind != "investigation":
            if not any(math.isclose(n, fact.value.value, rel_tol=1e-9, abs_tol=1e-12) for n in numbers_in(fact.quote)):
                return ValidationOutcome(None, (f"the quote does not contain the value {fact.value.value!r}",))
        quality = EvidenceQuality.VALIDATED if checks else EvidenceQuality.PROVISIONAL
        evidence = EvidenceObject(
            kind=KINDS[fact.kind], summary=fact.summary, observable=fact.observable, value=fact.value,
            conditions=dict(fact.conditions or {}), quality=quality, quality_checks=tuple(checks),
            assumptions=tuple(fact.assumptions), limitations=tuple(limitations),
            method={"source": fact.source_kind, "quote": fact.quote}, sources=tuple(sources), interpreted_by=reader)
        return ValidationOutcome(self._ledger.record(investigation_id, [EventDraft(
            actor=self._actor, payload=EvidenceRecorded(evidence_id=new_id("ev"), evidence=evidence))]))

