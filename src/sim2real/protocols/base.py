"""Protocols: what a calculation type accepts, how its inputs are examined, how its runs are checked.

A protocol never runs anything. It can generate inputs an agent did not write, examine
the complete set of files that would run, and check a finished run's health. What to take
from a healthy run is the agent's decision: the agent declares its own observable, and reads
or computes it from the outputs.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from sim2real.contracts.base import Quantity
from sim2real.contracts.evidence import QualityCheck
from sim2real.contracts.examination import Finding
from sim2real.contracts.status import EvidenceQuality, SchedulerState
from sim2real.contracts.task import ProtocolRef, ScientificTaskSpec


@dataclass(frozen=True)
class Reading:
    """A value the protocol read from a raw output file, with the lines it came from."""

    name: str
    value: Quantity
    file: str
    line_start: int
    line_end: int
    quote: str


@dataclass(frozen=True)
class RunReport:
    checks: tuple[QualityCheck, ...]
    failure_quality: EvidenceQuality
    """Quality to assign when any check fails."""
    readings: tuple[Reading, ...] = ()
    """Values the protocol knows how to read from this kind of run, e.g. the final total energy."""
    limitations: tuple[str, ...] = ()
    method: dict[str, JsonValue] = field(default_factory=dict)


class Protocol(ABC):
    ref: ProtocolRef
    input_files: tuple[str, ...]
    structure_file: str
    execution_files: tuple[str, ...]
    output_files: tuple[str, ...]

    @property
    @abstractmethod
    def code_version(self) -> str: ...

    def describe(self) -> str:
        """What an agent needs to know to request this protocol."""
        lines = [f"Protocol {self.ref.name} {self.ref.version}",
                 "Observable: declare your own (name, definition, unit): what you will take from this run. The "
                 "protocol checks the run's health; you read values from the outputs, citing the lines, or compute "
                 "them with an analysis script."]
        lines.append(f"Input files: {', '.join(self.input_files)} (structure file: {self.structure_file}; "
                     f"job script: {', '.join(self.execution_files)}). Files you do not write are generated.")
        lines.append(f"Outputs: {', '.join(self.output_files)}")
        return "\n".join(lines)

    @abstractmethod
    def generate(
        self, task: ScientificTaskSpec, present: Mapping[str, bytes]
    ) -> tuple[dict[str, bytes], list[Finding]]:
        """Write the input files missing from ``present``; never alter files that are present."""

    @abstractmethod
    def examine(self, task: ScientificTaskSpec, files: Mapping[str, bytes]) -> list[Finding]:
        """Check the complete set of files against the declared task and this protocol's policy."""

    @abstractmethod
    def check_run(
        self,
        task: ScientificTaskSpec,
        outputs: Mapping[str, bytes],
        scheduler_state: SchedulerState | None,
    ) -> RunReport: ...


class ProtocolRegistry:
    def __init__(self, protocols: Iterable[Protocol] = ()) -> None:
        self._protocols: dict[tuple[str, str], Protocol] = {}
        for protocol in protocols:
            self.register(protocol)

    def register(self, protocol: Protocol) -> None:
        key = (protocol.ref.name, protocol.ref.version)
        if key in self._protocols:
            raise ValueError(f"protocol {key} is already registered")
        self._protocols[key] = protocol

    def get(self, ref: ProtocolRef) -> Protocol | None:
        return self._protocols.get((ref.name, ref.version))

    def refs(self) -> list[ProtocolRef]:
        return [p.ref for p in self._protocols.values()]
