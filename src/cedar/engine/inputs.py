"""The protocol engine: the second layer between what agents write and what runs."""

import os
from pathlib import Path

from cedar.contracts.base import Actor, ActorKind
from cedar.contracts.events import CalculationInputsExamined, CalculationStateChanged, EventDraft
from cedar.contracts.examination import (
    ExaminationVerdict,
    Finding,
    FindingSeverity,
    InputExamination,
    scientific_input_hash,
    verdict_for,
)
from cedar.contracts.plan import PermissionAction
from cedar.contracts.status import CalculationState
from cedar.errors import RuleViolation
from cedar.protocols.base import ProtocolRegistry
from cedar.state.ledger import RecordResult, ResearchLedger
from cedar.state.projection import authorization_for, calculation_with_inputs
from cedar.store.artifacts import ArtifactStore

PROTOCOL_ENGINE = Actor(kind=ActorKind.PROTOCOL_ENGINE, id="protocol-engine")


class ProtocolEngine:
    def __init__(self, ledger: ResearchLedger, registry: ProtocolRegistry, artifacts: ArtifactStore,
                 runs_root: str | Path, *, actor: Actor = PROTOCOL_ENGINE) -> None:
        self._ledger = ledger
        self._registry = registry
        self._artifacts = artifacts
        self._runs_root = Path(runs_root)
        self._actor = actor

    def examine(self, investigation_id: str, calculation_id: str) -> RecordResult:
        """Generate any missing inputs, examine the full set, and record the verdict.

        Approved inputs move to VALIDATED_INPUT; rejected ones to REJECTED. Inputs that
        need review stay SPECIFIED until the user approves them (see ``approve``).
        """
        state = self._ledger.load(investigation_id)
        calc = state.calculations.get(calculation_id)
        if calc is None:
            raise RuleViolation(f"unknown calculation {calculation_id}")
        if calc.state is not CalculationState.SPECIFIED or calc.examination is not None:
            raise RuleViolation(f"calculation {calculation_id} has already been examined")
        task = calc.task
        protocol = self._registry.get(task.protocol)
        if protocol is None:
            return self._reject(investigation_id, calculation_id,
                                f"protocol {task.protocol.name} {task.protocol.version} is not available")

        findings: list[Finding] = []
        files: dict[str, bytes] = {}
        if task.structure is not None:
            self._read_into(files, findings, protocol.structure_file, task.structure)
        for name, ref in task.authored_files.items():
            if name not in protocol.input_files:
                findings.append(Finding(severity=FindingSeverity.ERROR, code="unexpected_file", file=name,
                                        message=f"{protocol.ref.name} does not take a file named {name}"))
            elif name == protocol.structure_file and task.structure is not None:
                findings.append(Finding(severity=FindingSeverity.ERROR, code="structure_twice", file=name,
                                        message="the structure is given both as structure and as an authored file"))
            else:
                self._read_into(files, findings, name, ref)
        generated, generation_findings = protocol.generate(task, files)
        files.update(generated)
        findings += generation_findings
        findings += protocol.examine(task, files)
        if not files:
            return self._reject(investigation_id, calculation_id, "; ".join(f.message for f in findings)
                                or "no input files")

        refs = {name: self._artifacts.put_bytes(data, label=name) for name, data in sorted(files.items())}
        execution = tuple(name for name in protocol.execution_files if name in refs)
        input_hash = scientific_input_hash(protocol.ref, protocol.code_version, refs, execution)
        duplicate = calculation_with_inputs(state, input_hash)
        if duplicate is not None:
            message = (f"these inputs are identical to registered run {duplicate}, which already has outputs; read "
                       "or analyse it instead of running it again" if state.calculations[duplicate].registered else
                       f"these inputs are identical to calculation {duplicate}; link it instead")
            findings.append(Finding(severity=FindingSeverity.ERROR, code="duplicate_inputs", message=message))
        verdict = verdict_for(findings)
        examination = InputExamination(protocol=protocol.ref, code_version=protocol.code_version, files=refs,
                                       execution_files=execution, input_hash=input_hash, findings=tuple(findings),
                                       verdict=verdict, **_requested(protocol.requested_resources(files)))
        workdir = f"{investigation_id}/{calculation_id}"
        if verdict is not ExaminationVerdict.REJECTED:
            self._write_run_directory(self._runs_root / workdir, files)

        drafts = [EventDraft(actor=self._actor, payload=CalculationInputsExamined(
            calculation_id=calculation_id, examination=examination, workdir=workdir))]
        if verdict is ExaminationVerdict.APPROVED:
            drafts.append(self._move(calculation_id, CalculationState.VALIDATED_INPUT))
        elif verdict is ExaminationVerdict.REJECTED:
            errors = [f.message for f in findings if f.severity is FindingSeverity.ERROR]
            drafts.append(self._move(calculation_id, CalculationState.REJECTED, reason="; ".join(errors)))
        return self._ledger.record(investigation_id, drafts)

    def approve(self, investigation_id: str, calculation_id: str) -> RecordResult:
        """Validate inputs that needed review, once the user has authorized them."""
        state = self._ledger.load(investigation_id)
        authorization = authorization_for(state, PermissionAction.APPROVE_INPUTS, calculation_id)
        if authorization is None:
            raise RuleViolation(f"no user approval names the inputs of {calculation_id}")
        return self._ledger.record(investigation_id, [
            self._move(calculation_id, CalculationState.VALIDATED_INPUT, authorization_id=authorization)])

    def reject(self, investigation_id: str, calculation_id: str, reason: str) -> RecordResult:
        """Reject examined inputs that wait for the researcher, to be reviewed or submitted, once the researcher
        has refused them; their reason goes to the agents."""
        state = self._ledger.load(investigation_id)
        calc = state.calculations.get(calculation_id)
        if calc is None:
            raise RuleViolation(f"unknown calculation {calculation_id}")
        if calc.examination is None or calc.examination.verdict is ExaminationVerdict.REJECTED:
            raise RuleViolation(f"{calculation_id} has no examined inputs that wait for the researcher")
        authorization = authorization_for(state, PermissionAction.REJECT_INPUTS, calculation_id)
        if authorization is None:
            raise RuleViolation(f"no user rejection names the inputs of {calculation_id}")
        return self._ledger.record(investigation_id, [
            self._move(calculation_id, CalculationState.REJECTED, reason=reason, authorization_id=authorization)])

    def _reject(self, investigation_id: str, calculation_id: str, reason: str) -> RecordResult:
        return self._ledger.record(investigation_id, [self._move(calculation_id, CalculationState.REJECTED,
                                                                 reason=reason)])

    def _move(self, calculation_id: str, state: CalculationState, **kwargs) -> EventDraft:
        return EventDraft(actor=self._actor, payload=CalculationStateChanged(
            calculation_id=calculation_id, state=state, **kwargs))

    def _read_into(self, files: dict[str, bytes], findings: list[Finding], name: str, ref) -> None:
        try:
            files[name] = self._artifacts.read(ref)
        except (OSError, ValueError) as exc:
            findings.append(Finding(severity=FindingSeverity.ERROR, code="unreadable_input", file=name,
                                    message=f"cannot read the stored {name}: {exc}"))

    @staticmethod
    def _write_run_directory(directory: Path, files: dict[str, bytes]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = directory / name
            if path.exists():
                if path.read_bytes() != data:
                    raise RuntimeError(f"{path} already exists with different content")
                continue
            path.write_bytes(data)
            os.chmod(path, 0o444)


def _requested(resources) -> dict:
    """What the job script asks for, as the examination records it."""
    if resources is None:
        return {}
    return {"requested_core_hours": resources.core_hours, "requested_tasks": resources.ntasks,
            "requested_minutes": resources.walltime_minutes, "partition": resources.partition}
