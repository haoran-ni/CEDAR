"""Domain errors shared across the research-state layer."""


class RuleViolation(ValueError):
    """An action that breaks a research-state rule. Such an event is never recorded."""


class InvalidTransition(RuleViolation):
    """A status change that the relevant state machine does not allow."""
