"""The agents' web access, as the researcher sets it for each investigation with ``/web``.

Off by default, so blind tests stay possible and nothing about the investigation reaches a
search engine unless the researcher wants it to. When it is on, the Chief Scientist and the
hypothesis agents get the model provider's own web search and web fetch; the Evidence Judge
and the approver never do, so judgements and approvals rest on the research record. Fetching
only reaches pages whose addresses already appear in the conversation, e.g. in search results.

The input reviewer always has the web, whatever the setting, but only the code documentation
sites in ``REVIEW_DOMAINS``: it checks inputs against the code's documentation, and nothing it
reads there can carry results about the system under investigation back to the agents.
"""

from cedar.contracts.investigation import RetrievalPolicy
from cedar.models.types import WebAccess
from cedar.state.projection import RetrievalRecord

WEB_ROLES = frozenset({"chief_scientist", "hypothesis_agent"})
REVIEW_DOMAINS = ("fhi-aims.org", "fhi-aims-club.gitlab.io", "vasp.at", "wiki.fysik.dtu.dk", "ase-lib.org")
"""Where the input reviewer may read the codes' documentation: FHI-aims and its tutorials, the VASP wiki, and ASE.
Add a code's documentation site here when a protocol for it is added."""


def web_access(setting: RetrievalRecord | None, role: str) -> WebAccess | None:
    """What a run of ``role`` gets under ``setting``: None for no web."""
    if role == "input_reviewer":
        return WebAccess(allowed_domains=REVIEW_DOMAINS)
    if setting is None or role not in WEB_ROLES or setting.policy is RetrievalPolicy.DISABLED:
        return None
    return WebAccess(allowed_domains=setting.allowed_domains)


def web_summary(setting: RetrievalRecord | None) -> str:
    if setting is None or setting.policy is RetrievalPolicy.DISABLED:
        return "off"
    if setting.policy is RetrievalPolicy.CURATED_ONLY:
        return "only " + ", ".join(setting.allowed_domains)
    return "on"
