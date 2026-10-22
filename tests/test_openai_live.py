"""The OpenAI adapter and the agent loop against the real API. Skipped unless CEDAR_LIVE_OPENAI is set.

They spend real money, well under a dollar in all: run them with ``CEDAR_LIVE_OPENAI=1 pytest -s
tests/test_openai_live.py``. The key is the one ``cedar auth add openai`` stored (or OPENAI_API_KEY), read by
the adapter as CEDAR reads it, and never printed. Every test runs with an isolated configuration and no keys
in its environment (``conftest.py``), so where the key is is noted when this module loads, before that. The
model is ``CEDAR_LIVE_OPENAI_MODEL``, else GPT-5.6 Sol.
"""

import io
import os

import pytest
from PIL import Image

from cedar.agents import AgentRunner, AgentTool, ToolOutcome, TranscriptStore
from cedar.agents.media import load_media
from cedar.contracts.base import Contract
from cedar.models import CredentialStore, ModelSettings, WebAccess, create_client
from cedar.paths import config_dir

pytestmark = pytest.mark.skipif("CEDAR_LIVE_OPENAI" not in os.environ,
                                reason="set CEDAR_LIVE_OPENAI to run against the real OpenAI API (costs money)")

SETTINGS = ModelSettings(provider="openai", model=os.environ.get("CEDAR_LIVE_OPENAI_MODEL", "gpt-5.6-sol"),
                         effort="medium")
CREDENTIALS = CredentialStore(config_dir() / "credentials.toml",
                              environ={"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")})


class Name(Contract):
    name: str


class Nothing(Contract):
    pass


@pytest.fixture
def runner(tmp_path):
    client = create_client(SETTINGS, CREDENTIALS)
    return AgentRunner(client, SETTINGS, TranscriptStore(tmp_path / "transcripts"))


def report(name, result):
    usage = result.usage
    print(f"\n{name}: {result.outcome}, {result.turns} calls, ${result.cost_usd:.4f}, served by "
          f"{result.served_models}; tokens {usage.input_tokens} fresh, {usage.cache_read_tokens} cached, "
          f"{usage.cache_write_tokens} written, {usage.output_tokens} out; {usage.web_searches} searches, "
          f"{usage.web_fetches} pages")


def tiny_pdf(text: str) -> bytes:
    """A one-page PDF showing ``text``."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_a_tool_round_trip_and_a_resumed_conversation(runner):
    tools = [AgentTool("lookup", "Look up the code word for a name.", Name, lambda args: f"code word: heliotrope-{len(args.name)}")]
    first = runner.run("live-tools", system="You are terse.", tools=tools,
                       instruction="Call the lookup tool once with the name 'alpha', then reply with only the code "
                                   "word it returned.")
    report("tool round trip", first)
    assert first.outcome == "finished" and "heliotrope-5" in first.final_text and first.turns == 2
    # The same session again: the earlier turns, reasoning included, go back as they were.
    second = runner.run("live-tools", system="You are terse.", tools=tools,
                        instruction="Which name did you look up? Reply with only the name.")
    report("resumed conversation", second)
    assert second.outcome == "finished" and "alpha" in second.final_text.lower()


def test_images_and_pdfs_in_a_tools_output(runner):
    image = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 20)).save(image, format="PNG")
    media = (load_media(image.getvalue(), "square.png")[0], load_media(tiny_pdf("CEDAR-PDF-7491"), "code.pdf")[0])
    tools = [AgentTool("show", "Show the image and the PDF.", Nothing,
                       lambda args: ToolOutcome("An image and a PDF.", media=media))]
    result = runner.run("live-media", system="You are terse.", tools=tools,
                        instruction="Call the show tool once. Then reply in one line with the colour of the image and "
                                    "the code written in the PDF.")
    report("images and PDFs", result)
    assert result.outcome == "finished"
    assert "red" in result.final_text.lower() and "7491" in result.final_text


def test_web_search_within_the_allowed_domains(runner):
    result = runner.run("live-web", system="You are terse.", tools=[],
                        instruction="Search the web for what FHI-aims is and answer in one sentence.",
                        web=WebAccess(allowed_domains=("fhi-aims.org",), max_searches=2, max_fetches=2))
    report("web search", result)
    assert result.outcome == "finished" and result.usage.web_searches >= 1


def test_a_reply_cut_off_at_its_output_limit_can_be_continued(runner, tmp_path):
    short = AgentRunner(runner._client, SETTINGS.model_copy(update={"max_output_tokens": 16}),
                        TranscriptStore(tmp_path / "transcripts"))
    cut = short.run("live-cut", system="You are terse.", tools=[],
                    instruction="Write a paragraph about adsorption of CO on copper.")
    report("cut off", cut)
    assert cut.outcome == "truncated"
    # The cut-off turn goes back as it was, with the next instruction.
    resumed = runner.run("live-cut", system="You are terse.", tools=[], instruction="Now just reply: ok")
    report("after the cut", resumed)
    assert resumed.outcome == "finished" and "ok" in resumed.final_text.lower()
