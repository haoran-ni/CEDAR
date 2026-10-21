# Round N: <investigation title>

<!--
The layout of every round page in the research notebook. Pages are generated from the
event log by src/sim2real/notebook.py and are never edited by hand. The "## " headings
below must match ROUND_SECTIONS in that file, in order; a test checks this.

A round is one stretch of investigating. Round 0 is the setup before the first one. A new
round starts each time the investigation moves into INVESTIGATING (after planning or
after a review), or when the Chief Scientist starts one explicitly.
-->

Investigation `<id>` · events #<first>–#<last> · <time of first> to <time of last> (UTC)

Who started the round, and the reason they recorded.

## What did we hypothesize?

Hypotheses proposed this round: statement, falsifiers, discriminating observations,
assumptions and links to related hypotheses. Hypotheses already open are listed with
their status.

## Why?

The recorded reasons for this round's decisions: each new hypothesis's rationale, changes
of investigation status, the observable alignment, plan revisions (with the researcher's
message that prompted them and the priorities they changed), and the Chief Scientist's
or researcher's changes to a hypothesis's status. Each reason is quoted and attributed.

## What did we predict?

Predictions registered this round: statement, expected outcome, what would falsify it,
the observable, and the calculations it is about. The record refuses predictions about
calculations whose outcome was already known, and never changes them afterwards.

## What did we test?

Calculations requested or reused this round: what they estimate, why (the requester's
words), protocol, the variables varied, controlled and set for numerical accuracy, which
input files the agent wrote and which the protocol generated, the examination verdict
with its errors and review items, and the state at the end of the round. Submissions
(with the authorizing message), cancellations and rejections follow.

## What did we observe?

Evidence recorded this round: quality, value with unit and reference, who read it, the
cited output lines with links to the stored files, the quality checks that passed or
failed, limitations and non-comparability conditions. Jobs that the scheduler ended
without completing are listed too.

## What changed our assessment?

Assessments recorded this round: the previous and new assessment, PARTIAL or FINAL, the
evidence cited, and the judge's rationale and limitations.

## What remains unknown?

At the end of the round: whether the compared observables are aligned, hypotheses not yet
assessed or not settled (INCONCLUSIVE, NOT_TESTED, NOT_COMPARABLE, or PARTIAL), and this
round's evidence that is not VALIDATED.

## What is next?

The Chief Scientist's recommendations this round: each candidate next step with the
hypotheses it bears on, what each predicts it would show, why it is informative, its cost,
feasibility and priority, and which were taken and why. Then, at the end of the round: the
next round and why it started, queued agent work (with the recommendation it carries out),
what waits for the researcher (with the exact command), unfinished calculations, and the
plan's objective and priorities.

## Researcher's messages

Every message the researcher sent this round, verbatim, with any authorization it grants.

## Agent reports

Each hypothesis agent's and the Evidence Judge's final report, in their own words, with
the task, outcome, turns and serving model. Reports are not evidence. The Chief
Scientist's replies are counted here and kept in the agent transcripts. The round's model
token use closes the page.
