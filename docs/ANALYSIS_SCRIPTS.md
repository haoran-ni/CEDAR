# Analysis scripts

Hypothesis agents take results from finished calculations in two ways:

- **Readings.** A value the code printed is recorded with `submit_reading`, citing the file and lines. The validator finds the quote in the stored output before recording it.
- **Analysis scripts.** Anything that must be computed is handled by a Python script the agent writes, run with `run_analysis`. Examples: a gap from an eigenvalue table, a distance from a geometry, a planar average, or a difference between runs. The agent then records a result with `record_result`.

No quantity is built into CEDAR. The FHI-aims protocol (`fhiaims.run`) checks that a run is healthy; what to take from it is the agent's decision. The runs can be CEDAR's own or runs computed outside CEDAR and registered ([`WORKBENCH.md`](WORKBENCH.md)).

Analysis scripts are not the agent's general-purpose commands: those run in its workspace, in the workbench's sandbox, and give context, not evidence ([`WORKBENCH.md`](WORKBENCH.md)).

## How a script runs

1. **Examination.** The script is parsed and refused before it runs if it:
   - imports modules that start processes, open network connections, load foreign code or import dynamically (`subprocess`, `socket`, `urllib`, `multiprocessing`, `ctypes`, `importlib`, …)
   - calls `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `os.fork` or `os.kill`
   - uses `eval`, `exec`, `compile` or `__import__`
2. **Staging.** For each calculation it names, which must be one of its own hypothesis's or a registered run shared with every agent, the requested files are copied read-only to `inputs/<calculation_id>/` in a fresh working directory: by default the run's stored outputs and its input files, or any file of the run directory (for a registered run, the directory it came from) named by relative path. Each file is identified by its content hash, and a file the run check already stored must be that same file.
3. **Execution.** The script runs on the login node with the Python environment CEDAR runs in (NumPy, SciPy, ASE and matplotlib available), with:
   - a clean environment: no credentials, a private home and temporary directory, single-threaded numerical libraries
   - limits: 10 minutes of CPU, 15 minutes of wall-clock time, 4 GiB of memory, 200 MiB per file written; at most two scripts run at once, whichever agents started them
   - an audit hook that refuses to read anything but the staged inputs, the Python installation and system libraries; to write, delete or rename anything outside the working directory; to start any program; or to open a socket
4. **Collection.** What the script printed, every file it wrote, and the results and checks it reported are stored by content hash. The run is recorded as an `analysis_recorded` event whether it completed, failed or timed out. The working directory is then removed.

## Results and checks

A script reports its results, and the checks it made, in `results.json` in its working directory:

```json
{"results": {"gap": {"value": 6.99193, "unit": "eV", "description": "LUMO minus HOMO, final eigenvalue listing",
                     "checks": {"matches_printed_gap": {"passed": true, "detail": "differs by 3.5e-6 eV (table rounding)"}}},
             "homo": {"value": -8.78611, "unit": "eV"}},
 "checks": {"electron_count": {"passed": true, "detail": "14 electrons from the occupations"},
            "final_listing": {"passed": true, "detail": "last eigenvalue table, before 'Self-consistency cycle converged.'"}}}
```

- Top-level checks apply to every result; a result's own checks apply to it alone. Here a failed `matches_printed_gap` would affect the gap but not the HOMO.
- One script may report several results. Only the results and checks of a script that completed are kept.
- A result is not taken if any check that applies to it is malformed, because the malformed check could have been a failed one.

### Two kinds of checks

- **Run checks** are the same for every FHI-aims run, whatever its purpose. When a job ends, the protocol checks that the scheduler reported completion, the code version matches the compute profile, FHI-aims ended normally, the SCF converged (and the geometry, for a relaxation), and the input files are unchanged. They come from FHI-aims's own output markers, not from any quantity or site.
- **Script checks** are what this particular run's purpose needs, decided by the agent and computed by its script: the electron count, the right SCF block, agreement with a printed value, forces below a threshold, an expected output file present, the intended spin state, a quantity converged with respect to a numerical setting.

Both kinds are recorded on the evidence. The script's checks appear with the prefix `script:`.

## Scripts never run DFT

A script cannot start a program, so it cannot launch a DFT code, an MPI job or a shell command. Calculations run only as examined jobs through the job manager, through Slurm or locally on an interactive node, after the researcher authorizes them.

The guards stop mistakes and naive misuse. They are not a security boundary against code written to evade them: Python audit hooks can be bypassed from native code, and shared login nodes often do not allow operating-system sandboxes. Agents are instructed never to work around the guards.

## Correctness

The agents' instructions require that scripts:

- parse what the file actually contains, printing the lines they matched and the intermediate numbers
- take final values: the last converged SCF block, the final geometry of a relaxation, the right spin channel and k-point
- name the units of every value and never convert silently
- check what the run's purpose needs, and report every check in `results.json`, failed ones included
- say what is ambiguous instead of recording it

Each piece of evidence from an analysis:

- carries the value from the analysis record, never one typed by the agent, and names the result it is
- carries every check the script reported for that result; the record refuses evidence that leaves one out
- links the script, every staged input, what the script printed and its products
- is `VALIDATED` only if every run it used ended `COMPLETED` and passed its run checks, and every script check on it passed. A failed run gives the run's failure quality (`INVALID`, or `INCONCLUSIVE` for an unconverged relaxation); a failed script check gives `INCONCLUSIVE`
- states as a limitation that the script has not been reviewed by an independent agent or person

The Evidence Judge's dossier shows the script behind each analysis result, so the Judge can see how a decisive value was obtained.

## Future: a second layer of review

Today the agent that writes a script is also the one that records its result; nothing independent checks the script. Independent judge agents can later add a second layer of safety and correctness:

- **Script review.** A reviewer agent reads each analysis script, and the files it parsed, before its result can count as `VALIDATED`. Until then the result would be `PROVISIONAL`.
- **Independent re-derivation.** For decisive values, a second agent writes its own script from the same staged inputs. Disagreement makes the evidence `INCONCLUSIVE` and is reported to the researcher.
- **Safety review.** A reviewer flags scripts that try to reach outside their inputs, even when the guards stopped them.

The evidence record already keeps everything such a review needs: the script, the hashed inputs, the output, the results and the script's checks.

## Where it is implemented

- `src/cedar/engine/analysis.py`: examination, staging, execution, collection
- `src/cedar/engine/analysis_guard.py`: limits and the audit hook, run in the child process
- `src/cedar/engine/results.py`: run checks, readings and analysis results as evidence
- `src/cedar/agents/toolkits/research.py`: the agent's `run_outputs`, `submit_reading`, `run_analysis` and `record_result` tools
- `src/cedar/agents/prompts.py`: the correctness and safety requirements given to agents
