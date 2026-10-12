# TODO / Codebase Review

Findings from a review of the codebase (comments, hardcoded paths, duplication, structure). No actual `TODO`/`FIXME`/`HACK` comments exist in the source — this list is inferred from structural/code-quality issues. No security red flags found (no `eval`/`exec`/`pickle`/`shell=True`/hardcoded credentials).

## 🔴 Critical

1. **Remove hardcoded absolute paths** — [Prepare_input_data.py](src/Prepare_input_data.py#L44), [Prijstool_Deltascenarios.py](src/Prijstool_Deltascenarios.py#L8) hardcode `C:\git_repos\effectmodule_landbouw\...`, and [prijstool_preprocessing.py](src/prijstool_preprocessing.py#L28) does `sys.path.append(r"C:\git_repos\...")`. These break on any other machine. Should use relative paths / proper package imports / config file.
2. **Externalize the `P:\archivedprojects\...` UNC network path**, duplicated across all 4 Snakemake workflow files (e.g. [wf_landbouw_scenarios](snakes/wf_landbouw_scenarios#L26)). Single point of failure if the network share changes; put it in a config file instead.
3. **Consolidate the duplicate Prijstool implementations.** Three near-identical files exist:
   - [Prijstool_Deltascenarios.py](src/Prijstool_Deltascenarios.py) (thin wrapper, looks superseded)
   - [Prijstool_Deltascenarios_Class_rewrite_correct.py](src/Prijstool_Deltascenarios_Class_rewrite_correct.py)
   - [Prijstool_Deltascenarios_Class_rewrite_new_act.py](src/Prijstool_Deltascenarios_Class_rewrite_new_act.py) (only this one is actually referenced by the active workflow, [wf_landbouw_test_new](snakes/wf_landbouw_test_new#L122))
   
   Pick one canonical version, delete the others, and use git history/branches instead of keeping dead copies around.
4. **Dangling references to a non-existent file**: `snakes/wf_landbouw_scenarios` and `snakes/wf_landbouw_test` reference `Prijstool_Deltascenarios_v12.py`, which doesn't exist in `src/`. These workflows are likely broken/stale — confirm and remove or fix.
5. **Prune duplicate/stale Snakemake workflows**: 4 workflow files (`wf_landbouw_scenarios`, `wf_landbouw_scenarios_new`, `wf_landbouw_test`, `wf_landbouw_test_new`) with only `_test_new` apparently current.

## 🟠 High

6. **No tests** — no `tests/` folder, no pytest config. Add unit tests at least for the core calculation logic (Prijstool class, `bereken_droogterisico.py`).
7. **No logging** — code relies on `print()` instead of the `logging` module; makes debugging workflow runs harder.
8. **No README/usage docs** — add a README explaining how to run the Snakemake workflows, required input data, and environment setup via pixi.
9. **No linting/formatting config** (no ruff/black/flake8), no CI (`.github/workflows`) to catch regressions automatically.
10. **No input validation / error handling** around file reads — paths aren't checked for existence before use.

## 🟡 Medium

11. **Magic numbers without clear justification**: `19.4`, `36.3` (price-level adjustment %) in both Prijstool variants ([Class_rewrite_correct.py#L20](src/Prijstool_Deltascenarios_Class_rewrite_correct.py#L20), [Class_rewrite_new_act.py#L20-21](src/Prijstool_Deltascenarios_Class_rewrite_new_act.py#L20-L21)), and `0.0001` m²→ha conversion in [prijstool_preprocessing.py#L74](src/prijstool_preprocessing.py#L74) — pull into named constants/config.
12. **Commented-out/dead code blocks**, notably in `utils.py` (reportedly 57+ lines) — remove or explain why kept.
13. **Version history embedded in comments** ([prijstool_preprocessing.py#L5-L17](src/prijstool_preprocessing.py#L5-L17)) listing versions/dates/authors — redundant with git; replace with a changelog or remove.
14. **Inconsistent Dutch/English naming** across files (`bereken_droogterisico`, `Perc_prijspeil_voeding` mixed with English variable names) — pick one convention.
15. **Inconsistent formatting** (tabs vs spaces, spacing) in e.g. [Class_rewrite_correct.py](src/Prijstool_Deltascenarios_Class_rewrite_correct.py) — run a formatter.
16. **No type hints/docstrings** on most functions/classes.

## 🟢 Low

17. Dependencies in `pixi.toml` use loose version ranges (`>=1.0.0.post1,<2`) — fine with `pixi.lock` present, but worth double-checking lock is committed and CI uses it.
18. Leftover note about legacy `.idf` file support requiring an external, undocumented library (`raster_func` from a named person) — clarify if still needed or remove.
