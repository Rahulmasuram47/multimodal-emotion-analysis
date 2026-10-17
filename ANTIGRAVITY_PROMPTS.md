1. Repo cleanup (report, stop, apply only after “approved”)

RULES: Streamlit/Python, English only. Plan max 5 lines; read only needed files;
minimal edits; no refactors or full-file dumps. Inspect before assuming. Never
change fusion rules, thresholds, weights, VOICE_CONF_CAP or label maps. Never
print/log/commit secrets. Never fabricate results. Do NOT stage or commit. Final
summary max 5 lines.
BUDGET: git ls-files, grep, ls only; read no file over 100 lines; no model runs;
count files in data/external_datasets, do not list them.

NEVER delete recordings, datasets, caches, evaluation results or sample audio
without my explicit approval, even if unreferenced. Do not delete code or files
merely because they look unused or non-English-related: check imports, dynamic
imports (emotion2vec, funasr, pyannote, torchaudio), config references and tests
first. Preserve the English-only fallback (non-English or repetitive transcript
-> voice-only with a short notice).

STAGE A (read-only): inspect scratch/, scripts/*, .agents/, .claude/, AGENTS.md,
GEMINI.md, PROGRESS.md, PROJECT_SPEC.md, __pycache__, requirements packages with
no import, dead code in app/ and ui/ (vulture if installed, else grep), duplicate
config, tracked files over 5 MB, whether .env or any secret file is tracked (do
not print contents), whether data/external_datasets, sample_audio and recordings
are ignored. Table: path | tracked? | references | risk | verdict (KEEP /
REVIEW / UNTRACK+GITIGNORE / MOVE / DELETE). DELETE only for zero-reference
junk like __pycache__ or empty scratch files. Everything uncertain is REVIEW.
STOP and wait for my message "approved" (I may edit verdicts).

STAGE B (only after approval): apply only approved verdicts; update .gitignore
without breaking existing rules; verify with git check-ignore -v; remove a package
from requirements*.txt only if confirmed unused. Run the full suite once. Do not
purge git history.

2. Repetition collapse and banner fix

RULES: Streamlit/Python, English only. Plan max 5 lines; minimal edits; no
refactors or full-file dumps. Never change fusion rules, thresholds, weights,
VOICE_CONF_CAP or label maps. Mock models; tests need no data, network or HF
token. Max 2 fix attempts per failure. Do NOT stage or commit. Final summary max
5 lines, then wait.
BUDGET: read only the transcript sanitizer, the banner string and their tests; run
only touched tests, then the full suite once.

(a) Reproduce through the real UI sanitizer (no mock): "Thank you. " x40,
"the the the ..." x30 ending in truncated "th", "I don't know what to do " x8,
"... " x12. Find why any displays uncollapsed (truncation before collapse,
min-run threshold, truncated last token, a display path skipping the sanitizer).
Fix minimally: keep the first occurrence, append "… ×N (repetitive transcript)".
Display only; exports and downstream analysis keep raw text. "very, very good" and
"no, no, no" must NOT collapse.
(b) The banner prints "Low Voice Reliability (100%)" while that turn's voice
reliability is 0.50 (seen on neutral_sample.wav). Find what the 100% measures;
show the real reliability or label the number correctly (e.g. class-collapse
score). Text only; do not just rename to hide a calculation bug.
Tests: patterns above, displayed length under 200 chars, marker present, first
occurrence kept, natural repeats untouched, raw unchanged in export, banner number
matches "Why this result".

3. Engineering, security and privacy audit

RULES: Streamlit/Python, English only. Plan max 5 lines; minimal edits; no
refactors. Never change fusion rules, thresholds, weights, VOICE_CONF_CAP or
label maps. Mock models; tests need no data, network or HF token. Run touched
tests, then the full suite once; max 2 fix attempts per failure. Never
print/log/commit secrets. Do NOT stage or commit. Final summary max 5 lines.
HF: keep the existing token setup; never expose the token in UI, logs, errors or
docs; separate auth/gated-access errors from network, missing-file and package
errors; short safe messages, no tracebacks.
BUDGET: grep first, read only matched files; no model runs; fix High/Medium only,
one test each; Low items go to docs/KNOWN_ISSUES.md (max 20 lines).

Audit table: area | finding | severity | evidence | fix.
(a) Privacy: network calls beyond model download, upload type/size/duration
limits, temp files deleted on success and failure, transcripts not logged at
INFO, .env and HF token never logged or tracked.
(b) Friendly messages, app stays usable: corrupt, silent, sub-second, mic denied,
missing HF token, gated-model denial, offline start.
(c) Performance: st.cache_resource, models loaded once, no unbounded memory,
long-audio guard.
(d) Reproducibility: HF revisions pinned (commit hashes), seeds fixed, deps
pinned, Python version stated; whether emotion2vec downloads from HF or
ModelScope; whether Pyannote is actually enabled and needs accepted terms.
(e) Quality: ruff clean, no print() in app code, no dead code, type hints on
public functions in app/.
(f) CI runs lint and tests with no data or token; add ruff to requirements-ci.txt
and the workflow only if missing.
(g) DEMO_MODE exists? yes/no with file:line; do NOT build it.

4A. CREMA-D loader, split and dev-only baseline

RULES: Streamlit/Python, English only. Plan max 5 lines; minimal edits; no
refactors or full-file dumps. Never change fusion rules, thresholds, weights,
VOICE_CONF_CAP or label maps. Mock models in unit tests; tests need no dataset,
network or HF token. Max 2 fix attempts per failure. Never print/log/commit
secrets. Never fabricate results. Datasets, caches, embeddings git-ignored. Do NOT
stage or commit. Final summary max 5 lines, then wait.
HF: keep the existing token setup; record model IDs and revisions, never
credentials; short safe error messages.
BUDGET: new scripts only under scripts/ or evaluation/. Smoke test
--max-per-class 5, then 100, report runtime, then STOP. Never run --full without
my approval. Never re-run a model for a cached clip.

STEP 0 (before any benchmark): inspect config and code to report which voice
backend the app ACTUALLY uses by default (VOICE_BACKEND or equivalent), its model
id and revision, whether emotion2vec and SUPERB are installed and loadable here,
and which one exposes logits and embeddings. Benchmark the backend the app
genuinely uses. Add emotion2vec / SUPERB as comparisons only if both load and the
comparison is valid (same clips, same label mapping, same preprocessing). A
missing or incomparable backend gives a clear message, never an invented number.

DATA: data/external_datasets/crema_d/AudioWAV/ (7,442 clips present; do not
download). Verify the filename convention against the dataset's own docs
(ActorID_Sentence_Emotion_Level.wav; ANG DIS FEA HAP NEU SAD; actors 1001-1091).
1 Loader with one explicit mapping dict to the active backend's labels; report raw
6-class and any collapsed set separately.
2 Split by ACTOR, seed 42, ~60 dev actors, rest test; save
evaluation/results/crema_d_split.json.
3 Resumable runner (--max-per-class N default 100, --full). Cache logits and
embeddings in a git-ignored .npy cache with stable keys (model id/revision,
preprocessing, labels). The runner may cache test-actor outputs but MUST NOT
compute, print or save any test-split metric. Enforce this in code: test
evaluation refuses to run unless evaluation/results/head_decision.json exists
(created in 4B).
4 Dev-only baseline report: accuracy, UAR, macro-F1, per-class recall (sad and
neutral highlighted), confusion matrix, chance (1/6), N clips, N actors on DEV
actors, actor-cluster bootstrap CI. Save evaluation/results/crema_d_dev_baseline.md
and .json with mapping, seed, model ids/revisions, preprocessing, exclusions.
5 Print the dataset licence from its own file, or say it is missing.
6 Tests: filename parsing, label mapping, zero actor overlap, toy cluster
bootstrap, cache keys, the test-metrics refusal when head_decision.json is missing.

4B. Candidate head with dev-only decision (freeze before testing)

RULES: same as 4A (Streamlit/Python, English only, minimal edits, no fusion or
threshold changes, mock models in tests, never fabricate, do NOT stage or commit,
final summary max 5 lines then wait).
BUDGET: new script under evaluation/ only. Reuse the 4A cache; no re-extraction.
DEV actors only. Do not touch, compute or print anything about the test split.

Candidate: logistic regression (small MLP only if justified) on FROZEN embeddings
of the backend the app actually uses, optionally plus prosody features. If that
backend exposes no embeddings, report that and stop with "no candidate".
1 Declare the adoption rule BEFORE running, and write it to
evaluation/results/head_decision.json as "rule": candidate is promising only if,
in actor-grouped CV on dev actors (GroupKFold by actor, fixed seed), its mean
UAR beats the dev baseline UAR by at least 0.03 AND beats it in a majority of
folds AND neither sad nor neutral recall drops versus baseline. Hyperparameters
selected inside the dev folds only (nested or fixed small grid).
2 Run the CV, then write the outcome into head_decision.json: decision
(promising / not promising), fold-level UAR and per-class recall, hyperparameters,
timestamp, data hash of the dev manifest.
3 If promising: train the final head on ALL dev actors with the frozen
hyperparameters, save the artifact (joblib or json) and a model card (data, split,
CV results, limits). It is loaded only when VOICE_HEAD_PATH is set; default OFF.
If not promising: no artifact, baseline stays.
4 Freeze: after head_decision.json is written, the model, hyperparameters and
decision must not be changed. State that clearly. Any re-tuning needs a fresh
dev-only decision and a new file, never a test-informed edit.
5 The 8 self-recorded clips are a sanity check only (report, never a gate; too
small to resolve differences).
6 Tests on toy embeddings: grouped CV has zero actor leakage; decision rule on
toy fold scores; head loads only when the flag is set.

4C. One-shot test-split evaluation

RULES: same as 4A. Final summary max 5 lines, then wait.
BUDGET: run only after head_decision.json exists and I say "run the test". Reuse
the cache. Evaluate the test split exactly ONCE; no reruns after seeing results,
no tuning, no changing mapping, head or decision.

Evaluate on the held-out test actors: the baseline (the backend the app actually
uses) and, only if head_decision.json says promising, the frozen head. Report
accuracy, UAR, macro-F1, per-class precision/recall (sad and neutral highlighted),
confusion matrix, chance (1/6), N clips, N actors, actor-cluster bootstrap 95% CI,
and a paired actor-cluster bootstrap CI for the UAR difference between candidate
and baseline. Include SUPERB/emotion2vec comparisons only if valid and available.
Report honestly even if the head is worse; if so, say it was not adopted.
Write evaluation/results/crema_d_benchmark.md and .json with mapping, seed,
model ids/revisions, preprocessing, exclusions, and "acted speech, actor-disjoint
split, not natural speech". If --full was not run, state the cap and that clips
were seeded-sampled. Set VOICE_HEAD_PATH default only per the frozen decision and
my explicit approval.

5A. Priority analysis: errors, calibration (if usable), latency

RULES: Streamlit/Python, English only. Plan max 5 lines; minimal edits; no
refactors. Never change app behaviour, fusion rules, thresholds, weights,
VOICE_CONF_CAP or label maps. Mock models in unit tests; tests need no dataset.
Max 2 fix attempts per failure. Never fabricate results. Do NOT stage or commit.
Final summary max 5 lines, then wait.
BUDGET: reuse the 4A/4C cache and results; no re-extraction; no model re-runs
except the latency script, which runs only after I confirm.

1 Error analysis (priority): from the 4C results, the confusion matrix, top 5
confusions (count, rate), per-class precision/recall, recall by CREMA-D level
(LO/MD/HI/XX after validating the filename field). docs/ERROR_ANALYSIS.md max 60
lines; every number from saved results; untested explanations marked "hypothesis".
2 Calibration (only if the active backend exposes usable logits or scores; else
write "not applicable: no usable scores" and skip): fit one temperature T on DEV
logits by minimizing NLL; report ECE (10 bins) and NLL before/after on TEST with
actor-cluster bootstrap CI; state that argmax and accuracy are unchanged, so
calibration does not fix poor classification. Save T and reliability bins to
evaluation/results/calibration.json. Show where VOICE_CONF_CAP sits on the
reliability curve; recommend only, never change it.
3 Latency: scripts/benchmark_latency.py. Cold start (fresh process: model load +
first clip) vs warm (median of 5), RTF (processing seconds per audio second) for
~5/15/30 s clips, per ACTIVE stage only (ASR, voice, text, fusion, diarization
only if a token exists). evaluation/results/latency.md with CPU, RAM, OS, Python,
torch version, GPU yes/no; do not imply GPU numbers from a CPU run.
4 Toy tests: T recovery, ECE ~0 on calibrated data, RTF arithmetic.

5B. Optional: synthetic-noise robustness (run only if time and CPU allow)

RULES: same as 5A. Run only after I say "go". BUDGET: seeded subset max 20
clips/class (cut to 10 if CPU time would exceed ~30 min and say so); reuse
caches; never overwrite clean results.
Conditions: clean, white noise at SNR 20/10/0 dB, 8 kHz down-then-up round trip.
Report UAR with actor-cluster CI and drop vs clean. Label "synthetic noise, not
real environments". Write evaluation/results/robustness.md/.json. Toy test: SNR
mixing within 0.5 dB.

6. Natural-speech evaluation tooling (build tooling now; collect when ready)

RULES: Python, English only. Plan max 5 lines; minimal edits. No app or model
changes. Mock models; tests need no audio. Never fabricate results. Do NOT stage
or commit. Final summary max 5 lines.
BUDGET: new files only under evaluation/natural_set/ and
docs/NATURAL_SET_PROTOCOL.md; toy-CSV tests only; the scorer exits with a
friendly message when no label files exist. This is tooling only. Do not claim the
evaluation is done until real clips are annotated and scored.

1 Protocol (max 50 lines): why it exists (acted corpora do not show natural-speech
accuracy); 40-60 clips, 3-15 s, >=8 consenting speakers incl. Indian English;
speakers describe a real recent event (not acting); consent text; anonymous names
spk01_001.wav; no names or unnecessary personal data; labels
neutral/happy/sad/angry/other mapped to the model's labels only after reviewing
what it can predict; >=2 raters label independently from audio alone without
model output or speaker intent; label CSV template; speaker "intent" stored
separately (intent is not ground truth for expressed emotion).
2 make_label_sheet.py: shuffled per rater, hides intent and model output.
3 score.py: Cohen's kappa (2 raters) or Fleiss (3+) with bootstrap CI, raw
agreement, consensus (majority, ties = "disputed"), model accuracy/UAR/macro-F1 vs
(a) consensus where raters agree, (b) all consensus, (c) intent, each with 95% CI,
N per class, confusion matrix, per-rater agreement with the model. Report
human-human agreement as context, not a perfect ceiling; explain why the three
comparisons differ. Write evaluation/results/natural_set.md and .json. Never pool
with CREMA-D or the 8-clip set.
4 Tests: known kappa values, tie handling.
5 .gitignore the audio folder; anonymised label CSVs committed only if consent
allows.

7. DEFERRED: RAVDESS (do not run unless I explicitly ask)

Skip this phase. If later requested: data/external_datasets/ravdess/ must exist
(do not download); test-only cross-corpus evaluation of the frozen baseline/head
under the same rules as 4C (no tuning, one explicit label map, report exclusions,
actor-cluster CI, licence from the dataset's own file, labelled "acted speech,
cross-corpus; not natural speech").

8. Streamlit deployment and web hygiene

RULES: Keep Streamlit; no framework migration. English only. Plan max 5 lines;
minimal edits; no redesign. Never change fusion rules, thresholds, weights or
VOICE_CONF_CAP. Mock models; tests need no data or token. Max 2 fix attempts per
failure. Never print/log/commit secrets. Do NOT stage or commit. Final summary max
5 lines.
HF: token server-side only (host secret manager or env); never in UI, logs, errors
or docs; safe messages for auth, gated-access and download failures.
BUDGET: config, UI and pages only. Check host RAM, disk, upload limit, sleep
behaviour and model download size BEFORE choosing a host (verify current limits;
do not assume Streamlit Community Cloud fits Whisper + Pyannote + emotion2vec).
A minimal Dockerfile only if the host requires it. No analytics, CAPTCHA or
cookie banner unless something sets tracking cookies. Do not claim a deployed URL
until opened and one analysis run end to end.

1 Privacy and Limitations pages in the existing Streamlit page structure, linked
visibly; claims verified from code; educational/research use only: not clinical,
hiring, surveillance or risk scoring.
2 st.set_page_config title, icon, short description; one primary action "Analyze
a clip".
3 Document that HTTPS comes from the host; document gated-model setup (Pyannote
terms) without secrets.
4 Alt text/captions, contrast in both themes (WCAG AA 4.5:1), narrow-width layout;
fix tokens and overflow.
5 Upload validation (type, size, duration, empty, corrupt) with friendly messages
and a test; no tracebacks in the UI.
6 Link checker over README/docs; fix broken links.
7 Sitemap, robots.txt, Open Graph, custom 404 only if the host supports them;
otherwise note not applicable in docs/KNOWN_ISSUES.md.
8 Measure first-load and inference time and state the machine or host.

9. Project documentation (only missing files)

RULES: docs only; no code changes. English only. Plan max 5 lines. Every claim
verifiable in the repo (grep before claiming); mark anything not run as "not yet
measured". Do NOT stage or commit. No LICENSE file. Final summary max 5 lines.
BUDGET: Test-Path each file first and create only the missing ones; update
PROGRESS.md with one short section; read only what each claim needs.

1 docs/TEST_PLAN.md: scenario | input | expected (from code) | pass/fail. English
happy/sad/neutral/angry; a CREMA-D clip per emotion; noisy; silent; 1-second;
sub-second ("too short"); corrupt; oversized; mic denied; two-speaker; missing HF
token; HF_HUB_OFFLINE=1; both themes; JSON/CSV export keeps raw transcript;
repetition collapse; no "<div" text on page; fresh clone + install + pytest with
no data; latency spot check (cold vs warm); robustness spot check only if 5B ran.
2 docs/DEMO_SCRIPT.md: 5-minute flow; warm up with one clip first (first run took
90 s, later 20-50 s); fallback DEMO_MODE only if it exists in code, else a
pre-recorded screen capture.
3 docs/VIVA_QA.md: code-grounded answers: each active model and preprocessing,
every formula, intensity, evidence strength, confidence cap and why, why the
sanity rule is SUPERB-only, ablation, tiny labelled set, why CREMA-D with an
actor-disjoint split, why cluster bootstrap, why a head on frozen embeddings, and
the corrected protocol (dev-only grouped CV decides, decision frozen before one
test evaluation, results reported even if worse), acted vs natural speech, why
calibration does not change accuracy, why the cap was not changed, kappa and
human agreement as context, natural set collection and why small, latency/RTF,
no sarcasm or deception claims, abstention, voice-only fallback, short-segment
exclusion, privacy, deployment, why RAVDESS, robustness (unless run), Docker, XAI
and live streaming were deferred or scoped out, limitations.
4 CHANGELOG.md from git log: v1 milestones, v2 phases, NOT done (RAVDESS,
robustness unless run, live streaming, explainability, Docker, large-scale
natural-speech annotation).
5 docs/PRIVACY_AND_ETHICS.md (max 40 lines): data processed and where (grep
network calls), what is stored, recording consent, intended use, prohibited uses.
6 docs/KNOWN_ISSUES.md only if missing: issue, impact, workaround.

10. README and final verification

RULES: README.md only (plus read-only checks); do not run models. English only.
Plan max 5 lines. Describe only what code does (grep before each claim). Do NOT
stage or commit. No badges, no LICENSE file. Never print secret contents. Final
summary max 5 lines.
BUDGET: max 150 lines; results table max 6 rows; link evaluation/results/ files
and docs/ERROR_ANALYSIS.md instead of repeating them.

Numbers only from evaluation/results/ files (voice_benchmark_segments_sanity_on/
off.md, latest_evaluation.json, crema_d_dev_baseline.*, crema_d_benchmark.*,
head_decision.json, calibration.json, natural_set.md, latency.md, robustness.*),
always with N and 95% CI; otherwise "not yet measured". Labels: self-recorded =
"preliminary: self-recorded clips labelled by recording intent, not independent
annotators"; CREMA-D = "acted speech, actor-disjoint test split, N clips from N
actors, cluster-bootstrap 95% CI; not natural speech"; natural set =
"preliminary"; robustness (if run) = "synthetic noise". State which backend the
app actually uses, whether the head was promising per the dev-only decision, and
what the single test evaluation showed (including if it was worse or not
adopted). RAVDESS: "not done". Licences from each dataset's or model's own file;
CREMA-D citation Cao et al., 2014.
Sections: overview and what it does NOT do (no sarcasm or deception detection;
not clinical, hiring or risk tooling); mermaid architecture (audio, VAD+Whisper,
segments, voice backend + prosody, text sentiment with ASR guards, fusion rules,
intensity v2, verdict and gauge, UI and exports; diarization in parallel); voice
backend and VOICE_CONF_CAP; fusion rule IDs from app/fusion.py; gauge and
Intensity v2 formulas with weights from config; model table (id from config,
purpose, licence); results; datasets and licences; limitations; how to run
(install, optional HF token, run, tests, benchmark commands, DEMO_MODE only if it
exists); future work; privacy and disclaimer.
Limitations must include: tiny self-recorded set; acted corpora do not show
natural-speech accuracy; sanity heuristic SUPERB-only; emotion2vec confidence
saturates so VOICE_CONF_CAP is a placeholder and confidence is not a correctness
signal (cite ECE if measured); calibration does not change accuracy; prosody
normalised per recording; segments under VOICE_MIN_SEGMENT_SEC excluded from
trend; English-only text sentiment so non-English is voice-only; diarization
needs an HF token and accepted terms; neutral vs polar counts as a mismatch; sad
speech may be labelled neutral (see per-class recall); emotion2vec labels are
bilingual and the UI shows the English part. Future work: RAVDESS cross-corpus,
robustness, Hindi/Telugu (needs multilingual models and a validated set),
independent annotators, larger natural set, explainability, live streaming,
Docker.

FINAL VERIFICATION (read-only after the README): git status (no secret contents),
full test suite once, lint if configured, app starts with the documented command,
.env/tokens/audio/datasets/caches not in intended changes, still Streamlit.
Report benchmark status, available metrics, limits, deployment status and next
action; do not claim passing tests or a live deployment that was not verified.