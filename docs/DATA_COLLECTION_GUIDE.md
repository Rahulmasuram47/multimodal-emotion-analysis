# Data Collection Guide — Multimodal Emotion Analysis System

**Team:** Sahithi · Akshitha · Rahul  
**Updated:** 2026-09-26  
**Purpose:** Produce real, ground-truth-annotated WAV recordings for evaluating the multimodal emotion analysis pipeline (Wav2Vec2 + Faster-Whisper + RoBERTa).

---

> [!IMPORTANT]
> **All recordings must be real speech — never fabricated, synthesised, or AI-generated.**  
> Annotation labels must reflect what the recording actually contains, not what was intended.  
> If a recording does not clearly belong to a category, leave that field blank; evaluation code skips blanks.

---

## 1. Recording Setup

| Setting | Required value |
|---|---|
| Format | **WAV** (uncompressed PCM) |
| Sample rate | **16 kHz** |
| Channels | **Mono** (1 channel) |
| Bit depth | 16-bit |
| Duration | **5 – 30 seconds** per clip |
| Environment | Quiet room, no echo, minimal background noise |
| Distance from microphone | 20 – 30 cm |
| Software | Audacity (free), Windows Voice Recorder → export as WAV, or `arecord` on Linux |

**How to verify sample rate in Audacity:**  
Project Rate (bottom-left) → set to 16000 → `File → Export → WAV`.

**File naming convention:**  
`recordings/<scenario_id>_<annotator_initials>_take<n>.wav`  
Example: `recordings/S01_SR_take1.wav` (scenario S01, Sahithi Rao, take 1)

---

## 2. Scenarios

Each scenario targets a specific emotional signal or cross-modal pattern.  
Read the **script** aloud naturally; delivery matters more than word-perfect reading.

> [!NOTE]
> **Labels to assign after recording:** Use the Annotation Helper tab in the app, or fill in `data/evaluation_dataset.csv` directly. Columns: `true_voice_emotion`, `true_text_sentiment`, `true_transcript`, `true_speaker_count`, `true_escalation`, `true_incongruent`.

---

### Group A — Single Emotion, Single Speaker (voice & text agree)

| ID | Emotion target | Script | true_voice_emotion | true_text_sentiment |
|---|---|---|---|---|
| S01 | **Happy / Positive** | *"We just got the results back and I can't believe it — we actually did it! Everything turned out perfectly, way better than we expected. I'm so happy right now."* | happy | positive |
| S02 | **Angry / Negative** | *"I've explained this three times already and it keeps happening again. This is completely unacceptable and I'm done trying to explain myself."* | angry | negative |
| S03 | **Sad / Negative** | *"She didn't come today. I kept checking the door but she never came. I don't know what happened — I just feel so lost without her."* | sad | negative |
| S04 | **Neutral / Neutral** | *"The meeting is at three in the afternoon in the conference room on the second floor. Please bring your laptop and the project files."* | neutral | neutral |
| S05 | **Calm Negative words** | *"I must point out that the last deliverable was inadequate and this situation is not acceptable to the team."* (calm, flat voice) | neutral | negative |

---

### Group B — Cross-Modal Mismatch (voice and text disagree)

These clips test the incongruence detector. Speak with the vocal emotion indicated, not the words.

| ID | Voice delivery | Script | true_voice_emotion | true_text_sentiment | true_incongruent |
|---|---|---|---|---|---|
| S06 | **Angry voice, polite words** | *"I appreciate the effort you've put into this and I respect your opinion."* (deliver with anger and frustration in your voice) | angry | positive | yes |
| S07 | **Happy voice, negative words** | *"The project failed completely and nobody cared about fixing it."* (deliver with a cheerful, upbeat tone) | happy | negative | yes |
| S08 | **Sad voice, positive words** | *"I'm so excited about this new opportunity! Everything is going great for us!"* (deliver slowly, with sadness) | sad | positive | yes |
| S09 | **Neutral voice, emotional words** | *"I hate every single part of this. It is an absolute disaster."* (flat, emotionless delivery) | neutral | negative | yes |

---

### Group C — Escalation Sequences (3-turn, single speaker)

Record each turn as a separate WAV clip named `_turn1`, `_turn2`, `_turn3`.  
Escalation means emotional intensity rises from turn 1 → 3.

| ID | Pattern | Turn 1 | Turn 2 | Turn 3 | true_escalation |
|---|---|---|---|---|---|
| S10 | **Escalating anger** | *"Could you please check the report?"* (neutral, calm) | *"I asked for this report an hour ago and it still isn't done."* (slightly irritated) | *"This is completely unacceptable! Every single time I ask for something, it gets ignored!"* (clearly angry) | yes |
| S11 | **De-escalation** | *"I'm really frustrated — this keeps breaking every day."* (angry) | *"OK, I understand it's a complicated issue. Let's try to figure it out together."* (calming) | *"Thanks for explaining that. I think we have a path forward now."* (neutral, resolved) | no |
| S12 | **Happy escalation** | *"We've made some good progress this week."* (mildly pleased) | *"The results are really impressive — the team has done great work!"* (more enthusiastic) | *"This is amazing! I am so proud of what everyone has achieved here!"* (very happy) | yes |

---

### Group D — Multi-Speaker Dialogue (2 speakers)

Record with **two different speakers** in the same WAV file, taking turns.

| ID | Pattern | Speaker A script | Speaker B script | true_speaker_count | true_escalation |
|---|---|---|---|---|---|
| S13 | **2-speaker neutral exchange** | *"The data pipeline ran successfully last night."* | *"That's great. Did the validation tests all pass?"* → A: *"Yes, everything looks clean."* | 2 | no |
| S14 | **2-speaker disagreement** | *"I think we should delay the deadline by two weeks."* | *"That's not possible — the client needs this by Friday. We can't push it back."* → A: *"If we don't, the quality will suffer."* → B: *"Then we work extra hours this week."* | 2 | yes |
| S15 | **2-speaker supportive** | *"I'm really worried about the presentation tomorrow."* | *"Don't be — you've prepared well and you know this material. You're going to do great."* → A: *"Thank you, that actually helps."* | 2 | no |

---

### Group E — Difficult / Edge Cases

| ID | Pattern | Script | Notes |
|---|---|---|---|
| S16 | **Noisy background** | Read S04 again but play light cafe/street noise in the background. | tests STT robustness; do NOT assign emotion labels you aren't sure about |
| S17 | **Very short clip (5 s)** | *"Everything's fine. Don't worry about it."* (flat delivery) | tests minimum-duration handling |
| S18 | **Very long clip (25–30 s)** | Read S10_turn1 + S10_turn2 + S10_turn3 as one continuous monologue with no pause. | tests segment splitting |
| S19 | **Whispered speech** | Whisper S04 script. | tests STT on low-amplitude input |
| S20 | **Crying / distressed voice** | *"I don't know what to do anymore. I've tried everything and nothing is working."* (genuinely distressed delivery) | extreme sadness; annotate only if clear |
| S21 | **Code-switching / Hindi-English** | *"Yeh kaam bahut important hai. We need to finish this by tomorrow."* | STT stress test; WER will differ by language segment |
| S22 | **Fast speech** | Read S04 at fast pace (~2× normal speed). | tests Whisper segmentation |
| S23 | **Sarcastic tone (positive words, ironic delivery)** | *"Oh sure, that was a great decision. Very smart indeed."* (dripping sarcasm) | annotate as `true_incongruent = yes`; note the system does NOT claim to detect sarcasm |
| S24 | **3-speaker conversation** | Three people take turns: A: *"Let's review the test results."* B: *"The first module passed."* C: *"But module two had three failures."* A: *"We'll need to patch that today."* | true_speaker_count = 3 |
| S25 | **Silence-heavy clip** | 3 seconds silence → *"I... I'm not sure."* → 3 seconds silence. | tests silence handling; duration ~9 s |

---

## 3. Recording Tips

- **Warm up your voice** with one full read-through before the take.
- **Do 2–3 takes** per scenario; keep the best one (consistent emotion throughout).
- **Avoid clipping:** check the recording meter stays below 0 dBFS (no red).
- **Breath pops on B/P/T:** use a pop filter or move slightly off-axis.
- **Check playback** before saving — discard takes where you broke character mid-clip.
- **Don't pad:** if the script is done, stop. Silence at the end is fine; 30+ seconds is too long.
- For **multi-speaker clips (Group D, S24):** sit 0.5–1 m apart, facing the same microphone.

---

## 4. Annotation Protocol

### 4.1 Label Definitions

| Column | Allowed values | Definition |
|---|---|---|
| `true_voice_emotion` | `neutral`, `happy`, `angry`, `sad`, or **blank** | Dominant acoustic/vocal emotion throughout the clip. Blank if unclear or mixed. |
| `true_text_sentiment` | `positive`, `neutral`, `negative`, or **blank** | Overall sentiment of the transcribed words. Blank if no clear text. |
| `true_transcript` | Free text or **blank** | Verbatim transcript (punctuation optional). Blank only if unintelligible. |
| `true_speaker_count` | Integer ≥ 1, or **blank** | Number of distinct speakers. Blank if impossible to determine. |
| `true_escalation` | `yes`, `no`, or **blank** | Does emotional intensity clearly increase from start to end? Blank for single-turn clips. |
| `true_incongruent` | `yes`, `no`, or **blank** | Do voice and text express opposite emotional directions? |

### 4.2 Two-Annotator Protocol

1. **Each clip is annotated independently** by two team members before comparing.
2. Use the **Annotation Helper** tab in the app, or edit `data/evaluation_dataset.csv` directly.
3. After both annotators finish, **compare row-by-row:**
   - **Agreement** → keep the agreed value.
   - **Disagreement on emotion/sentiment** → discuss for ≤ 5 minutes; majority wins.
   - **Persistent disagreement** → leave the cell **blank** (unannotated). Never flip a coin or guess.
4. Record both annotators' initials in the `annotator` column (comma-separated).

### 4.3 Disagreement Resolution Rules

| Type of disagreement | Rule |
|---|---|
| `happy` vs `neutral` | Listen to the energy/pitch. If clearly elevated, use `happy`. Otherwise `neutral`. |
| `angry` vs `sad` | Check for aggression (raised pitch + speed) vs. downcast (slow + low pitch). |
| `positive` vs `neutral` (text) | If any clearly valenced words present, prefer `positive`. |
| `yes` vs `no` for escalation | Requires clear monotonic rise. If only partial, use `no`. |
| `yes` vs `no` for incongruence | Only mark `yes` if both annotators hear clear vocal/textual opposition. |

### 4.4 What NOT to annotate as

- Do **not** use `true_voice_emotion` to describe intended emotion if the recording didn't land.
- Do **not** mark `true_incongruent = yes` for ambiguous clips — leave blank.
- **Never fabricate or guess** a transcript word you couldn't hear. Use `[inaudible]` or leave blank.

---

## 5. File Submission

1. Save each WAV to `recordings/` following the naming convention in §1.
2. Open the **Annotation Helper** tab in the Streamlit app.
3. Select the file, play it back, fill in all fields you are confident in.
4. Click **Save Annotation** — the app writes one row to `data/evaluation_dataset.csv`.
5. Do not manually edit already-annotated rows without using the app's confirmation dialog.

**Backup:** After each session, copy `data/evaluation_dataset.csv` to a shared team folder.

---

## 6. What This Data Is Used For

| Metric | Needs |
|---|---|
| Voice emotion accuracy, macro-F1 | `true_voice_emotion` (≥ 5 annotated samples) |
| Text sentiment accuracy, macro-F1 | `true_text_sentiment` + `true_transcript` |
| WER / CER | `true_transcript` |
| Speaker-count MAE | `true_speaker_count` |
| Escalation P/R/F1 | `true_escalation` (multi-turn clips) |
| Incongruence P/R/F1 | `true_incongruent` |

All metrics are computed only when N ≥ 5 annotated samples; below that threshold the UI displays **"Insufficient data (N = …)"** rather than a potentially misleading number.

---

> [!CAUTION]
> **Privacy:** Do not record third parties without consent. Personal audio (outside these scenarios) must never be committed to the repository. Only the CSV rows (no audio files) are committed to git.
