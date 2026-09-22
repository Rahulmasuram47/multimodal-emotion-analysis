"""
7-step primary analysis workflow stepper card for empty state.
"""

import streamlit as st


def render_workflow_stepper() -> None:
    """Render the 7-step guided workflow card when no audio is analyzed."""
    st.markdown(
        """
        <div class="empty-state-card">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--voice); font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;">
                PRIMARY ANALYSIS WORKFLOW
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--text); margin: 0.4rem 0;">
                Multimodal Voice & Conversation Emotion Analysis Engine
            </div>
            <div style="font-size: 0.82rem; color: var(--muted); max-width: 750px; margin: 0 auto 1.2rem auto;">
                Follow the 7-step analysis workflow below to perform dual-modality acoustic/linguistic inference, decision-level fusion, conflict tracking, and actionable recommendation generation.
            </div>
            <div class="empty-steps-grid" style="grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.6rem;">
                <div class="empty-step-box">
                    <div class="empty-step-num">STEP 01</div>
                    <div class="empty-step-heading">Upload / Record</div>
                    <div class="empty-step-desc">Upload audio file or record via browser microphone.</div>
                </div>
                <div class="empty-step-box" style="border-top: 2px solid var(--voice);">
                    <div class="empty-step-num" style="color: var(--voice);">STEP 02</div>
                    <div class="empty-step-heading">Analyze Audio</div>
                    <div class="empty-step-desc">Click 'ANALYZE AUDIO ▶' to run full ML pipeline.</div>
                </div>
                <div class="empty-step-box" style="border-top: 2px solid var(--text-accent);">
                    <div class="empty-step-num" style="color: var(--text-accent);">STEP 03</div>
                    <div class="empty-step-heading">Review Results</div>
                    <div class="empty-step-desc">Inspect voice emotion, transcript, and sentiment.</div>
                </div>
                <div class="empty-step-box" style="border-top: 2px solid var(--positive);">
                    <div class="empty-step-num" style="color: var(--positive);">STEP 04</div>
                    <div class="empty-step-heading">Acoustic Signals</div>
                    <div class="empty-step-desc">Examine spectrogram, pitch (F0), and RMS energy.</div>
                </div>
                <div class="empty-step-box" style="border-top: 2px solid var(--negative);">
                    <div class="empty-step-num" style="color: var(--negative);">STEP 05</div>
                    <div class="empty-step-heading">Conflict & Escalation</div>
                    <div class="empty-step-desc">Track modality mismatch and emotional escalation.</div>
                </div>
                <div class="empty-step-box" style="border-top: 2px solid var(--voice);">
                    <div class="empty-step-num" style="color: var(--voice);">STEP 06</div>
                    <div class="empty-step-heading">Recommendations</div>
                    <div class="empty-step-desc">Review advisory action steps with safety notices.</div>
                </div>
                <div class="empty-step-box" style="border-top: 2px solid var(--fused);">
                    <div class="empty-step-num" style="color: var(--fused);">STEP 07</div>
                    <div class="empty-step-heading">Export Report</div>
                    <div class="empty-step-desc">Download structured JSON and CSV segment logs.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
