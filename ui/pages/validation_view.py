"""
Multimodal Validation & Empirical Benchmarks page orchestrator (Tab 3).
"""

import streamlit as st

from evaluation.ablation import run_all_ablations
from evaluation.dataset import load_evaluation_dataset


def render_multimodal_validation_section() -> None:
    """Render empirical validation dataset benchmarks and modality ablation table."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">📊 Multimodal Validation & Empirical Benchmarks</div>
            <div class="section-badge">OFFLINE LABELED DATASET EVALUATION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_data = load_evaluation_dataset()
    labelled_count = len(df_data)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("EVALUATION SAMPLES", f"{labelled_count} Audio Files")
    with c2:
        dataset_status = "Available (Active)" if labelled_count > 0 else "Template Only"
        st.metric("DATASET STATUS", dataset_status)
    with c3:
        st.metric("ANNOTATION LEVEL", "4 Emotion Classes")

    st.markdown("### Modality Ablation & Model Comparison")
    st.caption(
        "Empirical evaluation comparing unimodal vs multimodal performance on labelled data. "
        "Missing annotations are marked as 'Pending Data' rather than displaying simulated metrics."
    )

    ablation_results = run_all_ablations()
    if ablation_results:
        import pandas as pd
        df_ablation = pd.DataFrame(ablation_results)
        st.dataframe(df_ablation, use_container_width=True, hide_index=True)
    else:
        st.info("Ablation benchmarks require annotated ground-truth test data.")

    if labelled_count == 0:
        st.info(
            "ℹ️ **Evaluation Dataset Notice**: `data/evaluation_dataset_template.csv` contains unannotated "
            "reference recordings. Populate audio ground-truth emotion labels to display empirical accuracy and confusion matrices."
        )
