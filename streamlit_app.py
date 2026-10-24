"""
Streamlit App — JanSeva AI Government Scheme Eligibility & Discovery Portal.

Combines every module into a modern, accessible civic-tech application:
  Module 1  schema.py, scrapers/       -> data/sample_schemes.json loaded as corpus
  Module 3  rule_engine/                -> deterministic eligibility decisions
  Module 4  nlp_extraction/              -> free text -> UserProfile
  RAG       retrieval/                    -> ChromaDB + TF-IDF semantic search
  Module 5  orchestration/                -> LangGraph turn-by-turn state machine
  Module 6  explanation/                   -> decision -> natural language
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Dict, List, Optional

import streamlit as st

from schema import Scheme, SchemeCategory
from explanation.generator import LLMExplanationGenerator, TemplateExplanationGenerator, anthropic_llm_call
from orchestration.graph import build_graph, submit_message
from orchestration.state import new_conversation_state
from retrieval.retriever import SchemeRetriever
from rule_engine.engine import EligibilityStatus

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_schemes.json")
SCRAPED_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "myscheme_schemes.json")

STATUS_DISPLAY = {
    EligibilityStatus.ELIGIBLE: ("✅", "Eligible", "#059669", "Eligible — You meet the verified criteria"),
    EligibilityStatus.NOT_ELIGIBLE: ("❌", "Not eligible", "#dc2626", "Not Eligible — Does not meet current requirements"),
    EligibilityStatus.PARTIALLY_ELIGIBLE: ("🟡", "Need more info", "#d97706", "Need More Info — Additional details required"),
    EligibilityStatus.NEEDS_MANUAL_REVIEW: ("🔎", "Needs manual review", "#2563eb", "Manual Review — Specific criteria need verification"),
}

CATEGORY_ICONS = {
    "agriculture": "🌾",
    "education": "🎓",
    "higher-education": "🏛️",
    "housing": "🏠",
    "rural-housing": "🏡",
    "pension": "👵",
    "insurance": "🛡️",
    "business-loan": "💼",
    "entrepreneurship": "🚀",
    "savings": "💰",
    "skill-development": "⚡",
    "social-security": "🤝",
    "social-welfare": "🤲",
    "employment": "👷",
    "energy": "⚡",
    "healthcare": "🏥",
    "women_and_child": "👩‍👧",
    "disability": "♿",
    "other": "📋",
}


@st.cache_resource
def load_schemes() -> list[Scheme]:
    paths = [DATA_PATH]
    if os.path.exists(SCRAPED_DATA_PATH):
        paths.append(SCRAPED_DATA_PATH)

    schemes_by_source_url = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for item in json.load(f):
                scheme = Scheme(**item)
                schemes_by_source_url[str(scheme.source_url)] = scheme
    return list(schemes_by_source_url.values())


@st.cache_resource
def get_retriever() -> SchemeRetriever:
    return SchemeRetriever.from_schemes(load_schemes(), default_top_k=5)


@st.cache_resource
def get_graph_and_mode():
    retriever = get_retriever()
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            generator = LLMExplanationGenerator(anthropic_llm_call())
            return build_graph(generator, retriever=retriever), "llm"
        except Exception:
            pass
    return build_graph(TemplateExplanationGenerator(), retriever=retriever), "template"


def init_session_state():
    if "conversation_state" not in st.session_state:
        schemes = load_schemes()
        st.session_state.conversation_state = new_conversation_state(schemes, max_turns=6)
        st.session_state.chat_history = [
            ("assistant", "Hi! Tell me a bit about yourself — age, income, "
                          "occupation, state — and I'll check which schemes you may qualify for.")
        ]


def reset_conversation():
    for key in ("conversation_state", "chat_history"):
        st.session_state.pop(key, None)
    init_session_state()


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Modern Civic Portal Styling */
        .civic-header {
            background: linear-gradient(135deg, #091e3a 0%, #173866 50%, #1e4976 100%);
            border-radius: 16px;
            padding: 22px 28px;
            color: #ffffff;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px -4px rgba(10, 30, 60, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .civic-header h1 {
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0;
            color: #ffffff !important;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .civic-header p {
            margin: 6px 0 0 0;
            font-size: 0.95rem;
            color: #d1d5db;
        }
        .civic-badge-container {
            display: flex;
            gap: 10px;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        .civic-pill {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(8px);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            color: #f3f4f6;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .civic-pill-green {
            background: rgba(16, 185, 129, 0.25);
            color: #6ee7b7;
            border-color: rgba(16, 185, 129, 0.4);
        }

        /* Profile Card in Sidebar */
        .citizen-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(156, 163, 175, 0.2);
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 14px;
        }
        .profile-chip-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
            margin-top: 8px;
        }
        .profile-chip {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(125, 125, 125, 0.08);
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 0.85rem;
        }
        .profile-chip-key {
            color: #94a3b8;
            font-weight: 500;
        }
        .profile-chip-val {
            font-weight: 600;
        }

        /* Quick Persona Chips */
        .persona-container {
            background: rgba(59, 130, 246, 0.06);
            border: 1px dashed rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 16px;
        }
        .persona-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: #2563eb;
            margin-bottom: 6px;
        }

        /* Metric Highlights */
        .metric-card {
            border-radius: 12px;
            padding: 12px 16px;
            text-align: center;
            border: 1px solid rgba(150, 150, 150, 0.15);
            background: rgba(125, 125, 125, 0.04);
        }
        .metric-card-num {
            font-size: 1.5rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-card-label {
            font-size: 0.78rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }

        /* Scheme Card Details */
        .benefit-box {
            background: rgba(16, 185, 129, 0.08);
            border-left: 4px solid #10b981;
            padding: 10px 14px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 0.92rem;
        }
        .doc-tag {
            display: inline-block;
            background: rgba(100, 116, 139, 0.12);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            margin: 2px 4px 2px 0;
        }
        .apply-btn {
            display: inline-block;
            background: #2563eb;
            color: white !important;
            padding: 7px 16px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.88rem;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(graph, mode):
    with st.sidebar:
        # Keep button[0] in sidebar as Start over
        if st.button("Start over", key="sidebar_reset_btn", use_container_width=True, type="secondary"):
            reset_conversation()
            st.rerun()

        st.markdown("### 👤 Citizen Profile")
        profile = st.session_state.conversation_state["profile"]
        known = profile.model_dump(exclude_none=True)

        # Profile Completeness Indicator
        tracked_fields = ["age", "annual_income", "state", "occupation", "gender", "caste_category"]
        filled_count = sum(1 for f in tracked_fields if getattr(profile, f, None) is not None)
        completeness_pct = int((filled_count / len(tracked_fields)) * 100)
        
        st.progress(completeness_pct / 100, text=f"Profile Completeness: {completeness_pct}%")

        # Visual Card Presentation
        def format_val(val, prefix="", suffix=""):
            if val is None:
                return "<span style='color: #94a3b8; font-style: italic;'>Not specified</span>"
            if isinstance(val, int):
                return f"{prefix}{val:,}{suffix}"
            return f"{prefix}{str(val).title()}{suffix}"

        profile_html = f"""
        <div class="citizen-card">
            <div class="profile-chip-grid">
                <div class="profile-chip">
                    <span class="profile-chip-key">🎂 Age</span>
                    <span class="profile-chip-val">{format_val(profile.age, suffix=' yrs')}</span>
                </div>
                <div class="profile-chip">
                    <span class="profile-chip-key">💰 Annual Income</span>
                    <span class="profile-chip-val">{format_val(profile.annual_income, prefix='₹')}</span>
                </div>
                <div class="profile-chip">
                    <span class="profile-chip-key">📍 State</span>
                    <span class="profile-chip-val">{format_val(profile.state)}</span>
                </div>
                <div class="profile-chip">
                    <span class="profile-chip-key">💼 Occupation</span>
                    <span class="profile-chip-val">{format_val(profile.occupation)}</span>
                </div>
                <div class="profile-chip">
                    <span class="profile-chip-key">🚻 Gender</span>
                    <span class="profile-chip-val">{format_val(profile.gender.value if profile.gender else None)}</span>
                </div>
                <div class="profile-chip">
                    <span class="profile-chip-key">🏷️ Caste Category</span>
                    <span class="profile-chip-val">{format_val(profile.caste_category.value.upper() if profile.caste_category else None)}</span>
                </div>
            </div>
        </div>
        """
        st.markdown(profile_html, unsafe_allow_html=True)

        # Keep raw st.json for tests and developer inspection
        st.write("**What we know about you:**")
        st.json(known if known else {"(nothing yet)": ""})

        st.divider()

        # Session & System Info
        st.markdown("### ⚙️ Session Status")
        turn_count = st.session_state.conversation_state["turn_count"]
        max_turns = st.session_state.conversation_state["max_turns"]
        st.write(f"🔄 **Turn:** `{turn_count}` / `{max_turns}`")
        mode_label = "🤖 Anthropic Claude LLM" if mode == "llm" else "⚡ Template Engine (Zero API delay)"
        st.write(f"Explanation mode: **{'LLM (Anthropic)' if mode == 'llm' else 'Template (offline)'}**")

        st.divider()

        # Schemes in this demo
        all_schemes = load_schemes()
        candidates = st.session_state.conversation_state.get("candidate_schemes", [])
        candidate_ids = {s.scheme_id for s in candidates} if candidates else set()

        st.subheader("Schemes in this demo")
        for s in all_schemes:
            if turn_count > 0 and s.scheme_id not in candidate_ids:
                st.caption(f"·  ~~{s.name}~~  _(not retrieved this turn)_")
            else:
                st.caption(f"•  {s.name}")
        if turn_count > 0:
            st.caption(
                f"RAG retrieved {len(candidate_ids)} of {len(all_schemes)} schemes as "
                f"candidates this turn (TF-IDF + ChromaDB, ranked against everything "
                f"you've said so far)."
            )


def render_results(state: dict):
    results = state.get("results", [])
    if not results:
        st.info("No eligibility results generated yet. Share your details above to begin!")
        return

    all_schemes = {s.scheme_id: s for s in load_schemes()}

    # Compute tally metrics
    eligible_count = sum(1 for r in results if r.status == EligibilityStatus.ELIGIBLE)
    partial_count = sum(1 for r in results if r.status == EligibilityStatus.PARTIALLY_ELIGIBLE)
    review_count = sum(1 for r in results if r.status == EligibilityStatus.NEEDS_MANUAL_REVIEW)
    not_eligible_count = sum(1 for r in results if r.status == EligibilityStatus.NOT_ELIGIBLE)

    # Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid #059669;">
            <div class="metric-card-num" style="color: #059669;">{eligible_count}</div>
            <div class="metric-card-label">Eligible</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid #2563eb;">
            <div class="metric-card-num" style="color: #2563eb;">{review_count}</div>
            <div class="metric-card-label">Manual Review</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid #d97706;">
            <div class="metric-card-num" style="color: #d97706;">{partial_count}</div>
            <div class="metric-card-label">Need Info</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid #dc2626;">
            <div class="metric-card-num" style="color: #dc2626;">{not_eligible_count}</div>
            <div class="metric-card-label">Ineligible</div>
        </div>""", unsafe_allow_html=True)

    st.write("")

    # Filter selector
    filter_option = st.radio(
        "Filter results:",
        ["All Schemes", "✅ Eligible Only", "🔎 Manual Review Required", "🟡 Need More Info", "❌ Ineligible"],
        horizontal=True,
        label_visibility="collapsed",
    )

    filtered_results = []
    for r in results:
        if filter_option == "✅ Eligible Only" and r.status != EligibilityStatus.ELIGIBLE:
            continue
        if filter_option == "🔎 Manual Review Required" and r.status != EligibilityStatus.NEEDS_MANUAL_REVIEW:
            continue
        if filter_option == "🟡 Need More Info" and r.status != EligibilityStatus.PARTIALLY_ELIGIBLE:
            continue
        if filter_option == "❌ Ineligible" and r.status != EligibilityStatus.NOT_ELIGIBLE:
            continue
        filtered_results.append(r)

    if not filtered_results:
        st.info(f"No schemes in filter '{filter_option}'.")
        return

    for result in filtered_results:
        icon, label, color_code, subtitle = STATUS_DISPLAY[result.status]
        scheme_obj = all_schemes.get(result.scheme_id)

        # Standard st.expander with the exact expected label format for tests
        with st.expander(f"{icon} **{result.scheme_name}** — {label}", expanded=(result.status == EligibilityStatus.ELIGIBLE)):
            explanation = state["explanations"].get(result.scheme_id, "")
            if explanation:
                st.write(explanation)

            if scheme_obj and scheme_obj.benefits:
                st.markdown(f"""<div class="benefit-box">
                    <strong>💰 Benefits:</strong> {scheme_obj.benefits}
                </div>""", unsafe_allow_html=True)

            with st.popover("Show condition-by-condition detail"):
                for c in result.conditions:
                    c_icon = {
                        "pass": "✅", "fail": "❌",
                        "missing_info": "❓", "needs_review": "🔎",
                    }.get(c.status.value, "•")
                    st.markdown(f"{c_icon} `{c.field}` — {c.message}")

            if scheme_obj:
                c_col1, c_col2 = st.columns([2, 1])
                with c_col1:
                    if scheme_obj.documents_required:
                        st.markdown("**📄 Documents Required:**")
                        doc_html = " ".join([f"<span class='doc-tag'>✓ {doc}</span>" for doc in scheme_obj.documents_required])
                        st.markdown(doc_html, unsafe_allow_html=True)
                with c_col2:
                    if scheme_obj.application_link:
                        st.markdown(
                            f"<a href='{scheme_obj.application_link}' target='_blank' class='apply-btn'>Apply Online ↗</a>",
                            unsafe_allow_html=True,
                        )

    # Download Report Button
    st.divider()
    report_text = generate_report_markdown(state, all_schemes)
    st.download_button(
        label="📥 Download Official Eligibility Summary Report (.md)",
        data=report_text,
        file_name=f"JanSeva_Eligibility_Report_{date.today().isoformat()}.md",
        mime="text/markdown",
        help="Export your tailored scheme assessment for printing or offline use.",
    )


def generate_report_markdown(state: dict, all_schemes: dict) -> str:
    profile = state.get("profile")
    lines = [
        "# 🏛️ JanSeva AI — Government Scheme Eligibility Assessment Report",
        f"*Generated on: {date.today().strftime('%B %d, %Y')}*",
        "",
        "## 👤 Citizen Profile",
        f"- **Age:** {profile.age if profile.age is not None else 'Not specified'}",
        f"- **Annual Family Income:** {'₹' + f'{profile.annual_income:,}' if profile.annual_income is not None else 'Not specified'}",
        f"- **State:** {profile.state.title() if profile.state else 'Not specified'}",
        f"- **Occupation:** {profile.occupation.title() if profile.occupation else 'Not specified'}",
        f"- **Gender:** {profile.gender.value.title() if profile.gender else 'Not specified'}",
        f"- **Caste Category:** {profile.caste_category.value.upper() if profile.caste_category else 'Not specified'}",
        "",
        "---",
        "## 📋 Scheme Eligibility Results",
        "",
    ]
    for result in state.get("results", []):
        icon, label, _, _ = STATUS_DISPLAY[result.status]
        scheme = all_schemes.get(result.scheme_id)
        lines.append(f"### {icon} {result.scheme_name} — {label}")
        if result.scheme_id in state.get("explanations", {}):
            lines.append(f"**Evaluation:** {state['explanations'][result.scheme_id]}")
        if scheme:
            lines.append(f"- **Department:** {scheme.department}")
            lines.append(f"- **Benefits:** {scheme.benefits}")
            if scheme.documents_required:
                lines.append(f"- **Documents Needed:** {', '.join(scheme.documents_required)}")
            if scheme.application_link:
                lines.append(f"- **Official Link:** {scheme.application_link}")
        lines.append("")
    lines.append("---")
    lines.append("*Disclaimer: This evaluation is generated by the JanSeva AI deterministic rule engine. Final eligibility is determined by the respective government department upon formal application.*")
    return "\n".join(lines)


def render_scheme_directory():
    st.markdown("### 📚 Explore Government Schemes Directory")
    st.caption("Search and filter all registered central and state welfare schemes.")

    all_schemes = load_schemes()
    categories = sorted(list({s.category.value if hasattr(s.category, "value") else str(s.category) for s in all_schemes}))

    col_search, col_cat = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍 Search schemes by name or keywords...", placeholder="e.g. Kisan, Awas, Pension, Scholarship, Mudra")
    with col_cat:
        selected_cat = st.selectbox("Filter by Category:", ["All Categories"] + categories)

    filtered = all_schemes
    if selected_cat != "All Categories":
        filtered = [s for s in filtered if (s.category.value if hasattr(s.category, "value") else str(s.category)) == selected_cat]
    if search_query:
        sq = search_query.lower()
        filtered = [s for s in filtered if sq in s.name.lower() or sq in s.description.lower() or sq in s.department.lower()]

    st.write(f"Showing **{len(filtered)}** of **{len(all_schemes)}** schemes:")

    for s in filtered:
        cat_str = s.category.value if hasattr(s.category, "value") else str(s.category)
        icon = CATEGORY_ICONS.get(cat_str, "📋")
        with st.container(border=True):
            st.markdown(f"#### {icon} {s.name}")
            st.caption(f"Category: **{cat_str.title()}** | Department: **{s.department}**")
            st.write(s.description)
            st.markdown(f"""<div class="benefit-box">
                <strong>💰 Benefit Details:</strong> {s.benefits}
            </div>""", unsafe_allow_html=True)

            col_crit, col_doc = st.columns(2)
            with col_crit:
                st.markdown("**Key Criteria:**")
                e = s.eligibility
                if e.age_min is not None or e.age_max is not None:
                    st.write(f"- Age: {e.age_min or 'Any'} to {e.age_max or 'Any'} yrs")
                if e.income_max is not None:
                    st.write(f"- Max Income: ₹{e.income_max:,}/yr")
                if e.state:
                    st.write(f"- States: {', '.join([stt.title() for stt in e.state])}")
                if e.occupation:
                    st.write(f"- Occupations: {', '.join([o.title() for o in e.occupation])}")
                if e.caste_category:
                    st.write(f"- Castes: {', '.join([c.upper() for c in e.caste_category])}")
            with col_doc:
                if s.documents_required:
                    st.markdown("**Required Documents:**")
                    for d in s.documents_required:
                        st.write(f"• {d}")

            if s.application_link:
                st.markdown(f"[🌐 Visit Official Application Portal ↗]({s.application_link})")


def render_transparency_tab():
    st.markdown("### 🛡️ System Architecture & Zero-Hallucination Guarantee")
    st.markdown(
        """
        JanSeva AI is architected specifically to eliminate hallucinations in government eligibility determinations.
        
        #### Why Traditional Chatbots Fail at Citizen Services:
        Standard Large Language Models (LLMs) tend to fabricate or approximate income cutoffs, age brackets, and exclusion clauses. In government welfare, this causes citizens to be falsely encouraged or wrongly disqualified.
        
        #### How JanSeva AI Works:
        ```
        Citizen Message ──▶ [1. NLP Extraction & Spelling Correction]
                                      │ (structured UserProfile)
                                      ▼
                            [2. Semantic RAG Retrieval]
                                      │ (candidate schemes)
                                      ▼
                            [3. Deterministic Rule Engine]
                                      │ (PASS / FAIL / MISSING_INFO)
                                      ▼
                            [4. Transparent Explanation Generator]
        ```
        1. **Deterministic Rule Engine (Module 3):** The eligibility verdict is 100% computed by verified boolean rules in Python.
        2. **Constrained Natural Language (Module 6):** The LLM is strictly confined to explaining the already-computed verdict in clear, friendly citizen language. It is never allowed to determine eligibility.
        3. **RAG Candidate Narrowing:** ChromaDB + TF-IDF index prioritizes relevant schemes as the catalog scales.
        """
    )


def main():
    st.set_page_config(
        page_title="Govt. Scheme Eligibility Chatbot",
        page_icon="🏛️",
        layout="wide",
    )
    inject_custom_css()

    graph, mode = get_graph_and_mode()
    init_session_state()

    # The very first button rendered ensures at.button[0] in AppTest is Start over
    col_banner, col_reset = st.columns([6, 1])
    with col_reset:
        if st.button("Start over", key="main_reset_btn", use_container_width=True, type="secondary"):
            reset_conversation()
            st.rerun()

    # Top Hero Banner
    with col_banner:
        st.markdown(
            f"""
            <div class="civic-header">
                <h1>🏛️ JanSeva AI <span class="civic-pill civic-pill-green">Verified Rule Engine</span></h1>
                <p>Hybrid architecture: rule engine decides eligibility deterministically; an LLM only ever rephrases an already-made decision, never invents one.</p>
                <div class="civic-badge-container">
                    <span class="civic-pill">🛡️ 100% Deterministic Rules</span>
                    <span class="civic-pill">📂 25 Welfare Schemes</span>
                    <span class="civic-pill">⚡ Mode: {'Anthropic Claude LLM' if mode == 'llm' else 'Template Engine (Offline)'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_sidebar(graph, mode)

    # Main Tabbed Interface
    tab_chat, tab_directory, tab_arch = st.tabs([
        "💬 Scheme Eligibility Assistant",
        "📚 Browse Scheme Directory",
        "🛡️ Architecture & Verification",
    ])

    with tab_chat:
        # Quick-Start Persona Chips
        st.markdown(
            """
            <div class="persona-container">
                <div class="persona-title">💡 Quick Start — Click a sample citizen profile to test instantly:</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4 = st.columns(4)
        sample_prompt = None

        with p1:
            if st.button("🌾 Small Farmer (UP)\nAge 48, ₹80k income", use_container_width=True, key="p_farmer"):
                sample_prompt = "I am a 48 year old farmer living in Uttar Pradesh with annual income 80,000"
        with p2:
            if st.button("👵 Senior Citizen (Gujarat)\nAge 65, ₹1 Lakh income", use_container_width=True, key="p_senior"):
                sample_prompt = "I am 65 years old, living in Gujarat, and my annual family income is 1 lakh rupees"
        with p3:
            if st.button("🎓 OBC Student (Maharashtra)\nAge 19, ₹1.5 Lakh income", use_container_width=True, key="p_student"):
                sample_prompt = "I am a 19 year old college student from OBC category in Maharashtra with family income 1.5 lakh rupees"
        with p4:
            if st.button("🏪 Street Vendor (Mudra)\nAge 32, ₹1.2 Lakh income", use_container_width=True, key="p_vendor"):
                sample_prompt = "I am 32 years old, working as an urban street vendor, annual income 1.2 lakh rupees"

        # Render conversation chat history
        for role, content in st.session_state.chat_history:
            avatar = "🏛️" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar):
                st.write(content)

        # Clarification Quick Options (if bot is currently asking for info)
        cur_state = st.session_state.conversation_state
        if cur_state.get("awaiting_clarification") and not cur_state.get("done"):
            clarif_field = cur_state.get("clarification_field", "")
            clarif_q = (cur_state.get("clarification_question") or "").lower()

            if "income" in clarif_field or "income" in clarif_q:
                st.caption("⚡ Quick reply for income:")
                q1, q2, q3, q4 = st.columns(4)
                if q1.button("₹80,000 / year", key="inc_80k"):
                    sample_prompt = "My annual income is 80000 rupees"
                if q2.button("₹1,50,000 / year", key="inc_150k"):
                    sample_prompt = "My annual income is 150000 rupees"
                if q3.button("₹2,50,000 / year", key="inc_250k"):
                    sample_prompt = "My annual income is 250000 rupees"
                if q4.button("₹5,00,000 / year", key="inc_500k"):
                    sample_prompt = "My annual income is 500000 rupees"
            elif "occupation" in clarif_field or "occupation" in clarif_q:
                st.caption("⚡ Quick reply for occupation:")
                o1, o2, o3, o4 = st.columns(4)
                if o1.button("Farmer", key="occ_farmer"):
                    sample_prompt = "I work as a farmer"
                if o2.button("Student", key="occ_student"):
                    sample_prompt = "I am a student"
                if o3.button("Street Vendor", key="occ_vendor"):
                    sample_prompt = "I work as a street vendor"
                if o4.button("Salaried Employee", key="occ_salaried"):
                    sample_prompt = "I work as a salaried private employee"

        # Handle user text submission
        user_input = st.chat_input("Tell me about yourself...")
        effective_input = sample_prompt or user_input

        if effective_input:
            st.session_state.chat_history.append(("user", effective_input))
            state = submit_message(st.session_state.conversation_state, effective_input, graph=graph)
            st.session_state.conversation_state = state

            if state["corrections"]:
                corrections = ", ".join(
                    f"`{original}` as `{corrected}`"
                    for original, corrected in state["corrections"]
                )
                st.session_state.chat_history.append(
                    ("assistant", f"I read {corrections}.")
                )

            if state["awaiting_clarification"]:
                st.session_state.chat_history.append(("assistant", state["clarification_question"]))
            else:
                st.session_state.chat_history.append(
                    ("assistant", "Here's what I found — see the summary below.")
                )
            st.rerun()

        # Render eligibility summary results below chat
        has_results = bool(st.session_state.conversation_state.get("results"))
        is_done = st.session_state.conversation_state.get("done", False)
        has_eligible = any(r.status == EligibilityStatus.ELIGIBLE for r in st.session_state.conversation_state.get("results", []))

        if has_results and (is_done or has_eligible):
            st.divider()
            st.subheader("Latest eligibility summary")
            render_results(st.session_state.conversation_state)

    with tab_directory:
        render_scheme_directory()

    with tab_arch:
        render_transparency_tab()


if __name__ == "__main__":
    main()
