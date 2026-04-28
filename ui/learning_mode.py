# ui/learning_mode.py

import streamlit as st

from user.db import get_saved_papers


def _init_workspace_state():
    if "learning_workspace" not in st.session_state:
        st.session_state["learning_workspace"] = []

    if "right_sidebar_collapsed" not in st.session_state:
        st.session_state["right_sidebar_collapsed"] = False


def _enrich_saved_papers(saved_items, index):
    papers = []
    for item in saved_items:
        arxiv_id = item["arxiv_id"]
        paper = next((p for p in index.paper_meta if p["id"] == arxiv_id), None)

        papers.append(
            {
                **item,
                "title": paper.get("title", arxiv_id) if paper else arxiv_id,
                "abstract": paper.get("abstract", "") if paper else "",
                "categories": paper.get("categories", "") if paper else "",
            }
        )

    return papers


def _add_to_workspace(arxiv_id: str, stay_on_tab: str | None = None):
    if arxiv_id not in st.session_state["learning_workspace"]:
        st.session_state["learning_workspace"].insert(0, arxiv_id)

    if stay_on_tab:
        st.session_state["requested_tab"] = stay_on_tab


def _remove_from_workspace(arxiv_id: str):
    st.session_state["learning_workspace"] = [
        pid for pid in st.session_state["learning_workspace"] if pid != arxiv_id
    ]


def _open_in_research_mode(arxiv_id: str):
    st.session_state["active_arxiv_id"] = arxiv_id
    st.session_state["requested_tab"] = "Research Lab"
    st.rerun()


def _render_workspace_action(label: str, key: str):
    if st.button(label, key=key, use_container_width=True):
        st.info(f"{label.title()} will be connected to the workspace AI tools.")


def _render_workspace(paper_lookup):
    workspace_ids = st.session_state["learning_workspace"]
    workspace_papers = [
        paper_lookup[pid] for pid in workspace_ids if pid in paper_lookup
    ]

    st.title("Workspace")
    st.caption(
        "Use summarize, compare, and visualize to learn more about connections between papers."
    )

    if not workspace_papers:
        st.info("Add saved papers from the Papers folder to start a workspace.")
    else:
        for paper in workspace_papers:
            with st.container(border=True):
                title = paper.get("title", paper["arxiv_id"])
                st.markdown(f"**{title}**")
                st.caption(paper["arxiv_id"])

                abstract = paper.get("abstract", "")
                if abstract:
                    st.caption(abstract[:260] + ("..." if len(abstract) > 260 else ""))

                cols = st.columns(3)
                with cols[0]:
                    if st.button(
                        "Remove",
                        key=f"remove_workspace_{paper['arxiv_id']}",
                        use_container_width=True,
                    ):
                        _remove_from_workspace(paper["arxiv_id"])
                        st.rerun()

                with cols[1]:
                    if st.button(
                        "Research",
                        key=f"workspace_research_{paper['arxiv_id']}",
                        use_container_width=True,
                    ):
                        _open_in_research_mode(paper["arxiv_id"])

                with cols[2]:
                    st.link_button(
                        "PDF",
                        f"https://arxiv.org/pdf/{paper['arxiv_id']}",
                        use_container_width=True,
                    )

    st.divider()

    action_cols = st.columns(3)
    with action_cols[0]:
        _render_workspace_action("compare", "workspace_compare")
    with action_cols[1]:
        _render_workspace_action("summarize", "workspace_summarize")
    with action_cols[2]:
        _render_workspace_action("visualize", "workspace_visualize")


def _toggle_right_sidebar():
    st.session_state["right_sidebar_collapsed"] = not st.session_state[
        "right_sidebar_collapsed"
    ]
    st.rerun()


def _inject_right_sidebar_styles():
    is_collapsed = st.session_state["right_sidebar_collapsed"]
    rail_width = "3.75rem" if is_collapsed else "21rem"
    content_padding = "6.25rem" if is_collapsed else "24rem"

    st.markdown(
        """<style>
        .st-key-workspace_right_sidebar {
            position: fixed !important;
            top: 0;
            right: 0;
            width: %s;
            height: 100vh;
            padding: 5.25rem 1rem 1rem;
            background: rgb(240, 242, 246);
            border-left: 1px solid rgba(49, 51, 63, 0.18);
            overflow-y: auto;
            z-index: 999;
        }

        .st-key-workspace_right_sidebar_collapsed {
            padding-left: 0.55rem;
            padding-right: 0.55rem;
        }

        .st-key-workspace_right_sidebar button[kind='secondary'] {
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 0.6rem 1rem !important;
            border-radius: 0.5rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
        }

        .st-key-workspace_right_sidebar div[data-testid='stButton'] button p {
            text-align: left !important;
        }

        .st-key-collapse_right_sidebar button,
        .st-key-expand_right_sidebar button {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        .st-key-collapse_right_sidebar button div[data-testid='stMarkdownContainer'],
        .st-key-expand_right_sidebar button div[data-testid='stMarkdownContainer'] {
            display: flex !important;
            justify-content: center !important;
            width: 100%% !important;
        }

        .st-key-collapse_right_sidebar button p,
        .st-key-expand_right_sidebar button p {
            text-align: center !important;
            margin: 0 auto !important;
            width: 100%% !important;
        }

        .st-key-workspace_right_sidebar div[data-testid='stExpander'] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        .st-key-workspace_right_sidebar div[data-testid='stExpander'] details {
            border: none !important;
        }

        .st-key-workspace_right_sidebar div[data-testid='stExpander'] summary {
            font-size: 1rem !important;
            font-weight: 700 !important;
        }

        .stMainBlockContainer,
        [data-testid='stMainBlockContainer'],
        .block-container {
            padding-right: %s !important;
            padding-left: 2rem !important;
            max-width: none !important;
            width: 100%% !important;
        }

        @media (max-width: 1000px) {
            .st-key-workspace_right_sidebar {
                position: static !important;
                width: 100%%;
                height: auto;
                padding: 1rem 0 0;
                border-left: none;
                background: transparent;
            }

            .stMainBlockContainer,
            [data-testid='stMainBlockContainer'],
            .block-container {
                padding-right: 1rem !important;
                padding-left: 1rem !important;
            }
        }
        </style>"""
        % (rail_width, content_padding),
        unsafe_allow_html=True,
    )


def _render_saved_paper_row(paper, active_tab: str):
    arxiv_id = paper["arxiv_id"]
    title = paper.get("title", arxiv_id)

    st.markdown(f"**{title[:80]}**")

    if active_tab not in {"Workspace", "Research Lab"}:
        return

    is_added = arxiv_id in st.session_state["learning_workspace"]
    cols = st.columns(2)
    with cols[0]:
        if st.button(
            "Added" if is_added else "Add",
            key=f"right_folder_workspace_{arxiv_id}",
            use_container_width=True,
            disabled=is_added,
        ):
            _add_to_workspace(arxiv_id, stay_on_tab=active_tab)
            st.rerun()

    with cols[1]:
        if st.button(
            "Read",
            key=f"right_folder_research_{arxiv_id}",
            use_container_width=True,
        ):
            _open_in_research_mode(arxiv_id)


def _render_right_sidebar(saved_papers, active_tab: str):
    with st.container(key="workspace_right_sidebar"):
        collapsed = st.session_state["right_sidebar_collapsed"]

        if collapsed:
            if st.button("<", key="expand_right_sidebar", help="Expand folders"):
                _toggle_right_sidebar()
            return

        heading, collapse = st.columns([0.78, 0.22], vertical_alignment="center")
        with heading:
            st.subheader("Folders")
        with collapse:
            if st.button(">", key="collapse_right_sidebar", help="Collapse folders"):
                _toggle_right_sidebar()

        with st.expander(f"Papers ({len(saved_papers)})", expanded=True):
            if not saved_papers:
                st.caption("No saved papers yet.")
            else:
                for paper in saved_papers:
                    _render_saved_paper_row(paper, active_tab)
                    st.divider()

        with st.expander("Notes (0)", expanded=True):
            st.caption("No notes yet.")


def render_learning_mode(index):
    _init_workspace_state()

    user_id = st.session_state["user_id"]
    saved = get_saved_papers(user_id)
    saved_papers = _enrich_saved_papers(saved, index)
    paper_lookup = {paper["arxiv_id"]: paper for paper in saved_papers}

    _render_workspace(paper_lookup)


def render_workspace_sidebar(index, active_tab: str):
    _init_workspace_state()
    _inject_right_sidebar_styles()

    user_id = st.session_state["user_id"]
    saved = get_saved_papers(user_id)
    saved_papers = _enrich_saved_papers(saved, index)

    _render_right_sidebar(saved_papers, active_tab)
