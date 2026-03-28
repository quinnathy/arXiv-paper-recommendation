"""Daily feed page for onboarded users.

Main page showing 3 recommended papers per day with like/save/skip actions.
Handles feedback processing: logging to DB, updating user embedding via EMA,
and persisting the updated embedding.
"""

from __future__ import annotations

from pipeline.index import PaperIndex


def render_daily_feed(index: PaperIndex, db_path: str) -> None:
    """Render the daily paper feed for an onboarded user.

    Page layout:
        1. Header: "Good morning, {name}" + today's date.
        2. Subheader: "Your 3 papers for today".
        3. If today's recommendations not yet generated:
            - Call recommend() with current user_embedding + seen_ids.
            - Store results in st.session_state["todays_recs"].
        4. Render three paper_card() widgets.
        5. Sidebar:
            - User name + "since {created_at}".
            - "Liked papers: N" counter.
            - "Settings" expander with topic re-selection.

    Feedback handling (on_like / on_save / on_skip callbacks):
        - Call log_feedback() to write the interaction to the DB.
        - Call apply_feedback() to compute the updated user embedding via EMA.
        - Call update_embedding() to persist the new embedding to the DB.
        - Call save_embedding_to_session() to update session state.
        - Add the paper's arxiv_id to st.session_state["responded"] (a set)
          to disable buttons for that card (feedback idempotency).
        - If all 3 papers have been responded to, show a message:
          "Come back tomorrow for new recommendations!"

    Edge cases:
        - If recommend() returns fewer than 3 papers, show however many are
          available with a note: "You've seen most papers in your areas."

    Args:
        index: The loaded PaperIndex for recommendation lookups.
        db_path: Path to the SQLite database for feedback logging.
    """
    raise NotImplementedError
