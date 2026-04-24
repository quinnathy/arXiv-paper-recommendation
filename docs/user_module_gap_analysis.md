# User Module Gap Analysis (Based on `final_proj_alg_v1.pdf`)

## Purpose

This document summarizes what is implemented vs. missing in the **user-related** part of the project, based on:

- Product/algorithm notes in `docs/final_proj_alg_v1.pdf`
- Current code in:
  - `user/db.py`
  - `user/profile.py`
  - `user/session.py`
  - `ui/onboarding.py`
  - `ui/daily_feed.py`

---

## What Is Already Implemented

### 1) Cold-start user initialization (multi-vector)

Implemented and working:

- User selects topic tags during onboarding (`ui/onboarding.py`).
- Optional Google Scholar profile import is supported (`pipeline/scholar_parser.py` via onboarding flow).
- `init_user_profile()` in `user/profile.py` builds user centroids from:
  - selected category centroids
  - optional paper embeddings
- `k_u` is constrained to `1-3` (`min(max_k, len(topic_keys))`).
- Centroids are normalized to unit vectors.

This matches the PDF intent: initialize user representation in embedding space at signup.

---

### 2) User preference feedback update (online)

Implemented and working:

- Daily feed records user actions: `like`, `save`, `skip` (`ui/daily_feed.py`).
- Feedback is logged in SQLite (`log_feedback()` in `user/db.py`).
- Nearest user centroid is updated with EMA (`apply_feedback()` in `user/profile.py`):
  - nearest centroid via max dot product
  - weighted signal (+1.0 like, +1.5 save, -0.3 skip)
  - update with `alpha=0.15`
  - renormalization after update
- Updated centroids persist to DB (`update_centroids()`), and are also synced to session state.

This aligns with PDF’s online update idea.

---

### 3) User data persistence layer

Implemented and working:

- SQLite schema exists for:
  - `users` table (`user_id`, `display_name`, `centroids`, `k_u`, `diversity`, timestamps)
  - `feedback` table (interaction log)
- CRUD helpers exist in `user/db.py`:
  - create/get user
  - update centroids
  - log feedback
  - get seen paper IDs

---

### 4) Session lifecycle basics

Implemented and working:

- Session defaults are initialized (`load_or_init_session()`).
- Onboarding completion is reflected in session state (`onboarded` flag).
- In-session centroid updates are supported (`save_centroids_to_session()`).

---

## Gaps / Missing Features

### A) No returning-user login flow in actual UI

Status: **missing in user experience path**

- `login_user()` exists in `user/session.py`.
- But there is no current UI entry point that asks for a user identifier and calls it.
- Result: practical flow is mostly “new onboarding user each time,” unless state persists in same session.

Why this matters:

- weak continuity across browser/device restart
- hard to reuse previously learned profile without re-onboarding

---

### B) No user-profile visualization (2D / over-time evolution)

Status: **missing**

PDF mentions “visualization in 2D” and profile updates “over time.”

Current code:

- updates profile vectors internally
- does not visualize centroid movement or temporal evolution
- does not store per-step centroid history needed for longitudinal plots

Why this matters:

- no inspectability/explainability for personalization behavior
- cannot demonstrate adaptation quality over sessions

---

### C) Limited onboarding provenance storage

Status: **partially missing** (depends on intended scope)

Current `users` table stores final profile state (`centroids`, `k_u`, `diversity`) and name.
It does **not** explicitly persist:

- selected onboarding topics
- scholar URL used (if any)
- seed-paper metadata used at initialization

Why this may matter:

- harder debugging and reproducibility of personalization initialization
- limited user-facing “edit profile” options later

---

## Potential Risks Observed

### 1) Schema evolution fragility

You already encountered this:

- app code expected `users.centroids`
- local DB schema was older and missing that column
- `CREATE TABLE IF NOT EXISTS` does not migrate existing tables

Risk:

- future schema changes can cause runtime failures unless migration logic is added.

---

### 2) Session/DB path coupling edge cases

`load_or_init_session(db_path)` accepts `db_path`, but most DB operations in `user/db.py` rely on module-global `DB_PATH` updated by `init_db()`.
Current flow is okay, but this pattern can become brittle if multiple DB paths are introduced later.

---

## Priority Recommendations

### P0 (highest)

1. Add a simple “Returning user” login path in onboarding:
   - input `user_id`
   - call `login_user()`
   - route directly to daily feed if valid

2. Add schema migration guard:
   - check existing `users` columns at startup
   - add missing columns automatically (or fail with clear remediation message)

---

### P1 (high value, product clarity)

3. Persist onboarding provenance:
   - selected topics
   - scholar source info (or normalized identifier)
   - optional initialization metadata

4. Add profile introspection UI:
   - show current `k_u`, diversity, thread-level stats
   - optionally show top category affinities per centroid

---

### P2 (demo/evaluation enhancement)

5. Implement 2D visualization + timeline:
   - project centroids/papers to 2D (PCA/UMAP)
   - persist centroid snapshots after each feedback event
   - provide “before/after” and trend views

---

## Suggested Acceptance Criteria for “User Module Complete”

- New user can onboard once and receive recommendations.
- Returning user can recover profile via explicit login path.
- Feedback reliably updates nearest centroid and persists to DB.
- DB startup is robust against old schema (migration or explicit bootstrap script).
- At least one diagnostic view exists to inspect user profile state.
- (Optional stretch) 2D temporal profile visualization for evaluation/demo.

---

## Bottom Line

The core user personalization loop is already implemented (cold-start + feedback update + persistence).  
Main gaps are around **returning-user product flow**, **schema migration robustness**, and **profile observability/visualization** that the PDF hints at.
