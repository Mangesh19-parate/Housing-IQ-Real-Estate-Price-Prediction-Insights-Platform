# Project Rules
## Project: HousingIQ — India Real Estate Price Prediction & Insights Platform

These are binding working rules for this project — data handling, modeling, engineering, and UX rules that must not be silently violated as the build progresses.

---

## 1. Data & Privacy Rules
1. Never surface dealer/agent contact fields, phone-like fields, or raw photo/media URLs in the UI or in any exported artifact — they are dropped at the cleaning stage (TRD §4.6) and must not reappear via a later join.
2. Raw source CSVs (`/data/raw`) are immutable — all cleaning happens by writing new files to `/data/processed`, never by overwriting raw files in place.
3. Any new derived table (locality_stats, amenity_uplift, etc.) must state its computation date and source dataset version in a header/metadata field, so stale precomputed stats can be identified later.
4. Outlier rows are flagged (`is_outlier`), never silently deleted from the processed store — only excluded from the training subset.

## 2. Modeling Rules
1. Every model must be evaluated with the same protocol: 70/15/15 train/val/test split, `random_state=42`, metrics = R², MAE, RMSE, MAPE, reported on the original price scale (not just log scale).
2. No feature is added to the final model without being justified by at least one method in the feature-selection report (TRD §9) — no "gut feel" features slipped in silently.
3. Any leakage-prone feature (e.g., locality aggregates that include the target row itself) must be computed leakage-safely (train-only fit, or out-of-fold) — this is a hard requirement, not a nice-to-have, given how easy it is to leak price into locality-average features.
4. The model in production must be the exact same `sklearn.Pipeline` object used during evaluation — no "re-implementing" preprocessing separately in the API layer (this is the single most common source of train/serve skew and is explicitly disallowed).
5. Every trained model version is saved with a version suffix (`_v1`, `_v2`, ...) and a paired `metrics_v{n}.json` — never overwrite a previous model file in place.
6. SHAP explanations shown to users must come from the same model instance making the prediction — never a proxy/simplified model, even for speed.

## 3. Recommender Rules
1. The recommender is content-based only in v1. Do not simulate or fabricate collaborative-filtering signal (e.g., fake click/view counts) to make it look more sophisticated — there is no real interaction data, and pretending otherwise would misrepresent the system to users and to anyone reviewing this project.
2. Cold-start fallback results must be visually and functionally distinguished from true similarity-based results in the UI (labeled "Popular in this area," not given a fake similarity score).
3. Recommendations must respect the city (and by default, locality) scope of the seed unless the user explicitly opts into "expand search" — never silently mix cities.

## 4. Insights Module Rules
1. All insight sentences must be generated from precomputed, auditable aggregate tables via templates — never from an unconstrained generative text call. If an LLM is ever introduced for phrasing variety in a future version, its output must still be constrained to only restate numbers pulled from the precomputed tables, and must be checked against those numbers before display.
2. Every insight sentence must be traceable back to a specific row/value in `locality_stats`, `amenity_uplift`, or the relevant trend table — no unsupported claims.
3. If a locality has too few listings (define a minimum sample threshold, e.g. n < 10) to produce a statistically meaningful stat, fall back to city-level aggregates and say so explicitly in the sentence (e.g., "city-wide average" instead of implying locality-level precision).

## 5. Engineering Rules
1. Flask never imports or loads model files directly — all inference goes through the FastAPI service over HTTP. This isolation must be preserved even under time pressure; do not "temporarily" call the model from Flask to save time.
2. If FastAPI is unreachable, Flask must degrade gracefully (friendly message), never surface a raw stack trace or 500 page to the end user.
3. All pipeline stages (clean → EDA offline → engineer → select → train → evaluate → export) run through one reproducible entry point; no ad-hoc notebook-only steps that can't be rerun.
4. All randomness is seeded (`random_state=42`) everywhere it appears (splits, model init, sampling for permutation importance) so results are reproducible run to run.
5. Config values (paths, hyperparameters, feature lists, thresholds like outlier percentiles) live in a config file, not hardcoded inline across multiple scripts.
6. No secrets/credentials are ever committed to the repo (not applicable to model files, but relevant if any deployment config with API keys is added later).

## 6. UI/UX Rules
1. Every async operation (predict, recommend, analytics fetch) must have a visible loading state and a visible failure state — a silent/blank result is never acceptable.
2. Every prediction and every recommendation result must be paired with an explanation element (SHAP chart / matched attributes / insight sentence) — the app must never show a bare number with zero context, per the core design principle in the UI/UX doc.
3. Confidence/outlier warnings (input far outside training distribution) must be shown prominently, not buried in fine print.
4. Color must never be the sole carrier of meaning (e.g., SHAP positive/negative bars always paired with +/− text, not just green/red).

## 7. Documentation & Process Rules
1. Any scope change to the 4 modules, tech stack, or dataset assumptions must be reflected back into the PRD and TRD — these documents are living, not "write once."
2. The Tracker's Decision Log must be updated whenever a non-obvious choice is made (e.g., global vs per-city model) — "we just did it" is not sufficient; the rationale must be written down.
3. Every weekly checkpoint in the Implementation Plan must have a corresponding entry marked Done (or explicitly rescoped) in the Tracker before moving to the next week's tasks.
4. This Rules document itself is enforceable — if a rule is found to be unworkable during implementation, it must be explicitly revised here (with reasoning), not silently ignored.

---

## UPDATE v2 — Classification Module & Research-Integrity Rules

### 8. Classification Module Rules
1. **`price_tier` is derived from `price_per_sqft` and must never be used as an input feature to the price regression model** — doing so is direct target leakage (the model would effectively be told the answer). It may be used as an input feature to the Recommender (a different, non-leaking task) and is always valid as a display/analytics artifact.
2. Tier quantile boundaries must be computed on the **training set only** and re-applied (not recomputed) on validation/test data and at inference time — recomputing boundaries per request would make the tier definition drift and be non-reproducible.
3. Tier thresholds are **per-city**, not global — a "Luxury" Kolkata listing and a "Luxury" Mumbai listing are each top-quartile *for their own city*, and the UI/insights copy must make this relative framing explicit (e.g., "top 25% for this city") rather than implying an absolute price-tier standard across India.
4. The `good_deal_flag` classifier (if built) must be trained on out-of-fold regression predictions, not in-sample predictions from the same regression model instance — otherwise the "good deal" signal is circular and meaningless.
5. Classification confusion matrices must be reviewed per city before shipping — a classifier that performs well in aggregate but poorly for one city (e.g., Kolkata, the smallest dataset) must not be silently shipped as "done."

### 9. Research-Integrity Rules (added following the literature review)
1. Every claim in the literature review document must be traceable to an actual reviewed paper — no fabricated papers, authors, or results. If a specific number (e.g., an R² or % improvement) cannot be found in a real source, it must be labeled as an estimate/target, not stated as a fact.
2. The "35% improvement" figure is a **literature-informed target to validate empirically**, not a claim already proven for this specific dataset — every version's actual measured improvement (or shortfall) must be logged honestly in the Tracker's KPI table, even if it comes in below the target.
3. When citing a paper's finding in any project document (PRD/TRD/etc.), reference it by its ID (B1–B4, S1–S18 from the literature doc) so the source is always traceable, rather than a vague "research shows..." statement.
4. Match-percentage scores assigned to papers (base vs. supporting) are qualitative judgments based on domain/method/system-scope overlap — they must be presented as such, not as a formal, standardized citation metric.

---

## UPDATE v3 — Input Schema & Documentation-Consistency Rules

### 10. Input Schema Rules
1. The 16-field input contract (`10_FINALIZED_INPUT_SCHEMA.md`) is the single source of truth for what the Predict/Classify forms collect — no field is added to or removed from the live form without updating that schema doc first, then propagating the change to TRD, Backend Schema, App Flow, and UI/UX.
2. `luxury_category` must never be collected as a raw self-reported dropdown from the end user — it must be derived server-side (or client-side pre-submit) from the finish/amenity checklist per UI/UX Update v3 §U-UX-7, to avoid self-report bias corrupting the training-serving distribution over time as more live predictions get logged.
3. `transact_type` (Sale/Rent) is a **routing key**, not a plain model feature — Sale and Rent listings must never be trained in the same regression target space. Mixing them is a hard rule violation, not a modeling nuance to tune later.
4. Any new field proposed for the input form must state, in one line, which of the 16 fields it's replacing/supplementing and why — no silent, undocumented field creep.

### 11. Documentation-Consistency Rule (UML)
The UML diagrams in `11_UML_DIAGRAMS.md` and the prose in PRD/TRD/App Flow/Backend Schema must describe one single, consistent system. If a future change is made to any doc (e.g., a new module, a changed API contract), the corresponding diagram(s) must be updated in the same change — a diagram that has quietly drifted out of sync with the prose is treated as a documentation bug, not a cosmetic issue, since it's the artifact most likely to be handed to a reviewer or teammate first.

---

## UPDATE v4 — Good-Deal-First Classification Rules

### 12. Classification Priority & Threshold Rules
1. `good_deal_verdict` is the module's **primary** output and must be evaluated first, with **recall on the "Good Deal" class** weighted above raw accuracy in model selection — a false "Good Deal" call has a real financial cost to the user, which a plain accuracy metric would not surface if that class is small.
2. Residual thresholds (±10%) that define Good Deal / Fair Price / Overpriced must be re-validated per city before shipping — do not assume Gurgaon's residual distribution (largest sample) transfers cleanly to Kolkata (smallest sample); tune independently and document the chosen threshold per city in the Decision Log.
3. `price_tier` (Affordability Tier) remains supporting-only: it powers the Recommender's "Fits my budget" filter and the Affordability chip, and must never be presented as if it were the module's main feature in any UI copy or documentation going forward.
4. Both classifiers still inherit the v2 leakage rule unchanged: neither `price_tier` nor `good_deal_verdict` may be fed back into the price regression model as an input feature.

### 13. Scope-Discipline Rule (added after external review)
This project's scope is fixed at: Price Prediction, Analytics, Recommender, Insights, and the Good-Deal/Affordability Classification module, on a Flask+FastAPI stack, without added MLOps tooling (MLflow/DVC/CI-CD/drift detection) — this was an explicit, considered decision, not an oversight. Any future suggestion (external review, new idea, etc.) to add further modules or infrastructure must be evaluated against **the original PRD goals**, not adopted just because it would look more sophisticated — scope creep was explicitly identified as a risk and this rule exists to guard against it going forward.

---

## UPDATE v5 (final) — Documentation Freeze

### 14. Documentation Freeze Rule
As of this update, the 11-document set (PRD, TRD, App Flow, UI/UX, Backend Schema, Implementation Plan, Tracker, Rules, Literature Review, Input Schema, UML) is **frozen**. Further changes to these documents are permitted only for:
1. Corrections discovered *during* implementation (a doc turns out to be wrong once real code is written against it), or
2. Genuine scope decisions made deliberately (not reactively to the next external review that arrives).

No new documents, no new modules, and no new infrastructure (MLflow, DVC, CI/CD, monitoring, geospatial clustering, etc.) are added from here without an explicit, separate decision — these are logged below as considered-and-deferred, not silently dropped.

### 15. Deferred Items Log (considered, not built — revisit only if there's spare time after core modules ship)
| Item | Why deferred |
|---|---|
| MLflow experiment tracking | Real infrastructure scope; the lightweight `model_registry` table (Backend Schema §U-SCHEMA-13) covers the traceability need at near-zero cost |
| DVC data versioning | Same reasoning — dataset filename/hash in `model_registry` is sufficient for this project's size |
| CI/CD (GitHub Actions → tests → Docker → deploy) | Genuinely valuable but a separate scope commitment; revisit after Week 8 if time remains |
| Prediction/latency/CPU monitoring | Same — revisit post-launch, not pre-launch |
| Geospatial clustering / hotspot analysis / nearest-metro estimation | `latitude`/`longitude` are already used for the spatial analytics tile and the distance-to-metro improvement lever; further geospatial work (clustering, heatmaps) is a genuine v2 differentiator, not a v1 requirement |

**Next action is implementation, not another document.**