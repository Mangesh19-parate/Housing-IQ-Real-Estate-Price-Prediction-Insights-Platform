# UI/UX Design Document
## Project: HousingIQ — India Real Estate Price Prediction & Insights Platform

---

## 1. Design Principles
1. **Numbers first, chrome second** — the predicted price, key stats, and charts are the hero content; UI decoration stays minimal.
2. **Explain, don't just output** — every prediction/recommendation is paired with a "why" (SHAP chart, matched attributes, insight sentence). Never show a bare number with no context.
3. **Progressive disclosure** — forms are short by default (required fields only), with an "Advanced" collapsible section for amenities/floor/facing, so first-time users aren't overwhelmed.
4. **Consistent card language** — properties, insights, and chart tiles all use the same card component (rounded corners, consistent padding, subtle shadow) so the app feels like one system, not four bolted-together tools.
5. **Graceful degradation** — every async section (predictions, recommendations, charts) has a loading state and a failure state; nothing renders a raw error.

## 2. Visual Style
- **Palette**: Deep navy/slate (`#1E293B`) for headers/nav, warm accent (`#F59E0B` amber or `#2563EB` blue — pick one as primary CTA color) for buttons/highlights, off-white background (`#F8FAFC`), neutral grays for secondary text. Price-up/price-down SHAP bars use green (`#16A34A`) / red (`#DC2626`) consistently across the app.
- **Typography**: A clean sans-serif (e.g., Inter, or system font stack) — large numerals for price displays (e.g., 2.5–3rem), regular body text 0.95–1rem.
- **Iconography**: Simple line icons for bedrooms/bathrooms/area/floor (consistent icon set across cards, forms, and result pages).
- **Spacing**: 8px baseline grid; generous white space around the price hero number.

## 3. Layout — Landing Page
```
┌─────────────────────────────────────────────────────────┐
│  Logo   HousingIQ            [Predict|Analytics|Recommend|Insights] │  ← sticky nav
├─────────────────────────────────────────────────────────┤
│   Headline: "Know what a home is really worth."           │
│   Subtext + City quick-select chips: Gurgaon Hyderabad     │
│   Kolkata Mumbai  [All]                                    │
├─────────────────────────────────────────────────────────┤
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ │
│  │ Predict    │ │ Analytics  │ │ Recommend  │ │ Insights   │ │  ← 4 module cards
│  │ price      │ │ dashboard  │ │ properties │ │ market view│ │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────┤
│  Footer: dataset note, module links, "built with"          │
└─────────────────────────────────────────────────────────┘
```

## 4. Price Prediction Module — Screens

### 4.1 Form screen (`/predict`)
- Two-column form on desktop (single column mobile): left = core fields (City, Locality, Property Type, Area, Bedrooms, Bathrooms), right = secondary fields (Furnishing, Facing, Age, Floor/Total Floor).
- Amenities shown as a chip/checkbox grid below (collapsed under "Add amenities" by default).
- Sticky "Predict Price" primary button, disabled until required fields are valid; inline validation messages (not just red borders — actual text like "Area must be greater than 0").
- Locality dropdown is dependent on City (AJAX-populated) with a search-as-you-type filter since some cities have hundreds of localities.

### 4.2 Result screen (`/predict/result`)
```
┌─────────────────────────────────────────────┐
│  ₹ 1.42 Cr  (₹1.28 Cr – ₹1.56 Cr range)        │  ← hero price + range
│  3 BHK · 1450 sqft · Sector 84, Gurgaon        │
├─────────────────────────────────────────────┤
│  Why this price?                               │
│  [ horizontal SHAP bar chart: Area +18%,        │
│    Locality +9%, Age -4%, Furnish +3% ... ]     │
├─────────────────────────────────────────────┤
│  Market Insights                               │
│  • This price is 8% below the Sector 84 avg     │
│  • 3BHKs here average ₹1.5Cr                    │
│  • Clubhouse amenity adds ~6% in this locality   │
├─────────────────────────────────────────────┤
│  [ See similar properties → ]  (CTA to Recommender) │
└─────────────────────────────────────────────┘
```
- Confidence flag (if input is an outlier vs. training distribution) shown as a small amber banner above the hero price, not hidden in fine print.

## 5. Analytics Module — Dashboard Layout
- Grid of tiles (2-column desktop, 1-column mobile), each tile = one chart from the 13-chart list (App Flow §2.3), with a chart title, one-line takeaway caption (e.g., "Prices rise sharply above the 15th floor in Gurgaon high-rises"), and the chart itself.
- Global City filter pinned at top of dashboard; changing it updates all tiles via AJAX (no full reload), with a skeleton-loading shimmer per tile while new data loads.
- Spatial analysis tile is the only full-width tile (map needs more horizontal room) — placed first.
- Word cloud tile includes a toggle: "By city" vs "All cities."

## 6. Recommender Module — Screens
- Form screen mirrors the Prediction form (reuse form component) but with a "Budget range" slider instead of a single expected price, since recommendation is about matching a range of acceptable options.
- Results screen: card grid, each card shows: photo-less placeholder icon (or generic building illustration, since we don't want to depend on scraped photo URLs long-term), price, BHK, area, locality, and a small "Match: 92%" badge plus 2–3 highlighted matching attributes as tags (e.g., "Same locality," "Similar price/sqft," "Furnished").
- Cold-start fallback results are visually distinguished with a "Popular in this area" tag instead of a similarity percentage, so users aren't misled (per App Flow §2.4 step 4).

## 7. Insights Module — Screen
- Simple, readable, text-forward page: City selector at top, then a list of 6–10 insight sentences grouped under short headers ("Pricing," "Size & Layout," "Amenities," "Location"), each paired with a small inline sparkline/mini-chart where relevant.
- This page is intentionally the least "dashboard-y" — it should read more like a short market report than a BI tool.

## 8. Component Inventory (reusable across modules)
- `PriceHero` (big number + range)
- `ShapBarChart`
- `PropertyCard`
- `InsightCard` / `InsightList`
- `ChartTile` (title + caption + chart body + loading/error states)
- `DependentDropdown` (City → Locality)
- `AmenityChipGrid`
- `EmptyState` (used for "no recommendations found," "service unavailable," etc.)

## 9. Responsive Behavior
- Breakpoints: mobile (<640px, single column, sticky bottom CTA button), tablet (640–1024px, 2-column), desktop (>1024px, full grid layouts as above).
- Charts library (Chart.js/Plotly) configured with `responsive: true` and container-based resizing; map tile uses Leaflet's native responsive container.

## 10. Accessibility
- All form fields have associated `<label>`s; color is never the only signal (SHAP up/down bars also carry +/− text labels; confidence banner has an icon + text, not just color).
- Sufficient contrast ratio for text on the navy header and amber/blue CTA buttons (verify against WCAG AA).
- Charts include a text-summary fallback (e.g., a `<caption>` or adjacent one-line takeaway) for screen readers, since canvas-based charts aren't inherently accessible.

---

## UPDATE v2 — Classification Module UI

### U-UX-1. New Component: `TierBadge`
A pill/chip component with 4 fixed color mappings so tiers are visually consistent everywhere they appear (Prediction result, Analytics tiles, Recommender cards, standalone Classify page):
- `Budget` — green (`#16A34A`)
- `Mid-Range` — blue (`#2563EB`)
- `Premium` — purple (`#7C3AED`)
- `Luxury` — gold/amber (`#D97706`)

Each badge always carries the text label, never color alone (per existing accessibility rule in UI/UX §10).

### U-UX-2. Price Prediction Result Screen (updated layout)
```
┌─────────────────────────────────────────────┐
│  ₹ 1.42 Cr (range)        🏷 Premium           │  ← TierBadge added top-right of hero
│  3 BHK · 1450 sqft · Sector 84, Gurgaon        │
├─────────────────────────────────────────────┤
│  Why this price?  [SHAP bars]                  │
│  Why this tier? (collapsed) ▸ [SHAP bars]       │  ← new collapsible section
├─────────────────────────────────────────────┤
│  Market Insights (incl. new tier-mix sentence)  │
├─────────────────────────────────────────────┤
│  [ See similar properties → ]                    │
└─────────────────────────────────────────────┘
```
If the classification service is unavailable, the `TierBadge` slot simply disappears (empty-slot degradation, not an error banner) — the price hero must never be blocked or visually disrupted by a missing tier.

### U-UX-3. New Standalone Screen: `/classify`
Reuses the `PropertyForm` component (shared with `/predict`) but swaps the primary button to "Check Price Tier." Result view shows the winning tier as a large `TierBadge`, then a horizontal 4-bar probability chart (all tiers, not just the winner — this is important for trust: a 52%-confidence Premium call should visibly show it was close to Mid-Range too), then a SHAP breakdown reusing the `ShapBarChart` component.

### U-UX-4. Recommender Filter & Card Update
Add a "Tier" multi-select chip group to the Recommender form (reuse `AmenityChipGrid` component pattern). Each `PropertyCard` in results gains a small `TierBadge` (compact size) next to the existing similarity-score/"Popular in this area" tag — two badges max per card to avoid clutter.

### U-UX-5. Analytics Tile 14
"Price Tier Mix by City" — stacked horizontal bar chart, one bar per city, segments colored per the U-UX-1 palette, with an on-hover tooltip showing exact listing counts and %.

---

## UPDATE v3 — Finalized Form UI (16 Fields) & Guided Luxury-Category Input

### U-UX-6. Predict/Classify form — finalized layout
Reflecting the field order locked in App Flow Update v3 (§U-FLOW-6):
```
┌──────────────── Step 1: Location & Deal Type ────────────────┐
│ City ▾   Sector/Locality ▾ (searchable, city-scoped)           │
│ Property Type ▾   ○ Sale  ○ Rent                                │
├──────────────── Step 2: Core Structure ───────────────────────┤
│ Bedrooms [-][3][+]  Bathrooms [-][2][+]  Balconies ▾            │
│ Built-up Area: [____] sqft                                       │
├──────────────── Step 3: Details ──────────────────────────────┤
│ Property Age ▾   Furnishing ▾   Facing ▾   Floor Category ▾      │
├──────────────── Step 4: Extras ───────────────────────────────┤
│ Servant Room  ○ Yes ○ No     Store Room  ○ Yes ○ No              │
│ "What's the finish level?" (mini-checklist → auto-derives          │
│  Luxury Category, not a raw dropdown the user has to self-judge)  │
│   ☐ Branded developer  ☐ Imported fittings  ☐ Clubhouse access    │
├──────────────── Step 5: Amenities (optional, collapsed) ──────┤
│ ☐ Swimming Pool ☐ Gym ☐ Security ☐ Power Backup ☐ Lift ...       │
└────────────────────────────────────────────────────────────────┘
[ Predict Price ]   or   [ Check Price Tier ]
```

### U-UX-7. Why Luxury Category is guided, not a raw dropdown
Asking a user to self-classify their own property as "Low/Medium/High luxury" invites inconsistent, ego-biased input (nearly everyone rates their own property upward). Instead, per the reference project's pattern of deriving `luxury_category` from a computed `luxury_score`, the UI derives it from the same short finish/amenity checklist already being collected in Step 4 — the category itself is computed client-side (or server-side on submit) from those checkbox answers, never typed in directly by the user.

### U-UX-8. Sale vs. Rent toggle
The Sale/Rent radio in Step 1 changes the currency formatting and range language on the result screen ("₹1.42 Cr" for Sale vs. "₹42,000/month" for Rent) and determines which of the two backend pipelines is called (per TRD Update v3 §U-TRD-4) — this must be visually prominent (radio buttons, not a buried dropdown) since it's the single most consequential field in the form.

---

## UPDATE v4 — Good-Deal-First Visual Hierarchy

### U-UX-9. `TierBadge` component split into two, with Good Deal now primary
- **`VerdictBadge`** (new, primary) — Good Deal (green ✅) / Fair Price (blue ⚖️) / Overpriced (amber/red ⚠️) — larger, sits immediately next to the price hero.
- **`AffordabilityChip`** (was `TierBadge`) — Budget/Mid-Range/Premium/Luxury — smaller, secondary, sits below or beside the `VerdictBadge`.
This visual hierarchy directly reflects U4.2's PRD change: the "is this priced fairly" verdict is the headline; the affordability segment is supporting context.

### U-UX-10. Standalone page renamed
`/classify` page heading changes from "Check Price Tier" to **"Is this a good deal?"**, with the `VerdictBadge` as the large, above-the-fold result and the `AffordabilityChip` + SHAP breakdown below it — same layout skeleton as before, just re-prioritized content.

### U-UX-11. Recommender tier filter — copy update
Filter label changes from generic "Tier" to **"Fits my budget"** with the four affordability tiers as checkable options — small copy change, but makes the filter's purpose self-evident without needing a tooltip.