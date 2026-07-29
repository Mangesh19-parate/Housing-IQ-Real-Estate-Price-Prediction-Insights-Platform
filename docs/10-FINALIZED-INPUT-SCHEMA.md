# Finalized Price Prediction Input Schema (v3)

This is the authoritative input contract for the Price Prediction module (and, by extension, the Classification module, which reuses the same fields minus price-derived ones). It replaces the illustrative field list used in earlier doc versions. Cross-checked directly against the reference project's final model-ready file (`gurgaon_properties_post_feature_selection_v2.csv`, 13 columns: `property_type, sector, price, bedRoom, bathroom, balcony, agePossession, built_up_area, servant room, store room, furnishing_type, luxury_category, floor_category`).

## 1. Your 12 specified fields (kept exactly, mapped to canonical names)

| # | Your field | Canonical field name | Type | Values / Range |
|---|---|---|---|---|
| 1 | Property type | `property_type` | categorical | `flat`, `house` (extendable: villa, plot) |
| 2 | Sector | `sector` (paired with `city` — see addition #13) | categorical | e.g. `sector 36`, `sohna road`, `sector 89` (city-specific locality list) |
| 3 | No. of bedrooms | `bedRoom` | numeric (int) | 1–10 typical |
| 4 | No. of bathrooms | `bathroom` | numeric (int) | 1–10 typical |
| 5 | Balconies | `balcony` | categorical/ordinal | `0`, `1`, `2`, `3`, `3+` |
| 6 | Property age | `agePossession` | categorical | `New Property`, `Relatively New`, `Moderately Old`, `Old Property`, `Under Construction` |
| 7 | Built-up area | `built_up_area` | numeric (float, sqft) | e.g. 500–10,000 |
| 8 | Servant room | `servant room` | binary (0/1) | present / not present |
| 9 | Store room | `store room` | binary (0/1) | present / not present |
| 10 | Furnishing type | `furnishing_type` | categorical (encoded 0/1/2) | `Unfurnished`(0) / `Semifurnished`(1) / `Furnished`(2) |
| 11 | Luxury category | `luxury_category` | categorical | `Low`, `Medium`, `High` (derived from a `luxury_score`, per reference project) |
| 12 | Floor category | `floor_category` | categorical | `Low Floor`, `Mid Floor`, `High Floor` (derived from `floorNum`) |

## 2. Fields added (my recommendation), with rationale

| # | Added field | Canonical field name | Type | Why it's needed here specifically |
|---|---|---|---|---|
| 13 | **City** | `city` | categorical | The reference project is Gurgaon-only, so `sector` alone was a sufficient locality key. This project spans **4 cities** (Gurgaon, Hyderabad, Kolkata, Mumbai) — `sector`/`locality` values are not unique across cities, so `city` must be captured first and used to scope the `sector`/`locality` dropdown (per App Flow's dependent-dropdown pattern). Without it, "Sector 89" is ambiguous. |
| 14 | **Facing direction** | `facing` | categorical | Present in the raw dataset (`facets/FACING_DIRECTION.csv`) and a well-documented price driver in Indian residential real estate (Vaastu-linked premium/discount, e.g. North/East-facing commanding a premium in several markets). It was available but unused in the reference project — worth testing as a feature rather than assuming it's negligible. |
| 15 | **Amenities (multi-select)** | `amenities_list` → engineered to `n_amenities` + top-k `has_<amenity>` flags | categorical (multi) → engineered numeric/binary | The reference project only uses a single scalar `luxury_score`/`luxury_category` as a proxy for amenity richness. This project's dataset has an explicit, decodable `AMENITIES` field (clubhouse, pool, gym, security, etc. — see Backend Schema §2/§3), so exposing it directly gives both a stronger feature set and a more transparent SHAP explanation ("+9% because of Clubhouse + Swimming Pool") than a single opaque luxury score. |
| 16 | **Transaction type** | `transact_type` | categorical | `Sale` vs `Rent`. The reference project's `price` column is implicitly sale price only. Our 4-city dataset mixes both (`TRANSACT_TYPE` field present), and sale price and rent price live on entirely different scales — feeding both into one regression without this flag would silently corrupt the model. This field is used to **route the request to the correct model** (a sale-price model or a rent-price model), not as a plain feature. |

## 3. Final combined input form (16 fields, grouped for the UI)

```
Required (drives locality dropdown + core structure):
  City → Sector/Locality → Property Type → Transaction Type (Sale/Rent)
  Bedrooms, Bathrooms, Balconies, Built-up Area (sqft)

Secondary (still required, but shown after the above):
  Property Age/Possession, Furnishing Type, Facing Direction, Floor Category

Derived from a short guided sub-form (not raw numbers the user must know):
  Luxury Category  → inferred from an "Amenities & Finish" mini-checklist (clubhouse, imported fittings, branded developer, etc.) the same way the reference project derives it from a luxury_score, rather than asking the user to self-report "Low/Medium/High"
  Servant Room / Store Room → simple Yes/No toggles

Optional (collapsed under "Add amenities"):
  Amenities multi-select (Clubhouse, Swimming Pool, Gym, Security, Power Backup, Lift, ...)
```

This finalized schema is what all 8 updated documents below now reference as **"the 16-field input contract."**