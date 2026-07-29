# UML Diagrams — HousingIQ Platform

All diagrams use Mermaid syntax (renders directly in GitHub, VS Code, Claude, and most modern markdown viewers). They reflect the finalized 16-field input schema (`10_FINALIZED_INPUT_SCHEMA.md`), the 5 modules (Price Prediction, Analytics, Recommender, Insights, Classification), and the Flask + FastAPI architecture from the TRD.

---

## 1. Use Case Diagram

```mermaid
flowchart TB
    subgraph Actors
        Buyer((Home Buyer))
        Seller((Seller / Broker))
        Analyst((Analyst / Investor))
    end

    subgraph "HousingIQ System"
        UC1([Predict Property Price])
        UC2([Check Price Tier / Classification])
        UC3([View Market Analytics Dashboard])
        UC4([Get Similar Property Recommendations])
        UC5([View Auto-Generated Market Insights])
        UC6([Filter by City / Locality])
        UC7([View SHAP Explanation for a Prediction])
        UC8([Compare Listed Price vs Predicted -Good Deal-])
    end

    Buyer --> UC1
    Buyer --> UC2
    Buyer --> UC4
    Buyer --> UC5
    Buyer --> UC7
    Buyer --> UC8

    Seller --> UC1
    Seller --> UC2
    Seller --> UC4
    Seller --> UC8

    Analyst --> UC3
    Analyst --> UC5
    Analyst --> UC6

    UC1 -.includes.-> UC7
    UC2 -.includes.-> UC7
    UC1 -.extends.-> UC8
    UC3 -.includes.-> UC6
    UC4 -.includes.-> UC6
```

---

## 2. Class Diagram (domain + ML serving layer)

```mermaid
classDiagram
    class Listing {
        +string listing_id
        +string city
        +string sector
        +string property_type
        +string transact_type
        +int bedRoom
        +int bathroom
        +string balcony
        +string agePossession
        +float built_up_area
        +bool servant_room
        +bool store_room
        +int furnishing_type
        +string luxury_category
        +string floor_category
        +string facing
        +list~string~ amenities_list
        +int n_amenities
        +float latitude
        +float longitude
        +float price
        +float price_per_sqft
        +string price_tier
        +bool is_outlier
    }

    class PredictionRequest {
        +string city
        +string sector
        +string property_type
        +string transact_type
        +int bedRoom
        +int bathroom
        +string balcony
        +string agePossession
        +float built_up_area
        +bool servant_room
        +bool store_room
        +int furnishing_type
        +string luxury_category
        +string floor_category
        +string facing
        +list~string~ amenities
        +validate() bool
    }

    class PredictionResponse {
        +float predicted_price
        +float range_low
        +float range_high
        +bool is_outlier_input
        +string model_version
        +list~ShapContribution~ shap_contributions
    }

    class ClassificationResponse {
        +string price_tier
        +dict tier_probabilities
        +bool good_deal_flag
        +string model_version
        +list~ShapContribution~ shap_contributions
    }

    class ShapContribution {
        +string feature
        +float impact
    }

    class RecommendationRequest {
        +string city
        +string sector
        +int top_n
        +bool expand_search
        +list~string~ tier_filter
    }

    class RecommendedProperty {
        +string listing_id
        +float price
        +float area_sqft
        +int bedrooms
        +string locality
        +float similarity
        +list~string~ matched_on
    }

    class PricePredictionService {
        -Pipeline model
        -TreeExplainer explainer
        +predict(PredictionRequest) PredictionResponse
        +explain(PredictionRequest) list~ShapContribution~
    }

    class ClassificationService {
        -Pipeline model
        -dict tier_quantile_boundaries
        +classify(PredictionRequest) ClassificationResponse
    }

    class RecommenderService {
        -TfidfVectorizer vectorizer
        -NearestNeighbors index
        +recommend(RecommendationRequest) list~RecommendedProperty~
        +fallback_by_popularity(city, sector) list~RecommendedProperty~
    }

    class InsightsService {
        -DataFrame locality_stats
        -DataFrame amenity_uplift
        +generate(city, sector, predicted_price) list~string~
    }

    class FlaskApp {
        +render_predict_form()
        +render_predict_result()
        +render_analytics_dashboard()
        +render_recommend_form()
        +render_insights_page()
        +render_classify_page()
    }

    class FastAPIGateway {
        +POST /predict
        +POST /classify
        +POST /recommend
        +GET /insights
        +GET /health
    }

    FlaskApp --> FastAPIGateway : HTTP calls
    FastAPIGateway --> PricePredictionService
    FastAPIGateway --> ClassificationService
    FastAPIGateway --> RecommenderService
    FastAPIGateway --> InsightsService
    PricePredictionService --> ShapContribution
    ClassificationService --> ShapContribution
    PredictionRequest --> PredictionResponse : produces
    RecommendationRequest --> RecommendedProperty : produces
    Listing --> PricePredictionService : training data
    Listing --> ClassificationService : training data
    Listing --> RecommenderService : training data
```

---

## 3. Sequence Diagram — Price Prediction (+ parallel Classification call)

```mermaid
sequenceDiagram
    actor User
    participant Flask as Flask App
    participant API as FastAPI Gateway
    participant PS as PricePredictionService
    participant CS as ClassificationService
    participant IS as InsightsService

    User->>Flask: Submit 16-field prediction form
    Flask->>Flask: Validate required fields (city, sector, area>0, ...)
    par Predict price
        Flask->>API: POST /predict {features}
        API->>PS: predict(request)
        PS->>PS: preprocess -> pipeline.predict() -> inverse log1p
        PS->>PS: compute SHAP contributions
        PS-->>API: PredictionResponse
        API-->>Flask: 200 OK {price, range, shap}
    and Classify tier
        Flask->>API: POST /classify {features}
        API->>CS: classify(request)
        CS->>CS: preprocess -> pipeline.predict_proba()
        CS->>CS: compute SHAP contributions
        CS-->>API: ClassificationResponse
        API-->>Flask: 200 OK {tier, probabilities, shap}
    end
    Flask->>API: GET /insights?city&sector&predicted_price
    API->>IS: generate(city, sector, predicted_price)
    IS->>IS: lookup locality_stats + amenity_uplift tables
    IS-->>API: list of insight sentences
    API-->>Flask: 200 OK {insights}
    Flask-->>User: Render result page (price hero, tier badge, SHAP charts, insights)

    Note over Flask,API: If /classify fails, Flask omits the TierBadge only —<br/>price prediction result still renders (graceful degradation).
```

---

## 4. Sequence Diagram — Recommender System (with cold-start fallback)

```mermaid
sequenceDiagram
    actor User
    participant Flask as Flask App
    participant API as FastAPI Gateway
    participant RS as RecommenderService

    User->>Flask: Submit seed property / preference form
    Flask->>API: POST /recommend {features, top_n, tier_filter}
    API->>RS: recommend(request)
    RS->>RS: scope candidates to city (+ sector unless expand_search)
    RS->>RS: build combined feature vector (numeric+categorical+TF-IDF)
    RS->>RS: cosine_similarity(seed_vector, candidate_matrix)
    alt at least 5 candidates above similarity threshold
        RS-->>API: top-N similar properties (used_fallback=false)
    else fewer than 5 strong matches
        RS->>RS: fallback_by_popularity(locality listing_count + recency)
        RS-->>API: top-N popular properties (used_fallback=true)
    end
    API-->>Flask: 200 OK {results, used_fallback}
    Flask-->>User: Render property cards (labeled "Similar" or "Popular in this area")
```

---

## 5. Activity Diagram — Offline ML Pipeline (Data → Deployed Model)

```mermaid
flowchart TD
    A[Start: Load 4 raw city CSVs + 15 facet lookup tables] --> B[Parse PRICE / AREA / MAP_DETAILS strings]
    B --> C[Decode facet-coded columns via joins]
    C --> D[Map each city's raw columns into canonical schema]
    D --> E[Drop unusable columns + deduplicate by PROP_ID]
    E --> F{Missingness per column}
    F -->|<5%| G[Median/mode impute]
    F -->|5-40%| H[Group-wise impute + was_missing flag]
    F -->|40-70%| I[Model-based impute or Unknown category]
    F -->|>70%| J[Drop column]
    G --> K[EDA: univariate, bivariate, multivariate, pandas-profiling]
    H --> K
    I --> K
    K --> L[Outlier detection: percentile cap + IQR + domain rules]
    L --> M[Feature engineering: price_per_sqft, n_amenities, floor_ratio, facing, luxury_category, floor_category, etc.]
    M --> N[Feature selection: correlation, RF/GB importance, permutation importance, Lasso, RFE, linear weights, SHAP]
    N --> O{Model selection}
    O --> P[Train Linear/Ridge/Lasso, RF, GB, XGBoost]
    P --> Q[5-fold CV + compare R2/MAE/RMSE/MAPE]
    Q --> R[Apply improvement levers: stacking, Optuna tuning, geospatial features, target encoding, SHAP-guided refinement]
    R --> S[Select final regression pipeline]
    S --> T[Train price_tier classifier reusing selected features minus price]
    T --> U[Serialize model.pkl + metrics.json + SHAP explainer]
    U --> V[Precompute analytics_cache JSON + locality_stats + amenity_uplift tables]
    V --> W[Deploy via FastAPI /predict /classify /recommend /insights]
    W --> X[End: Live in Flask app]
```

---

## 6. Component Diagram

```mermaid
flowchart LR
    subgraph Client["Client (Browser)"]
        UI[HTML/CSS/JS Templates + Chart.js/Leaflet]
    end

    subgraph FlaskLayer["Flask Application (Web/UI Layer)"]
        Routes[Route Handlers: /predict /analytics /recommend /insights /classify]
        Templates[Jinja2 Templates]
        StaticAssets[Static CSS/JS]
    end

    subgraph FastAPILayer["FastAPI Service (Inference Layer)"]
        PredictRoute["/predict"]
        ClassifyRoute["/classify"]
        RecommendRoute["/recommend"]
        InsightsRoute["/insights"]
        HealthRoute["/health"]
    end

    subgraph MLArtifacts["Model Artifacts Store"]
        PriceModel[(price_model_v{n}.pkl)]
        TierModel[(tier_classifier_v{n}.pkl)]
        TfidfIdx[(tfidf_vectorizer + NN index)]
        ShapExp[(SHAP explainers)]
    end

    subgraph DataStore["Data Store"]
        CleanData[(clean_listings.parquet)]
        AnalyticsCache[(analytics_cache/*.json)]
        AggTables[(locality_stats, amenity_uplift, etc.)]
        AppDB[(SQLite/Postgres: prediction_log, classification_log, recommendation_log)]
    end

    UI --> Routes
    Routes --> Templates
    Routes -->|HTTP JSON| PredictRoute
    Routes -->|HTTP JSON| ClassifyRoute
    Routes -->|HTTP JSON| RecommendRoute
    Routes -->|HTTP JSON| InsightsRoute
    Routes -->|liveness check| HealthRoute

    PredictRoute --> PriceModel
    PredictRoute --> ShapExp
    ClassifyRoute --> TierModel
    ClassifyRoute --> ShapExp
    RecommendRoute --> TfidfIdx
    InsightsRoute --> AggTables

    PriceModel -.trained from.-> CleanData
    TierModel -.trained from.-> CleanData
    TfidfIdx -.trained from.-> CleanData
    Routes --> AnalyticsCache
    PredictRoute --> AppDB
    ClassifyRoute --> AppDB
    RecommendRoute --> AppDB
```

---

## 7. Deployment Diagram

```mermaid
flowchart TB
    subgraph UserDevice["User Device"]
        Browser[Web Browser - Desktop/Mobile]
    end

    subgraph WebServer["Web Server Node"]
        FlaskProc["Flask App (Gunicorn/WSGI)"]
    end

    subgraph InferenceServer["Inference Server Node"]
        FastAPIProc["FastAPI App (Uvicorn/ASGI)"]
        ModelFiles[/models directory - .pkl artifacts/]
    end

    subgraph DataServer["Data / Storage Node"]
        DB[(PostgreSQL / SQLite)]
        ParquetFiles[/data/processed - parquet + JSON cache/]
    end

    Browser -- HTTPS --> FlaskProc
    FlaskProc -- internal HTTP (localhost or service mesh) --> FastAPIProc
    FastAPIProc --> ModelFiles
    FlaskProc --> DB
    FastAPIProc --> DB
    FlaskProc --> ParquetFiles
    FastAPIProc --> ParquetFiles

    note1[["Flask never loads model files directly -\nall inference goes through FastAPI over HTTP\n(per Rules doc engineering rule)"]]
    FlaskProc -.-> note1
```

---

## 8. Entity-Relationship (Data Model) Diagram

```mermaid
erDiagram
    LISTING ||--o{ PREDICTION_LOG : "generates predictions for"
    LISTING ||--o{ CLASSIFICATION_LOG : "generates classifications for"
    LISTING }o--|| CITY_REF : "belongs to"
    LISTING }o--|| LOCALITY_REF : "located in"
    LISTING }o--|| PROPERTY_TYPE_REF : "has type"
    LISTING }o--|| FURNISH_REF : "has furnishing"
    LISTING }o--|| FACING_REF : "faces"
    LISTING }o--|| AGE_REF : "has age bucket"
    LISTING }o--o{ AMENITIES_REF : "has many"
    LISTING ||--o| RECOMMENDATION_LOG : "seeds"
    LOCALITY_REF ||--o{ LOCALITY_STATS : "aggregated into"
    LOCALITY_REF ||--o{ AMENITY_UPLIFT : "aggregated into"

    LISTING {
        string listing_id PK
        string city_id FK
        string locality_id FK
        string property_type_id FK
        int bedRoom
        int bathroom
        string balcony
        string agePossession
        float built_up_area
        bool servant_room
        bool store_room
        int furnishing_type_id FK
        string luxury_category
        string floor_category
        string facing_id FK
        float price
        float price_per_sqft
        string price_tier
        float latitude
        float longitude
        bool is_outlier
    }

    CITY_REF {
        string city_id PK
        string city_label
    }

    LOCALITY_REF {
        string locality_id PK
        string city_id FK
        string locality_label
    }

    PROPERTY_TYPE_REF {
        string type_id PK
        string type_label
    }

    FURNISH_REF {
        string furnish_id PK
        string furnish_label
    }

    FACING_REF {
        string facing_id PK
        string facing_label
    }

    AGE_REF {
        string age_id PK
        string age_label
    }

    AMENITIES_REF {
        string amenity_id PK
        string amenity_label
        string category
    }

    LOCALITY_STATS {
        string city
        string locality
        float avg_price
        float avg_price_per_sqft
        int listing_count
    }

    AMENITY_UPLIFT {
        string city
        string locality
        string amenity
        float pct_uplift
    }

    PREDICTION_LOG {
        int id PK
        datetime timestamp
        string listing_ref FK
        float predicted_price
        string model_version
    }

    CLASSIFICATION_LOG {
        int id PK
        datetime timestamp
        string listing_ref FK
        string predicted_tier
        string model_version
    }

    RECOMMENDATION_LOG {
        int id PK
        datetime timestamp
        string seed_listing_id FK
        bool used_fallback
    }
```

---

## 9. State Diagram — A Single Prediction Request's Lifecycle

```mermaid
stateDiagram-v2
    [*] --> FormFilled
    FormFilled --> Validating : Submit
    Validating --> Invalid : missing/out-of-range field
    Invalid --> FormFilled : show inline error
    Validating --> CallingServices : all fields valid
    CallingServices --> PredictOK : /predict succeeds
    CallingServices --> PredictFailed : /predict times out or errors
    PredictOK --> ClassifyOK : /classify succeeds
    PredictOK --> ClassifyFailed : /classify times out or errors
    ClassifyOK --> InsightsOK : /insights succeeds
    ClassifyFailed --> InsightsOK : tier badge omitted, continue
    InsightsOK --> ResultRendered
    PredictFailed --> ServiceUnavailableState : show friendly degraded message
    ServiceUnavailableState --> [*]
    ResultRendered --> [*]
```

---

*All diagrams are kept consistent with the 16-field input schema (`10_FINALIZED_INPUT_SCHEMA.md`), the 5-module scope (Price Prediction, Analytics, Recommender, Insights, Classification), and the Flask+FastAPI architecture defined across the TRD, App Flow, and Backend Schema updates below.*