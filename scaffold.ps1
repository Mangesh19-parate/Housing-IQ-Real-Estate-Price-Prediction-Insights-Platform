# Create root folder
mkdir housingiq
cd housingiq

# Root files
New-Item CLAUDE.md -ItemType File
New-Item TODO.md -ItemType File
New-Item requirements.txt -ItemType File

# .claude
mkdir .claude
mkdir .claude\agents
mkdir .claude\agent-memory
mkdir .claude\commands
mkdir .claude\skills
mkdir .claude\specs
mkdir .claude\plans

New-Item .claude\setting.json -ItemType File
New-Item .claude\settings.local.json -ItemType File

# Agents
New-Item .claude\agents\housingiq-test-writer.md -ItemType File
New-Item .claude\agents\housingiq-quality-reviewer.md -ItemType File
New-Item .claude\agents\housingiq-security-reviewer.md -ItemType File
New-Item .claude\agents\housingiq-ml-evaluator.md -ItemType File
New-Item .claude\agents\housingiq-test-runner.md -ItemType File

# Agent Memory
$memories = @(
"housingiq-test-writer",
"housingiq-quality-reviewer",
"housingiq-security-reviewer",
"housingiq-ml-evaluator",
"housingiq-test-runner"
)

foreach ($m in $memories){
    mkdir ".claude\agent-memory\$m"
    New-Item ".claude\agent-memory\$m\MEMORY.md" -ItemType File
}

New-Item .claude\agent-memory\housingiq-test-writer\no-prior-conftest.md -ItemType File
New-Item .claude\agent-memory\housingiq-test-writer\housingiq-test-patterns.md -ItemType File

New-Item .claude\agent-memory\housingiq-quality-reviewer\code-style-notes.md -ItemType File

New-Item .claude\agent-memory\housingiq-security-reviewer\privacy-rules-checklist.md -ItemType File

New-Item .claude\agent-memory\housingiq-ml-evaluator\metric-protocol-notes.md -ItemType File

New-Item .claude\agent-memory\housingiq-test-runner\flaky-test-log.md -ItemType File

# Commands
$commands=@(
"create-spec",
"create-plan",
"test-feature",
"code-review-feature",
"seed-listings",
"seed-user",
"train-price-model",
"evaluate-model",
"generate-shap-report",
"build-analytics-cache",
"generate-recommender-index",
"migrate-schema",
"update-tracker",
"lint-check",
"deploy-check"
)

foreach($c in $commands){
    New-Item ".claude\commands\$c.md" -ItemType File
}

# Skills
$skills=@(
"data-cleaning-and-parsing",
"facet-decoding",
"feature-engineering",
"model-training-regression",
"model-training-classification",
"model-evaluation-protocol",
"shap-explainability",
"fastapi-serving",
"flask-routing",
"sqlite-postgres-schema",
"parquet-caching",
"chartjs-plotly-charting",
"leaflet-mapping",
"recommender-similarity-search",
"tfidf-text-features",
"insights-narrative-templater",
"frontend-design",
"testing-pytest-flask-fastapi",
"security-review-data-privacy",
"api-schema-design-pydantic",
"css-design-tokens-and-card-system",
"accessibility-review",
"deployment-docker-uvicorn-wsgi",
"seo-landing-page",
"git-workflow-and-spec-driven-development"
)

foreach($s in $skills){
    mkdir ".claude\skills\$s"
    New-Item ".claude\skills\$s\SKILL.md" -ItemType File
}

# Data
mkdir data
mkdir data\raw
mkdir data\processed
mkdir data\processed\analytics_cache
mkdir data\stats

New-Item data\processed\clean_listings.parquet -ItemType File

# Notebooks
mkdir notebooks

# ML
mkdir ml
mkdir ml\cleaning
mkdir ml\features
mkdir ml\training
mkdir ml\evaluation
mkdir ml\recommender

# Models
mkdir models

# API
mkdir api
mkdir api\routers
mkdir api\schemas
mkdir api\services

New-Item api\main.py -ItemType File

$routers=@(
"predict",
"classify",
"analytics",
"recommend",
"insights"
)

foreach($r in $routers){
    New-Item "api\routers\$r.py" -ItemType File
}

# Flask App
mkdir app

New-Item app\app.py -ItemType File

mkdir app\database
New-Item app\database\db.py -ItemType File

mkdir app\templates

$templates=@(
"base",
"landing",
"login",
"register",
"predict",
"analytics",
"recommend",
"insights",
"map_explorer"
)

foreach($t in $templates){
    New-Item "app\templates\$t.html" -ItemType File
}

mkdir app\static
mkdir app\static\css
mkdir app\static\js

$css=@(
"style",
"analytics",
"cards"
)

foreach($c in $css){
    New-Item "app\static\css\$c.css" -ItemType File
}

$js=@(
"main",
"charts",
"map"
)

foreach($j in $js){
    New-Item "app\static\js\$j.js" -ItemType File
}

# Tests
mkdir tests

$tests=@(
"test_price_prediction",
"test_classification",
"test_analytics",
"test_recommender",
"test_insights",
"conftest"
)

foreach($t in $tests){
    New-Item "tests\$t.py" -ItemType File
}

Write-Host ""
Write-Host "================================="
Write-Host "HousingIQ Project Created Successfully!"
Write-Host "================================="