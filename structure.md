# Planned Structure

Reference tree for the project layout. Files get created when the todo
item that needs them comes up — this is a plan, not a reflection of what
currently exists. Cross-check against `git status` / `find src` for
what's actually been built.

    src/climate_agent/
    ├── api/            app.py, schemas.py              — FastAPI
    ├── frontend/        app.py                          — Streamlit
    ├── agents/
    │   ├── langgraph/   graph.py, state.py, nodes.py    — router → narrator graph
    │   └── crewai/      crew.py, agents.py, tasks.py    — router/narrator crew
    ├── tools/           gwl.py, agriculture.py, biome.py,
    │                    hazard.py, literature.py, analysis.py
    ├── models/          router.py, narrator.py, serve.py — Qwen2.5-7B shared base
    ├── data/            download.py, preprocess.py, schemas.py
    ├── rag/             corpus.py, embed.py, index.py
    ├── ml/
    │   ├── emulator/    train.py, evaluate.py, model.py, registry.py
    │   │                — 3 window-prediction models: gwl (direct), heat_extreme
    │   │                  and precip_extreme (indirect) GWL resolution
    │   └── tool_router/ generate_data.py, train.py, evaluate.py, registry.py
    ├── observability/   metrics.py, tracing.py
    ├── guardrails.py
    └── evaluation/      dataset.jsonl, scorers.py, evaluate_agent.py, evaluate_router.py

    pipelines/kubeflow/, pipelines/local/
    infrastructure/docker/, infrastructure/kubernetes/, infrastructure/kubeflow/
    tests/unit/, tests/agent/, tests/integration/
    scripts/
    .github/workflows/

See `ARCHITECTURE.md` for the reasoning behind each piece.
