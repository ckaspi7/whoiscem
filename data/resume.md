# Cem Kaspi

Vancouver, BC · [linkedin.com/in/cemkaspi](https://linkedin.com/in/cemkaspi)

## Summary

AI/ML Engineer who takes GenAI systems from proof of concept to production and keeps them
accountable once there — agentic applications, RAG pipelines, evaluation harnesses, and the guardrails
and observability that make them safe to operate. Nearly five years at TELUS on GCP and Azure across
LLM systems and classical ML, including a CEO-requested investor-relations agent shipped with
securities-compliance guardrails. Teaches applied AI to engineering cohorts and advises US client teams
on agent architecture. At home in stakeholder discovery and in the Python service that runs the thing.

## Technical Skills

| | |
|---|---|
| **Languages** | Python, JavaScript / TypeScript (Node.js, Apps Script), SQL, C/C++ |
| **GenAI & Agents** | LangGraph, LangChain, OpenAI Agents SDK, MCP, LLM function-calling, RAG, multi-model fallback |
| **Retrieval** | TurboPuffer, Azure AI Search, chunking & embedding pipelines, sentence-transformers, RedisVL |
| **Evaluation & Observability** | DeepEval, golden-dataset suites, Arize + OpenInference / OpenTelemetry tracing, Langfuse, MLflow |
| **Backend & Data** | FastAPI, Pydantic, Uvicorn, PostgreSQL, Redis, Firestore, BigQuery, SQLAlchemy |
| **Cloud & Infrastructure** | GCP (Cloud Run, Functions, Vertex AI, Secret Manager, KMS, DLP), Azure, Docker, Terraform, CI/CD (GitHub Actions) |
| **Machine Learning** | XGBoost, Random Forest, regression & time-series forecasting, computer vision, scikit-learn |
| **Engineering Practice** | pytest (asyncio / mock / coverage), integration & regression testing, code review, responsible AI and sensitive-data handling |
| **Tools** | Claude Code, Git, Splunk, JIRA, n8n, Streamlit, Appian |

## Experience

### TELUS Communications — AI/ML Engineer
**Vancouver, BC · January 2022 – Present**

**TELUS AI Financial Agent** — production RAG agent answering questions on TELUS annual reports,
initiated by direct request from the CEO.

- Built the document-to-vector ingestion pipeline (TurboPuffer) with primary and fallback embedding
  models, namespace health-check tooling, and surgical chunk replacement for targeted updates.
- Created the golden-dataset evaluation suite — DeepEval scoring and config-comparison notebooks —
  and ran iterative evaluation rounds including French-language feedback cycles.
- Implemented securities-compliance guardrails rejecting forward-looking, investment-advice,
  competitor-comparison, and non-public queries, returning source-grounded answers with inline citations.
- Contributed to the LangGraph workflow (query rewrite, guardrail validation, answer generation), added
  multi-LLM fallback for graceful degradation, and instrumented per-node token and latency tracking.
- Recognized internally as the second financial AI agent deployed in North America, after NVIDIA.

**eChange Bot / Unified Ops Copilot** — Google Chat copilot for network change management, taken
from PoC to production on Cloud Run.

- Led end-to-end design and delivery; migrated storage to Firestore for scalability, added caching on
  hot paths, and resolved OAuth integration with the change-management platform team.
- Built an intent-based LLM function-calling router that classifies each query (RFC detail / conflict /
  restriction / general) and calls only the relevant API, improving both accuracy and latency.
- Shipped the Restriction Event Change Manager: deny-by-default compliance logic,
  reschedule-versus-exception guidance, and timezone-aware freeze-window detection across ET/PT/MT.
- ~$400K estimated annual OPEX avoidance in conflict resolution, within $728K of broader
  change-management savings.

**Platform & engineering practice**

- Built the Python service layer behind these systems — FastAPI with Pydantic validation, containerized
  with Docker on Cloud Run, infrastructure and IAM managed in Terraform, and pytest suites covering
  async paths, mocked LLM calls, and retrieval regressions in CI.
- Built custom knowledge bases on Azure vector search serving 1,000+ cross-functional users; integrated
  APIs and GenAI capabilities across Azure and GCP to insource formerly outsourced work, reducing the
  need for 250 on-shore field technicians.

**Machine learning & forecasting**

- Replaced manual weekly averaging for RAN load balancing with predictive capacity forecasting
  (regression over 365-day rolling averages, 9-day horizon); deployed daily ingestion to the BI layer,
  moving optimizer tuning from a weekly to a daily cadence.
- Built a network churn classifier (XGBoost, Random Forest) over multi-source network KPIs to identify
  postpaid customers at risk from coverage and signal-quality issues, enabling proactive retention.
- Co-led a computer vision human-detection model for search-and-rescue scenarios at 80% accuracy,
  scoped at $630K in potential new annual revenue.

**Adoption & influence**

- Placed 1st and 2nd among 73 submissions in an organization-wide Innovation Challenge, securing $60K
  for two AI/ML projects that generated $572K in combined annual savings, presented to C-suite.
- Presented to SVPs, VPs and Directors, mentored new engineers, and published a reusable blueprint
  enabling other teams to build their own assistants. Contributed to $6.5M in annual OPEX savings
  delivered by the team.

### Great Learning — AI/ML Industry Mentor (UT Austin Program)
**Remote · September 2025 – Present**

- Mentor 100+ working professionals through the Post Graduate Program in AI & Machine Learning
  co-developed with the University of Texas at Austin — 90+ engagement hours at a 4.9+/5.0 rating.
- Teach the full arc: predictive modeling, ensemble methods and neural networks through to generative
  AI, LLMs, RAG, and single- and multi-agent systems, with a throughline on deployment and
  Responsible AI guardrails in production.
- Review enterprise capstone cycles end to end — code architecture, methodology critique, and
  production-readiness feedback.

### Independent AI Advisory — GLG, Tegus
**Remote · September 2025 – Present**

- Advise US client teams through expert networks on applied AI: agent architecture, LLM tooling and
  model selection, vendor evaluation, and AI infrastructure decisions.

### NeoWise — Co-Founder
**Vancouver, BC · September 2019 – May 2021**

- Co-founded a wearable thermal-device startup; led the prototype from concept to MVP across R&D,
  hardware design, and supplier negotiation (±5 °C skin-temperature regulation). Admitted to
  entrepreneurship@UBC's CORE Incubator; pitched investors and ran customer discovery.

### Mercedes-Benz Canada — Manufacturing Engineer, Co-op
**Burnaby, BC · January 2018 – August 2019**

- Led data-driven initiatives that cut material scrap loss by over $100K/year and raised production yield
  3%; improved station efficiency 25% through Lean Six Sigma value-stream mapping.

## Selected Projects

- **AI Travel & Productivity Assistant** — multi-step automation app orchestrating LLMs against external
  APIs via n8n, with conditional branching, webhook triggers, and multi-service connectors.

## Education

**BASc, Electrical Engineering** — University of British Columbia, Vancouver, BC · 2015 – 2021 ·
Minor in Entrepreneurship · Dean's List, 2019–2021
