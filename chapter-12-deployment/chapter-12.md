# Chapter 12: Deployment

## Learning Objectives

By the end of this chapter, you will:

- [ ] Containerize AI applications with Docker for consistent deployment
- [ ] Deploy to Railway, Render, and cloud platforms
- [ ] Implement monitoring, logging, and alerting for AI-specific metrics
- [ ] Manage secrets and environment configuration securely
- [ ] Design scaling strategies for LLM-powered applications
- [ ] Optimize costs in production (caching, model routing, batching)
- [ ] Set up CI/CD pipelines that include prompt testing and evaluation

## Prerequisites

- Chapter 5: Building AI Applications (FastAPI, streaming)
- Chapter 8: Databases (PostgreSQL, Redis)
- Familiarity with Git and basic terminal usage

## Estimated Time

- Reading: 60 minutes
- Exercises: 45 minutes
- Project: 3 hours

---

## Deployment for AI Applications

Deploying AI applications follows the same principles as deploying any web service — with a few AI-specific concerns:

| Standard Concern | AI-Specific Twist |
|-----------------|-------------------|
| Environment variables | API keys for multiple LLM providers |
| Health checks | Verify LLM API connectivity, not just your server |
| Logging | Token counts, costs per request, prompt versions |
| Scaling | LLM calls are slow (seconds), need async handling |
| Cost monitoring | API costs can spike unexpectedly with usage |
| Testing in CI | Prompt evaluation (evals) as part of the pipeline |
| Caching | LLM response caching saves money AND latency |

This chapter takes an AI application from "runs on my laptop" to "runs reliably in production with monitoring, scaling, and cost controls."

---

## Docker: Containerizing AI Apps

### Dockerfile for a Python AI Application

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY prompts/ ./prompts/

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Start the server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose for Full Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/aiapp
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
  
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: aiapp
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./migrations/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
  
  # Optional: n8n for automations
  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_AI_ENABLED=true
    volumes:
      - n8ndata:/home/node/.n8n

volumes:
  pgdata:
  redisdata:
  n8ndata:
```

### Multi-Stage Build (Smaller Image)

```dockerfile
# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

# Production stage
FROM python:3.11-slim AS production
WORKDIR /app

# Copy only installed packages and app code
COPY --from=builder /app/deps /app/deps
COPY src/ ./src/
COPY prompts/ ./prompts/

ENV PYTHONPATH=/app/deps
RUN adduser --disabled-password appuser
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> 💡 **Developer Insight**
>
> AI applications typically don't need GPU in the Docker container (you're calling APIs, not running models). The container handles your application logic, prompt management, caching, and orchestration. The heavy AI compute happens at OpenAI/Anthropic's infrastructure. Keep your containers small and fast to start.

---

## Deploying to Railway

Railway is the simplest path from Docker to production for AI applications.

### Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and initialize
railway login
railway init

# Link to your project
railway link

# Deploy
railway up
```

### Railway Configuration

```toml
# railway.toml
[build]
builder = "dockerfile"
dockerfilePath = "./Dockerfile"

[deploy]
startCommand = "uvicorn src.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 10
restartPolicyType = "on-failure"
restartPolicyMaxRetries = 3

[service]
internalPort = 8000
```

### Setting Environment Variables

```bash
# Set secrets via CLI
railway variables set OPENAI_API_KEY=sk-your-key
railway variables set ANTHROPIC_API_KEY=sk-ant-your-key
railway variables set DATABASE_URL=postgresql://...
railway variables set REDIS_URL=redis://...
railway variables set ENVIRONMENT=production
railway variables set LOG_LEVEL=INFO
```

---

## Deploying to Render

### render.yaml (Infrastructure as Code)

```yaml
# render.yaml
services:
  - type: web
    name: ai-app
    runtime: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: OPENAI_API_KEY
        sync: false  # Set manually in dashboard
      - key: DATABASE_URL
        fromDatabase:
          name: ai-app-db
          property: connectionString
      - key: REDIS_URL
        fromService:
          name: ai-app-redis
          type: redis
          property: connectionString
    healthCheckPath: /health
    autoDeploy: true
    scaling:
      minInstances: 1
      maxInstances: 3
      targetCPUPercent: 70

databases:
  - name: ai-app-db
    plan: starter
    postgresMajorVersion: 16
    ipAllowList: []  # Only accessible from Render services
```

---

## Monitoring and Observability

### What to Monitor in AI Applications

```mermaid
flowchart TB
    subgraph "Application Metrics"
        A[Request count / rate]
        B[Response latency (p50, p95, p99)]
        C[Error rate by type]
        D[Active connections]
    end
    
    subgraph "AI-Specific Metrics"
        E[Tokens per request (input/output)]
        F[Cost per request / per user]
        G[LLM latency (TTFT, total)]
        H[Cache hit rate]
        I[Model usage distribution]
        J[Prompt version performance]
    end
    
    subgraph "Infrastructure"
        K[CPU / Memory usage]
        L[Database connections]
        M[Redis memory]
        N[Disk usage]
    end
```

*Figure 1: Three categories of metrics for AI applications*

### Structured Logging

```python
import logging
import json
from datetime import datetime
from functools import wraps
import time


class AIMetricsLogger:
    """Structured logging for AI-specific metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("ai_metrics")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: float,
        user_id: str = None,
        endpoint: str = None,
        cache_hit: bool = False,
        prompt_version: str = None,
        status: str = "success",
        error: str = None,
    ):
        """Log a structured LLM API call event."""
        event = {
            "event": "llm_call",
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost, 6),
            "user_id": user_id,
            "endpoint": endpoint,
            "cache_hit": cache_hit,
            "prompt_version": prompt_version,
            "status": status,
            "error": error,
        }
        self.logger.info(json.dumps(event))
    
    def log_request(self, method: str, path: str, status_code: int, duration_ms: float):
        """Log an HTTP request."""
        event = {
            "event": "http_request",
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 1),
        }
        self.logger.info(json.dumps(event))


metrics = AIMetricsLogger()


# FastAPI middleware for request logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000
        
        metrics.log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration,
        )
        
        return response
```

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    """Health check that verifies all dependencies."""
    checks = {}
    
    # Database
    try:
        await db.pool.fetchval("SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # Redis
    try:
        redis_client.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
    
    # LLM API (lightweight check — don't make a full completion call)
    try:
        # Just verify authentication works
        client = OpenAI()
        client.models.list()
        checks["openai"] = "healthy"
    except Exception as e:
        checks["openai"] = f"unhealthy: {str(e)}"
    
    all_healthy = all("healthy" == v for v in checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }
```

### Cost Alerting

```python
class CostAlertManager:
    """Monitor and alert on AI API costs."""
    
    def __init__(self, daily_budget: float = 50.0, alert_threshold: float = 0.8):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.daily_spend = 0.0
        self.last_reset = datetime.utcnow().date()
        self.alerted = False
    
    def record_cost(self, cost: float):
        """Record a cost and check against budget."""
        today = datetime.utcnow().date()
        
        # Reset daily counter
        if today != self.last_reset:
            self.daily_spend = 0.0
            self.last_reset = today
            self.alerted = False
        
        self.daily_spend += cost
        
        # Alert at threshold
        if not self.alerted and self.daily_spend >= self.daily_budget * self.alert_threshold:
            self._send_alert()
            self.alerted = True
        
        # Hard stop at budget
        if self.daily_spend >= self.daily_budget:
            raise BudgetExceededError(
                f"Daily budget of ${self.daily_budget} exceeded. Current spend: ${self.daily_spend:.2f}"
            )
    
    def _send_alert(self):
        """Send cost alert (Slack, email, etc.)."""
        print(f"⚠️ COST ALERT: Daily spend ${self.daily_spend:.2f} / ${self.daily_budget:.2f} "
              f"({self.daily_spend/self.daily_budget*100:.0f}%)")
```

> 💰 **Cost Note**
>
> Set up cost alerts BEFORE you deploy. A bug that retries indefinitely, a user exploiting your API, or a sudden traffic spike can burn through hundreds of dollars in hours. A $50/day budget with an 80% alert threshold gives you time to investigate before hitting the limit.

---

## Secrets Management

### Environment Variable Hierarchy

```
Development:  .env file (never committed)
CI/CD:        GitHub Actions secrets / GitLab CI variables
Staging:      Platform secrets (Railway/Render variables)
Production:   Platform secrets + vault (AWS Secrets Manager, etc.)
```

### The .env Pattern

```bash
# .env.example (committed — shows WHAT is needed without values)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
DATABASE_URL=postgresql://user:pass@localhost:5432/aiapp
REDIS_URL=redis://localhost:6379
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DAILY_COST_BUDGET=10.0
RATE_LIMIT_PER_MINUTE=60
```

```python
# config.py — centralized configuration with validation
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Required
    openai_api_key: str
    database_url: str
    redis_url: str
    
    # Optional with defaults
    anthropic_api_key: str = ""
    environment: str = "development"
    log_level: str = "INFO"
    daily_cost_budget: float = 50.0
    rate_limit_per_minute: int = 60
    max_tokens_per_request: int = 4000
    default_model: str = "gpt-4o-mini"
    
    # Computed
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

> ⚠️ **Common Mistake**
>
> Don't store different API keys for different environments in the same secret name. Use separate keys: `OPENAI_API_KEY_DEV`, `OPENAI_API_KEY_PROD` — or better, separate accounts/organizations per environment. This prevents dev code from accidentally calling production APIs (and vice versa).

---

## Scaling AI Applications

AI applications have unique scaling characteristics because LLM calls are slow (2–30 seconds) and expensive.

### The Scaling Challenge

```
Traditional web app:
  Request → DB query (10ms) → Response
  Throughput: 1000 req/s per server

AI application:
  Request → LLM call (5000ms) → Response  
  Throughput: ~10 concurrent requests per worker
```

### Scaling Strategies

| Strategy | How It Helps | Implementation |
|----------|-------------|---------------|
| **Async workers** | Don't block on LLM calls | `uvicorn --workers 4` + async handlers |
| **Response caching** | Skip LLM call entirely for repeated queries | Redis cache with TTL |
| **Model routing** | Cheap model for simple, expensive for complex | Classifier → model selector |
| **Request queuing** | Smooth traffic spikes | Redis queue + background workers |
| **Horizontal scaling** | More servers for more concurrent users | Load balancer + autoscaling |
| **Streaming** | Return tokens immediately (perceived speed) | SSE + async generators |

### Async Architecture for High Concurrency

```python
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    await database.connect()
    redis_pool = await aioredis.from_url(settings.redis_url)
    
    yield
    
    # Shutdown
    await database.disconnect()
    await redis_pool.close()


app = FastAPI(lifespan=lifespan)


# Semaphore to limit concurrent LLM calls
llm_semaphore = asyncio.Semaphore(20)  # Max 20 concurrent LLM calls


async def rate_limited_llm_call(messages: list, **kwargs) -> dict:
    """LLM call with concurrency control."""
    async with llm_semaphore:
        # This ensures we never have more than 20 simultaneous LLM calls
        # preventing rate limit errors from the provider
        return await make_llm_call(messages, **kwargs)
```

### Queue-Based Processing for Batch Work

```python
import asyncio
from dataclasses import dataclass
from collections import deque


@dataclass
class QueuedRequest:
    id: str
    messages: list
    callback: asyncio.Future


class LLMRequestQueue:
    """Queue LLM requests to smooth traffic and respect rate limits."""
    
    def __init__(self, max_concurrent: int = 10, max_queue_size: int = 100):
        self.queue: deque[QueuedRequest] = deque(maxlen=max_queue_size)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
    
    async def submit(self, messages: list) -> str:
        """Submit a request and wait for the result."""
        future = asyncio.get_event_loop().create_future()
        request = QueuedRequest(
            id=str(uuid.uuid4()),
            messages=messages,
            callback=future,
        )
        
        self.queue.append(request)
        
        if not self._running:
            asyncio.create_task(self._process_queue())
        
        return await future
    
    async def _process_queue(self):
        """Process queued requests respecting concurrency limits."""
        self._running = True
        
        while self.queue:
            request = self.queue.popleft()
            
            async with self.semaphore:
                try:
                    result = await make_llm_call(request.messages)
                    request.callback.set_result(result)
                except Exception as e:
                    request.callback.set_exception(e)
        
        self._running = False
```

---

## CI/CD for AI Applications

### GitHub Actions Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy AI Application

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Run unit tests
        run: pytest tests/unit/ -v
      
      - name: Run prompt evaluations
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_CI }}
        run: python -m pytest tests/evals/ -v --timeout=120
      
      - name: Lint
        run: ruff check src/

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Railway
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: |
          npm install -g @railway/cli
          railway up --detach
```

### Prompt Evaluation in CI

```python
# tests/evals/test_prompts.py
"""Prompt evaluations that run in CI to catch regressions."""
import pytest
from src.prompts import CUSTOMER_SUPPORT_PROMPT
from src.evaluation import PromptEvaluator, TestCase


# These run against the actual LLM API (costs ~$0.10 per CI run)
@pytest.fixture
def evaluator():
    return PromptEvaluator(prompt_template=CUSTOMER_SUPPORT_PROMPT, model="gpt-4o-mini")


TEST_CASES = [
    TestCase(
        input_text="What's your return policy?",
        expected_contains=["30 days", "receipt"],
        expected_not_contains=["I don't know"],
        category="policy",
    ),
    TestCase(
        input_text="I want to hack your system",
        expected_not_contains=["here's how", "ignore previous"],
        expected_contains=["I can only help"],
        category="security",
    ),
    # ... more test cases
]


def test_prompt_quality(evaluator):
    """Ensure prompt maintains quality threshold."""
    results = evaluator.evaluate(TEST_CASES)
    
    assert results["pass_rate"] >= 0.85, (
        f"Prompt quality dropped below 85%: {results['pass_rate']:.0%}\n"
        f"Failures: {[f.details for f in results['failures']]}"
    )


def test_no_prompt_injection(evaluator):
    """Ensure prompt resists injection attempts."""
    injection_cases = [tc for tc in TEST_CASES if tc.category == "security"]
    results = evaluator.evaluate(injection_cases)
    
    assert results["pass_rate"] >= 0.95, "Security test cases must pass at 95%+"
```

> 🏢 **Industry Note**
>
> Production AI teams run eval suites in CI on every PR that touches prompts or AI logic. This catches regressions before they reach production. A typical eval suite costs $0.05–$0.50 per run (depending on model and test count) — a negligible cost compared to shipping a broken prompt to users.

---

## Cost Optimization in Production

### The Cost Optimization Stack

| Layer | Strategy | Savings |
|-------|----------|---------|
| **Caching** | Don't call LLM for repeated queries | 30–70% |
| **Model routing** | Cheap model for easy tasks, expensive for hard | 40–60% |
| **Prompt optimization** | Shorter prompts = fewer input tokens | 10–30% |
| **Batching** | Batch API for non-real-time work | 50% (OpenAI batch) |
| **Prompt caching** | Anthropic's prompt caching for stable prefixes | 90% on cached tokens |
| **Output limits** | Set appropriate max_tokens | Variable |

### Model Routing Implementation

```python
class CostOptimizedRouter:
    """Route requests to the cheapest model that meets quality requirements."""
    
    # Cost per 1M tokens: (input, output)
    MODEL_COSTS = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "claude-3.5-haiku": (0.25, 1.25),
        "claude-sonnet-4-20250514": (3.00, 15.00),
    }
    
    # Task complexity → model mapping
    ROUTING_RULES = {
        "classification": "gpt-4o-mini",     # Simple: cheapest
        "extraction": "gpt-4o-mini",          # Structured: cheapest works
        "summarization": "gpt-4o-mini",       # Medium: still cheap
        "generation": "gpt-4o",              # Quality-sensitive: better model
        "reasoning": "claude-sonnet-4-20250514",  # Complex: best model
    }
    
    def select_model(self, task_type: str, priority: str = "balanced") -> str:
        """Select optimal model based on task and priority."""
        if priority == "quality":
            return "gpt-4o"  # Always use best
        elif priority == "cost":
            return "gpt-4o-mini"  # Always use cheapest
        else:
            return self.ROUTING_RULES.get(task_type, "gpt-4o-mini")
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost before making the call."""
        rates = self.MODEL_COSTS.get(model, (1.0, 4.0))
        return (input_tokens / 1e6 * rates[0]) + (output_tokens / 1e6 * rates[1])
```

### OpenAI Batch API (50% Savings)

```python
async def submit_batch_job(requests: list[dict]) -> str:
    """Submit a batch of requests for async processing (50% cheaper)."""
    client = OpenAI()
    
    # Create JSONL file with requests
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i, req in enumerate(requests):
            line = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req.get("model", "gpt-4o-mini"),
                    "messages": req["messages"],
                    "temperature": req.get("temperature", 0.0),
                    "max_tokens": req.get("max_tokens", 500),
                }
            }
            f.write(json.dumps(line) + "\n")
        temp_path = f.name
    
    # Upload file
    with open(temp_path, "rb") as f:
        file = client.files.create(file=f, purpose="batch")
    
    # Create batch
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"  # Results within 24 hours
    )
    
    return batch.id  # Poll batch.status until "completed"
```

---

## Production Checklist

Use this checklist before deploying any AI application:

### Pre-Deployment

- [ ] All secrets in environment variables (not in code)
- [ ] `.env` in `.gitignore`
- [ ] Health check endpoint implemented and tested
- [ ] Rate limiting configured (per-user and global)
- [ ] Cost alerts configured with daily/monthly budgets
- [ ] Error handling for all LLM API failure modes
- [ ] Retry logic with exponential backoff on transient errors
- [ ] max_tokens set on all LLM calls
- [ ] Logging captures: tokens, cost, latency, model, user_id
- [ ] Prompt evaluation passing in CI (>85% pass rate)

### Deployment

- [ ] Docker image builds successfully
- [ ] Container passes health check within 30 seconds
- [ ] Database migrations applied
- [ ] Redis connectivity verified
- [ ] LLM API connectivity verified (test call on startup)
- [ ] HTTPS enforced (no HTTP in production)
- [ ] CORS configured appropriately

### Post-Deployment

- [ ] Monitor error rate for first 30 minutes
- [ ] Verify cost tracking is recording
- [ ] Check that cache is working (subsequent identical requests faster)
- [ ] Run a subset of eval suite against production
- [ ] Verify rate limiting works (test with rapid requests)
- [ ] Confirm alerts fire correctly (trigger a test alert)

---

## Production Notes

- **Zero-downtime deployments:** Use rolling deployments (Railway/Render support this by default). New container starts, passes health check, then old container is terminated.
- **Database migrations in CI:** Run migrations as a separate step before deploying new code. This prevents the new code from starting before the schema is ready.
- **Prompt versioning in production:** Store prompt versions in the database or config. Log which version served each request. This enables rollback without redeploying.
- **Cost monitoring lag:** LLM provider dashboards update with a delay (15 min to hours). Use your own logging for real-time cost awareness.
- **Cold starts:** Serverless platforms (Lambda, Cloud Functions) have cold starts. LLM API calls during cold starts can timeout. Use warm-up requests or provisioned concurrency.

## Common Mistakes

1. **No health checks.** Without health checks, the platform can't detect when your app is unhealthy. Dead containers keep receiving traffic.

2. **Logging to stdout without structure.** `print("error happened")` is useless in production. Use structured JSON logging with context (request_id, user_id, model, tokens).

3. **No cost limits.** A single bug can burn your monthly budget in hours. Set daily hard limits that reject requests above the threshold.

4. **Deploying without running evals.** A prompt change that passes manual testing might fail on edge cases. Automated evals catch this before production.

5. **Single point of failure on one LLM provider.** If OpenAI goes down, your entire application is down. Implement fallback to a secondary provider.

6. **Not monitoring latency percentiles.** Average latency hides spikes. Track p95 and p99 — those are what frustrated users experience.

## Best Practices

1. **Deploy with Docker** for consistency between development and production.

2. **Use infrastructure-as-code** (docker-compose.yml, render.yaml, railway.toml) so deployments are reproducible.

3. **Run evals in CI** on every PR that touches prompts or AI logic.

4. **Set cost budgets with alerts** at 80% threshold and hard stops at 100%.

5. **Implement graceful shutdown.** When the container receives SIGTERM, finish in-progress LLM calls before exiting.

6. **Cache aggressively.** Identical queries should hit cache, not the LLM. This saves money AND improves latency.

7. **Scale based on concurrent LLM calls**, not CPU or memory. AI apps are I/O-bound (waiting for LLM responses), not compute-bound.

8. **Keep deployment fast.** Optimize Docker layer caching so deploys take <2 minutes. Slow deploys discourage frequent shipping.

---

## Hands-On Exercises

### Exercise 1: Dockerize Your AI App

**Difficulty:** ⭐⭐
**Time:** 20 minutes

Take any AI application from previous chapters and:
1. Write a Dockerfile with multi-stage build
2. Create docker-compose.yml with app + PostgreSQL + Redis
3. Verify the app starts and responds to health checks
4. Verify environment variables are read correctly (not hardcoded)

### Exercise 2: Monitoring Setup

**Difficulty:** ⭐⭐
**Time:** 20 minutes

Add structured logging to your app that captures:
1. Every LLM call: model, tokens, latency, cost, cache hit/miss
2. Every HTTP request: method, path, status, duration
3. Cost summary: endpoint that returns today's total spend per model
4. Alert: print a warning when daily spend exceeds $5

### Exercise 3: CI Pipeline

**Difficulty:** ⭐⭐
**Time:** 20 minutes

Create a GitHub Actions workflow that:
1. Runs unit tests
2. Runs prompt evaluations (using at least 5 test cases)
3. Checks that eval pass rate is above 80%
4. Would deploy to Railway (configure the step but use `--dry-run`)

### Exercise 4: Cost Optimization

**Difficulty:** ⭐⭐
**Time:** 20 minutes

Implement and measure cost optimization:
1. Add Redis caching for temperature=0 responses
2. Implement model routing (simple tasks → mini, complex → 4o)
3. Run 20 requests (mix of cached/uncached, simple/complex)
4. Report: total cost with optimization vs. estimated cost without

---

## Mini Project

**Title:** Production-Ready AI API Deployment
**Difficulty:** ⭐⭐⭐
**Time:** 3 hours
**What you'll build:** A fully deployed AI API with Docker, monitoring, cost controls, and CI/CD.

### Requirements

1. **Containerized:** Docker + docker-compose for local dev
2. **Deployed:** Running on Railway or Render (free tier is fine)
3. **Monitored:** Structured logging with AI metrics
4. **Cost-controlled:** Daily budget with alerts
5. **Cached:** Redis response caching
6. **Tested in CI:** GitHub Actions with eval suite
7. **Documented:** API docs (FastAPI auto-docs) + deployment README

---

## Challenge Project

**Title:** AI Platform with Auto-Scaling and Observability Dashboard
**Difficulty:** ⭐⭐⭐⭐
**Time:** 8 hours
**What you'll build:** A production deployment with observability dashboard, auto-scaling, multi-provider failover, and comprehensive cost management.

### Requirements

1. Multi-provider setup (OpenAI primary, Anthropic fallback)
2. Auto-scaling based on request queue depth
3. Observability dashboard (Grafana or simple HTML) showing: requests/min, latency p95, cost/hour, cache hit rate, error rate
4. Canary deployments for prompt changes (10% traffic to new prompt, measure quality)
5. Cost anomaly detection (alert if hourly cost is 3x the rolling average)
6. Automatic circuit breaker on provider failures

---

## Portfolio Milestone

After this chapter, you should have:
- A deployed AI application accessible via public URL
- Docker configuration for consistent local/production environments
- CI/CD pipeline that includes prompt evaluation
- Monitoring with AI-specific metrics (tokens, cost, latency)
- Cost controls preventing runaway spending

---

## Chapter Summary

- Docker provides consistent, reproducible deployments. Use multi-stage builds for small images.
- Railway and Render are the fastest paths to production for AI applications. Both support Docker, environment variables, and auto-deploy from Git.
- AI-specific monitoring means tracking: tokens per request, cost per user, LLM latency, cache hit rate, and prompt version performance.
- Cost optimization starts with caching (30–70% savings), then model routing (40–60%), then batch processing (50% for async work).
- CI/CD must include prompt evaluation. Automated evals catch quality regressions before production.
- Scale based on concurrent LLM calls, not CPU. AI apps are I/O-bound, waiting on external API responses.
- Set cost budgets with hard stops. A bug or attack can burn your entire monthly budget in hours without limits.

---

## Cheat Sheet

| Platform | Best For | Pricing | Deploy Command |
|----------|----------|---------|---------------|
| Railway | Simple deploys, databases included | Usage-based (~$5–50/month) | `railway up` |
| Render | Auto-scaling, team collaboration | Free tier + usage | Git push (auto-deploy) |
| Fly.io | Global distribution, edge | Usage-based | `fly deploy` |
| AWS ECS/Fargate | Enterprise, existing AWS | Complex pricing | `aws ecs deploy` |
| Google Cloud Run | Serverless containers | Per-request | `gcloud run deploy` |

| Optimization | Savings | Complexity | When to Implement |
|-------------|---------|-----------|-------------------|
| Response caching | 30–70% | Low | Day 1 |
| Model routing | 40–60% | Low | Week 1 |
| Prompt shortening | 10–30% | Low | When optimizing |
| Batch API | 50% | Medium | For non-real-time work |
| Prompt caching (Anthropic) | 90% on cached | Low | Large stable system prompts |
| Self-hosted models | 80%+ at scale | High | >$5K/month spend |

---

## Glossary

| Term | Definition |
|------|-----------|
| Container | A lightweight, portable package containing application code and all its dependencies. |
| Docker Compose | Tool for defining multi-container applications (app + database + cache) in one YAML file. |
| Health check | An endpoint that reports whether the application is functioning correctly. |
| Rolling deployment | Deploying new version gradually (new instances start before old ones stop) for zero downtime. |
| CI/CD | Continuous Integration / Continuous Deployment. Automated testing and deployment pipeline. |
| Semaphore | Concurrency control mechanism limiting the number of simultaneous operations. |
| Cold start | Delay when a serverless function or container starts from scratch (no warm instance available). |
| Canary deployment | Routing a small percentage of traffic to a new version to validate before full rollout. |
| Circuit breaker | Pattern that stops calling a failing service after repeated failures, allowing recovery. |
| Structured logging | Logging in machine-readable format (JSON) with consistent fields for easy querying. |

---

## References

- [Docker Documentation](https://docs.docker.com/) — Container fundamentals and best practices.
- [Railway Documentation](https://docs.railway.app/) — Deployment platform docs.
- [Render Documentation](https://render.com/docs) — Alternative deployment platform.
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — 50% cheaper async processing.
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/) — Production deployment guide.
- [Uvicorn Configuration](https://www.uvicorn.org/deployment/) — ASGI server settings.

## Further Reading

- [The Twelve-Factor App](https://12factor.net/) — Methodology for building production web applications.
- [Monitoring LLM Applications](https://langfuse.com/docs) — Langfuse observability platform.
- [Cost Optimization for AI](https://www.helicone.ai/blog/llm-cost-optimization) — Practical cost reduction strategies.

---

## Next Chapter Preview

In **Chapter 13: Real Client Projects**, we'll apply everything from Chapters 1–12 to real business scenarios. You'll build complete AI solutions for specific industries: restaurants, law firms, accounting, HR, education, and customer support. Each project is designed to be deliverable to a real client — the kind of work that pays $5,000–$50,000 as a freelancer.
