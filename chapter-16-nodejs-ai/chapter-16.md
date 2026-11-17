# Chapter 16: Node.js + AI

## Learning Objectives

By the end of this chapter, you will:

- [ ] Set up a strict TypeScript Node.js service that calls LLM APIs safely
- [ ] Stream completions to clients over Server-Sent Events (SSE) with cancellation
- [ ] Design AI endpoints in Fastify with schema validation, authentication, and rate limits
- [ ] Return validated structured output from an LLM using JSON Schema and `zod`
- [ ] Build a Retrieval-Augmented Generation (RAG) endpoint on PostgreSQL with pgvector
- [ ] Move slow AI work into background jobs with BullMQ
- [ ] Control cost, handle provider errors, test without calling real models, and observe production traffic

## Prerequisites

- Chapter 3: Working with LLM APIs (request formats, tokens, streaming)
- Chapter 5: Building AI Applications (streaming and API patterns)
- Chapter 7: RAG Pipelines (retrieval concepts)
- Chapter 8: Databases for AI Applications (PostgreSQL, pgvector, Redis)
- Node.js 22 LTS or newer and working knowledge of TypeScript and `async`/`await`

## Estimated Time

- Reading: 80 minutes
- Exercises: 75 minutes
- Project: 5 hours

---

## Why Node.js for AI Backends

Most chapters in this book use Python. Python is the right choice for model training, data work, and heavy ML libraries. It is not the only choice for serving AI features.

Calling an LLM is network-bound work. A request waits seconds for tokens, and a server may hold thousands of these requests open at once. Node.js handles many concurrent, mostly idle connections well because of its event loop. If your team already ships TypeScript for web and mobile clients, one language across the stack also reduces friction.

| Factor | Node.js (TypeScript) | Python (FastAPI) |
|--------|---------------------|------------------|
| Long-lived streaming connections | Strong, event loop is built for it | Strong with `async`, needs care with blocking code |
| Shared types with a web or React Native client | Yes | No |
| ML and data science libraries | Limited | Extensive |
| Local model inference, training | Rare | Standard |
| Official provider SDKs | OpenAI, Anthropic, Google, AWS | All of them, usually first |

> 📐 **Architecture Decision**
>
> Use Node.js when the AI feature is an API layer over hosted models: chat, extraction, RAG, and orchestration. Use Python when you need ML libraries, evaluation notebooks, or local inference. Many teams run both and keep the boundary at an HTTP endpoint.

Do not use Node.js for CPU-heavy work such as embedding generation with a local model or large document parsing in the request path. Those tasks block the event loop. We move them to worker processes later in this chapter.

---

## Architecture of an AI Backend

An AI endpoint is a normal endpoint with three extra concerns: long latency, per-request cost, and a failure-prone upstream dependency.

```mermaid
flowchart TB
    Client[Web or Mobile Client] -->|HTTPS + SSE| API[Fastify API]
    API --> Auth[Auth and Rate Limit]
    Auth --> Budget[Token Budget Check]
    Budget --> LLM[LLM Provider]
    API --> Retrieval[Retriever]
    Retrieval --> PG[(PostgreSQL + pgvector)]
    API -->|enqueue| Queue[(Redis + BullMQ)]
    Queue --> Worker[Worker Process]
    Worker --> LLM
    API --> Logs[Structured Logs and Metrics]
```

*Figure 1: Components of a Node.js AI backend*

The API process stays thin. It validates input, authenticates the caller, enforces limits, and either answers within the request or hands work to a queue.

---

## Project Setup

### Dependencies

We use Fastify 5 for the HTTP layer. It validates requests against schemas, has a built-in logger (`pino`), and has a small plugin system. Express works too, and the patterns in this chapter transfer directly. Fastify gives us schema validation and structured logging without extra packages.

```bash
mkdir ai-api && cd ai-api
npm init -y
npm install fastify @fastify/rate-limit openai zod pg pgvector ioredis bullmq dotenv
npm install -D typescript tsx vitest @types/node @types/pg
```

Check the installed versions with `npm ls` and pin major versions in `package.json`. The `openai` package releases major versions often. The chat completions calls in this chapter have stayed stable across them.

> ⚠ **Evolving Landscape**
>
> This chapter covers SDK usage as of mid-2026. Provider SDKs and model names change often. Check the official documentation before copying model identifiers.

### TypeScript Configuration

```json
{
  "compilerOptions": {
    "target": "ES2023",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "outDir": "dist",
    "rootDir": "src",
    "esModuleInterop": true,
    "skipLibCheck": true
  },
  "include": ["src"]
}
```

With `module` set to `NodeNext`, relative imports need the `.js` extension in source files. The examples follow that rule, and you must also set `"type": "module"` in `package.json`.

### Project Structure

```
ai-api/
├── src/
│   ├── config.ts
│   ├── server.ts
│   ├── app.ts
│   ├── llm/
│   │   ├── client.ts
│   │   ├── stream.ts
│   │   └── structured.ts
│   ├── plugins/
│   │   ├── auth.ts
│   │   └── errors.ts
│   ├── routes/
│   │   ├── chat.ts
│   │   ├── extract.ts
│   │   ├── rag.ts
│   │   └── jobs.ts
│   ├── rag/
│   │   └── retriever.ts
│   ├── queue/
│   │   ├── queues.ts
│   │   └── worker.ts
│   └── budget.ts
├── tests/
├── .env.example
├── Dockerfile
├── package.json
└── tsconfig.json
```

### Configuration and Secrets

Validate environment variables once at startup. A missing key should stop the process before it serves traffic.

```bash
# .env.example
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=postgres://user:password@localhost:5432/ai
REDIS_URL=redis://localhost:6379
API_KEYS=dev-key-1,dev-key-2
DAILY_TOKEN_BUDGET=200000
```

```typescript
// src/config.ts
import "dotenv/config";
import { z } from "zod";

const EnvSchema = z.object({
  NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
  PORT: z.coerce.number().default(3000),
  OPENAI_API_KEY: z.string().min(1),
  OPENAI_MODEL: z.string().default("gpt-4o-mini"),
  DATABASE_URL: z.string().min(1),
  REDIS_URL: z.string().min(1),
  API_KEYS: z.string().min(1), // comma-separated client keys
  DAILY_TOKEN_BUDGET: z.coerce.number().default(200_000),
});

// Throws with a readable message if any variable is missing or malformed.
export const config = EnvSchema.parse(process.env);
```

Add `.env` to `.gitignore`. Never commit it.

---

## Calling LLM APIs from Node.js

### The Client

Create one client for the whole process. The OpenAI SDK retries connection errors, `408`, `409`, `429`, and `5xx` responses with exponential backoff. We set the retry count and a timeout explicitly instead of relying on defaults.

```typescript
// src/llm/client.ts
import OpenAI from "openai";
import { config } from "../config.js";

export const openai = new OpenAI({
  apiKey: config.OPENAI_API_KEY,
  maxRetries: 3,       // SDK handles backoff for 429 and 5xx
  timeout: 30_000,     // ms; fail instead of holding a connection forever
});

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};
```

> 💡 **Developer Insight**
>
> Retries multiply latency. Three retries with a 30-second timeout can hold a request for over a minute. For user-facing endpoints, use a lower timeout and fewer retries. Keep the higher values for background jobs.

### Provider Abstraction

Provider lock-in is a real tradeoff. A thin interface around your own types keeps provider details in one folder. Libraries such as the Vercel AI SDK and LangChain.js offer broader abstractions. They add dependencies and their own release cadence, so adopt them when you need multi-provider support, not by default.

| Option | Pros | Cons | Use When |
|--------|------|------|----------|
| Official SDK directly | Full feature access, fewer layers | Provider-specific types | One provider, simple flows |
| Own thin wrapper | Swappable, testable | You maintain it | Production services (recommended) |
| Vercel AI SDK | Streaming helpers, multi-provider | Extra abstraction | Full-stack TypeScript with React |
| LangChain.js | Many integrations | Heavy, fast-changing API | Prototyping agent graphs |

---

## Streaming with Server-Sent Events

As we covered in **Chapter 5: Building AI Applications**, streaming reduces perceived latency. Server-Sent Events (SSE) is the standard transport for it. SSE is a one-way HTTP stream of `data:` lines. Browsers support it natively through `EventSource`, and the Flutter client in **Chapter 11: Flutter + AI** parses the same format.

Use WebSockets instead when the client also streams data to the server, such as live audio. For text chat, SSE is simpler and works through most proxies.

### Streaming Generator

```typescript
// src/llm/stream.ts
import { openai, type ChatMessage } from "./client.js";
import { config } from "../config.js";

export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "usage"; totalTokens: number };

export async function* streamCompletion(
  messages: ChatMessage[],
  signal: AbortSignal,
): AsyncGenerator<StreamEvent, void, void> {
  const stream = await openai.chat.completions.create(
    {
      model: config.OPENAI_MODEL,
      messages,
      stream: true,
      stream_options: { include_usage: true }, // final chunk carries token counts
    },
    { signal }, // aborting cancels the upstream request
  );

  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content;
    if (text) yield { type: "token", text };
    if (chunk.usage) yield { type: "usage", totalTokens: chunk.usage.total_tokens };
  }
}
```

An async generator keeps the route code free of SDK details. The `signal` matters for cost: when a user closes the tab, you stop paying for tokens nobody reads.

### The Streaming Endpoint

```typescript
// src/routes/chat.ts
import type { FastifyInstance } from "fastify";
import { z } from "zod";
import { streamCompletion } from "../llm/stream.js";
import { assertWithinBudget, consumeBudget } from "../budget.js";

const Body = z.object({
  messages: z
    .array(z.object({
      role: z.enum(["user", "assistant"]),
      content: z.string().min(1).max(8_000),
    }))
    .min(1)
    .max(40),
});

export async function chatRoutes(app: FastifyInstance): Promise<void> {
  app.post("/v1/chat/stream", async (req, reply) => {
    const parsed = Body.safeParse(req.body);
    if (!parsed.success) return reply.code(400).send({ error: "invalid_request" });

    await assertWithinBudget(req.clientId); // throws 429 before any tokens are spent

    const controller = new AbortController();
    reply.hijack(); // we write to the raw socket ourselves
    reply.raw.on("close", () => {
      // The request's own "close" event fires once the body is read, so we
      // listen on the response. An unfinished response means the client left.
      if (!reply.raw.writableFinished) controller.abort();
    });
    reply.raw.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no", // stop nginx from buffering the stream
    });
```

The route continues with the streaming loop.

```typescript
    const system = { role: "system" as const, content: "You are a concise assistant." };
    try {
      for await (const event of streamCompletion([system, ...parsed.data.messages], controller.signal)) {
        if (event.type === "token") {
          reply.raw.write(`data: ${JSON.stringify({ token: event.text })}\n\n`);
        } else {
          await consumeBudget(req.clientId, event.totalTokens);
        }
      }
      reply.raw.write("data: [DONE]\n\n");
    } catch (err) {
      req.log.error({ err }, "stream failed");
      reply.raw.write(`event: error\ndata: ${JSON.stringify({ error: "upstream_failed" })}\n\n`);
    } finally {
      reply.raw.end();
    }
  });
}
```

We listen for `close` on the response, not the request. The request emits `close` as soon as its body is consumed, which would abort every stream immediately. The response emits `close` when the connection ends, and `writableFinished` tells us whether we completed normally.

Two details deserve attention. First, once headers are sent, the status code is fixed. Errors mid-stream must travel as SSE events, not as HTTP statuses. Second, we JSON-encode each token so that newlines inside a token cannot break the SSE framing.

> ⚠ **Common Mistake**
>
> Putting compression middleware or a buffering reverse proxy in front of an SSE endpoint. Tokens then arrive in large batches, and the stream looks frozen. Disable buffering for this path and test through your real load balancer.

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Fastify API
    participant L as LLM Provider
    C->>A: POST /v1/chat/stream
    A->>L: create(stream: true)
    L-->>A: chunk "Hel"
    A-->>C: data: {"token":"Hel"}
    L-->>A: chunk "lo"
    A-->>C: data: {"token":"lo"}
    C--xA: connection closed
    A->>L: abort signal
```

*Figure 2: Streaming with client disconnect and upstream cancellation*

---

## Structured Output

Backends rarely want prose. They want a typed object to store, route, or return. Structured output asks the model to follow a JSON Schema, and the provider constrains decoding to match it. This is more reliable than asking for JSON in the prompt, which is the older "JSON mode" approach.

Constrained output guarantees valid shape, not correct content. Always validate the parsed result again in your own code, and treat the values as untrusted input.

We define the schema once with `zod` and derive the JSON Schema from it. Zod 4 includes `z.toJSONSchema`.

```typescript
// src/llm/structured.ts
import { z } from "zod";
import { openai, type ChatMessage } from "./client.js";
import { config } from "../config.js";

export async function extractStructured<T extends z.ZodType>(
  schema: T,
  schemaName: string,
  messages: ChatMessage[],
): Promise<z.infer<T>> {
  const completion = await openai.chat.completions.create({
    model: config.OPENAI_MODEL,
    messages,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: schemaName,
        strict: true,
        schema: z.toJSONSchema(schema) as Record<string, unknown>,
      },
    },
  });

  const choice = completion.choices[0];
  if (!choice?.message.content || choice.message.refusal) {
    throw new Error("Model refused or returned no content");
  }
  return schema.parse(JSON.parse(choice.message.content)); // validate again
}
```

Strict mode supports a subset of JSON Schema. Every field must be required, and optional values are modeled as nullable. Keep schemas flat where possible.

```typescript
// src/routes/extract.ts (excerpt)
const Ticket = z.object({
  category: z.enum(["billing", "bug", "feature_request", "other"]),
  priority: z.enum(["low", "medium", "high"]),
  summary: z.string().max(200),
  customerEmail: z.string().nullable(),
});

app.post("/v1/tickets/classify", async (req, reply) => {
  const { text } = z.object({ text: z.string().min(1).max(10_000) }).parse(req.body);
  const ticket = await extractStructured(Ticket, "ticket", [
    { role: "system", content: "Classify the support ticket. Use null if no email appears." },
    { role: "user", content: text },
  ]);
  return reply.send(ticket);
});
```

> 💡 **Developer Insight**
>
> Keep field descriptions in the schema. Models use `describe()` text as extra instructions, and it costs less than a longer system prompt.

---

## Authentication, Rate Limiting, and Budgets

An unauthenticated AI endpoint is an open wallet. Every request costs money, so we limit three things: who can call, how often, and how many tokens they can spend.

### API Key Authentication

```typescript
// src/plugins/auth.ts
import { timingSafeEqual } from "node:crypto";
import type { FastifyInstance } from "fastify";
import { config } from "../config.js";

declare module "fastify" {
  interface FastifyRequest { clientId: string }
}

const keys = config.API_KEYS.split(",").map((k) => Buffer.from(k.trim()));

function isValid(candidate: string): boolean {
  const buf = Buffer.from(candidate);
  // Compare against every key to avoid leaking which prefix matched.
  return keys.reduce(
    (ok, k) => (k.length === buf.length && timingSafeEqual(k, buf)) || ok,
    false,
  );
}

export async function authPlugin(app: FastifyInstance): Promise<void> {
  app.decorateRequest("clientId", "");
  app.addHook("onRequest", async (req, reply) => {
    if (req.url.startsWith("/health")) return;
    const key = req.headers["x-api-key"];
    if (typeof key !== "string" || !isValid(key)) {
      return reply.code(401).send({ error: "unauthorized" });
    }
    req.clientId = key.slice(0, 8); // never log or store the full key
  });
}
```

Static API keys suit service-to-service use. For end users, verify a JSON Web Token (JWT) from your identity provider and set `clientId` to the user ID.

### Rate Limiting

Rate limits protect your upstream quota and your bill. Back the limiter with Redis so the limit holds across multiple server instances.

```typescript
// src/app.ts (excerpt)
import rateLimit from "@fastify/rate-limit";
import { Redis } from "ioredis";

const redis = new Redis(config.REDIS_URL);

await app.register(rateLimit, {
  redis,
  max: 20,
  timeWindow: "1 minute",
  keyGenerator: (req) => req.clientId || req.ip,
});
```

Register the authentication plugin first so `clientId` is set before the limiter reads it. Request-count limits are not enough for LLM traffic, because one request can cost 100 tokens or 100,000.

### Token Budgets

```typescript
// src/budget.ts
import { Redis } from "ioredis";
import { config } from "./config.js";

const redis = new Redis(config.REDIS_URL);
const dayKey = (id: string): string =>
  `budget:${id}:${new Date().toISOString().slice(0, 10)}`;

export async function assertWithinBudget(clientId: string): Promise<void> {
  const used = Number((await redis.get(dayKey(clientId))) ?? 0);
  if (used >= config.DAILY_TOKEN_BUDGET) {
    throw Object.assign(new Error("Daily token budget exceeded"), { statusCode: 429 });
  }
}

export async function consumeBudget(clientId: string, tokens: number): Promise<void> {
  const key = dayKey(clientId);
  await redis.incrBy(key, tokens);
  await redis.expire(key, 60 * 60 * 36); // outlive the day, then disappear
}
```

Call `assertWithinBudget` at the start of each AI route and `consumeBudget` once usage is known. The check is approximate: concurrent requests can overshoot slightly. That is acceptable for cost protection. Use a stricter reservation scheme only if you bill customers by usage.

---

## Retrieval-Augmented Generation Endpoint

Retrieval-Augmented Generation (RAG) grounds the LLM in your own data. We covered the concepts in **Chapter 7: RAG Pipelines** and pgvector in **Chapter 8: Databases for AI Applications**. Here we wire retrieval into a Node.js endpoint.

Assume a `chunks` table with an `embedding vector(1536)` column populated at ingestion time using the `text-embedding-3-small` model.

```typescript
// src/rag/retriever.ts
import pg from "pg";
import pgvector from "pgvector/pg";
import { openai } from "../llm/client.js";
import { config } from "../config.js";

export const pool = new pg.Pool({ connectionString: config.DATABASE_URL, max: 10 });
pool.on("connect", async (client) => { await pgvector.registerTypes(client); });

export type Chunk = { id: number; source: string; content: string; score: number };

export async function retrieve(question: string, tenantId: string, k = 5): Promise<Chunk[]> {
  const emb = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: question,
  });
  const vector = emb.data[0]?.embedding;
  if (!vector) throw new Error("Embedding request returned no data");

  // <=> is cosine distance. The tenant filter is part of the query, not post-processing.
  const { rows } = await pool.query<Chunk>(
    `SELECT id, source, content, 1 - (embedding <=> $1) AS score
       FROM chunks
      WHERE tenant_id = $2
      ORDER BY embedding <=> $1
      LIMIT $3`,
    [pgvector.toSql(vector), tenantId, k],
  );
  return rows;
}
```

Use parameterized queries as shown. Never interpolate user text into SQL. The tenant filter belongs in the query so that retrieval cannot cross customer boundaries.

```typescript
// src/routes/rag.ts (excerpt)
app.post("/v1/ask", async (req, reply) => {
  const { question } = z.object({ question: z.string().min(1).max(2_000) }).parse(req.body);
  await assertWithinBudget(req.clientId);

  const chunks = await retrieve(question, req.clientId);
  const context = chunks.filter((c) => c.score > 0.3)
    .map((c, i) => `[${i + 1}] (${c.source}) ${c.content}`)
    .join("\n\n");

  if (!context) return reply.send({ answer: "I could not find this in the documents.", sources: [] });

  const answer = await extractStructured(Answer, "answer", [
    { role: "system", content: "Answer only from the context. Cite sources by number. If the context is insufficient, say so." },
    { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` },
  ]);
  return reply.send({ ...answer, sources: chunks.map((c) => c.source) });
});
```

The score threshold of `0.3` is a starting value. Tune it with real queries, as described in **Chapter 7: RAG Pipelines**. The `Answer` schema is a `zod` object with an `answer` string and a `citations` array of numbers.

> 🔑 **Key Concept**
>
> Retrieved text is untrusted input. A document can contain instructions that try to hijack the prompt. Delimit the context clearly, instruct the model to treat it as data, and never give the model tools that can act on its own output without checks.

---

## Background Jobs and Queues

Some AI work does not fit in a request: summarizing a 200-page document, embedding thousands of chunks, or running a multi-step agent. HTTP clients time out, and load balancers close idle connections. Queue the work and return a job ID.

BullMQ is a Redis-backed queue for Node.js. It provides retries with backoff, delayed jobs, concurrency control, and rate limiting. Alternatives include Amazon SQS (see **Chapter 17: AWS Cloud + AI**), RabbitMQ, and Postgres-based queues such as `pg-boss`. Choose BullMQ when you already run Redis and want a Node-native API.

```typescript
// src/queue/queues.ts
import { Queue } from "bullmq";
import { Redis } from "ioredis";
import { config } from "../config.js";

// BullMQ workers require maxRetriesPerRequest to be null.
export const connection = new Redis(config.REDIS_URL, { maxRetriesPerRequest: null });

export type SummaryJob = { documentId: string; text: string; clientId: string };

export const summaryQueue = new Queue<SummaryJob>("summaries", {
  connection,
  defaultJobOptions: {
    attempts: 4,
    backoff: { type: "exponential", delay: 2_000 },
    removeOnComplete: { age: 3_600 },
    removeOnFail: { age: 86_400 },
  },
});
```

```typescript
// src/queue/worker.ts
import { Worker } from "bullmq";
import { connection, type SummaryJob } from "./queues.js";
import { openai } from "../llm/client.js";
import { config } from "../config.js";

export const worker = new Worker<SummaryJob, string>(
  "summaries",
  async (job) => {
    const res = await openai.chat.completions.create({
      model: config.OPENAI_MODEL,
      messages: [
        { role: "system", content: "Summarize the document in five bullet points." },
        { role: "user", content: job.data.text },
      ],
    });
    return res.choices[0]?.message.content ?? "";
  },
  { connection, concurrency: 5, limiter: { max: 30, duration: 60_000 } },
);

worker.on("failed", (job, err) => console.error({ jobId: job?.id, err: err.message }));
```

The `limiter` caps the worker at 30 jobs per minute, which protects your provider rate limit. Run workers as a separate process so a slow job cannot affect API latency.

```typescript
// src/routes/jobs.ts (excerpt)
app.post("/v1/summaries", async (req, reply) => {
  const body = SummaryBody.parse(req.body);
  const job = await summaryQueue.add("summarize", { ...body, clientId: req.clientId });
  return reply.code(202).send({ jobId: job.id }); // 202 Accepted
});

app.get("/v1/summaries/:id", async (req, reply) => {
  const job = await summaryQueue.getJob((req.params as { id: string }).id);
  if (!job || job.data.clientId !== req.clientId) return reply.code(404).send({ error: "not_found" });
  return reply.send({ state: await job.getState(), result: job.returnvalue ?? null });
});
```

Check `clientId` on the status endpoint. Without it, any caller who guesses a job ID can read another customer's summary. Clients can poll the status endpoint, or you can push completion through a webhook or SSE.

---

## Error Handling and Retries

Provider failures are routine. Plan for rate limits, timeouts, overloaded models, content filters, and malformed output.

| Failure | Typical Status | Retry? | Response to Client |
|---------|---------------|--------|--------------------|
| Rate limited by provider | 429 | Yes, with backoff | 503 with `Retry-After` |
| Provider outage or overload | 500, 502, 503 | Yes, bounded | 503 |
| Timeout | none | Once | 504 |
| Invalid request or too many tokens | 400 | No | 422 |
| Authentication to provider failed | 401 | No | 500, page the on-call engineer |
| Content refused | 200 with refusal | No | 422 with safe message |

Centralize the mapping in one error handler.

```typescript
// src/plugins/errors.ts
import OpenAI from "openai";
import type { FastifyInstance } from "fastify";
import { ZodError } from "zod";

export function registerErrorHandler(app: FastifyInstance): void {
  app.setErrorHandler((err, req, reply) => {
    if (err instanceof ZodError) {
      return reply.code(400).send({ error: "invalid_request", details: err.issues });
    }
    if (err instanceof OpenAI.RateLimitError) {
      reply.header("Retry-After", "5");
      return reply.code(503).send({ error: "busy" });
    }
    if (err instanceof OpenAI.APIConnectionTimeoutError) {
      return reply.code(504).send({ error: "upstream_timeout" });
    }
    if (err instanceof OpenAI.APIError) {
      req.log.error({ status: err.status, type: err.type }, "provider error");
      return reply.code(502).send({ error: "upstream_error" });
    }
    const status = (err as { statusCode?: number }).statusCode ?? 500;
    if (status >= 500) req.log.error({ err }, "unhandled error");
    return reply.code(status).send({ error: status >= 500 ? "internal" : err.message });
  });
}
```

Return generic messages to clients. Provider error bodies can contain prompt text or account details.

A circuit breaker adds protection when a provider is down. After a threshold of failures, it rejects requests immediately for a short period instead of letting them queue and time out. Libraries such as `opossum` implement this. Add one when you see cascading timeouts, not before.

For provider failover, a second provider as fallback improves availability but changes output style. Test prompts against both, as we discussed in **Chapter 4: Prompt Engineering**.

---

## Cost Control

LLM cost scales with tokens, and tokens scale with your traffic and your prompts. Control it at several layers.

1. **Cap input size.** Validate message length and count, as the `Body` schema does above.
2. **Cap output size.** Set `max_completion_tokens` on every request that does not need long output.
3. **Route by difficulty.** Send classification and extraction to a small model. Reserve larger models for reasoning tasks.
4. **Cache.** Cache identical requests in Redis, keyed by a hash of model, prompt, and parameters. Providers also offer prompt caching for long, repeated prefixes. Put stable content first in the prompt.
5. **Budget.** Enforce per-client daily token budgets, as shown earlier.
6. **Batch.** Use the provider's batch API for non-urgent bulk jobs. Providers typically discount batch pricing, so check current rates.

```typescript
// Exact-match response cache for deterministic requests
import { createHash } from "node:crypto";

export async function cached(
  redis: Redis, model: string, prompt: string, produce: () => Promise<string>,
): Promise<string> {
  const key = "llm:" + createHash("sha256").update(`${model}\n${prompt}`).digest("hex");
  const hit = await redis.get(key);
  if (hit) return hit;
  const value = await produce();
  await redis.set(key, value, "EX", 3_600);
  return value;
}
```

> 💰 **Cost Note**
>
> Exact-match caching only helps when users repeat identical prompts, which is common for classification and FAQ-style endpoints and rare for open chat. Measure your hit rate before investing in semantic caching.

Log token usage for every request with the client ID and endpoint. You cannot reduce a cost you cannot attribute.

---

## Testing AI Endpoints

Tests must not call real models. Real calls are slow, cost money, and return different text each run. Isolate the provider behind your own functions, then mock those functions.

Fastify's `inject` method runs requests without opening a port. Export a `buildApp()` function from `src/app.ts` so tests can create an instance.

```typescript
// tests/chat.test.ts
import { describe, it, expect, vi, beforeAll, afterAll } from "vitest";
import { buildApp } from "../src/app.js";

vi.mock("../src/llm/stream.js", () => ({
  async *streamCompletion() {
    yield { type: "token", text: "Hel" };
    yield { type: "token", text: "lo" };
    yield { type: "usage", totalTokens: 12 };
  },
}));
vi.mock("../src/budget.js", () => ({
  assertWithinBudget: vi.fn(),
  consumeBudget: vi.fn(),
}));

describe("POST /v1/chat/stream", () => {
  const app = buildApp();
  beforeAll(async () => { await app.ready(); });
  afterAll(async () => { await app.close(); });

  it("rejects requests without an API key", async () => {
    const res = await app.inject({ method: "POST", url: "/v1/chat/stream", payload: {} });
    expect(res.statusCode).toBe(401);
  });

  it("streams tokens and ends with [DONE]", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/v1/chat/stream",
      headers: { "x-api-key": "dev-key-1" },
      payload: { messages: [{ role: "user", content: "hi" }] },
    });
    expect(res.body).toContain('data: {"token":"Hel"}');
    expect(res.body).toContain("data: [DONE]");
  });
});
```

Test at three levels:

- **Unit tests** for pure logic: prompt builders, schema validation, budget math.
- **Contract tests** with mocked providers: status codes, error mapping, SSE framing.
- **Evaluation runs** against a real model on a fixed dataset, run on a schedule or before releases rather than on every commit. **Chapter 12: Deployment** covers prompt evaluation in CI.

> ⚠ **Common Mistake**
>
> Asserting on exact model text in automated tests. Assert on structure, status codes, and properties of the output instead. Exact text breaks whenever the provider updates a model.

---

## Observability

You need to answer these questions about any AI request: what did it cost, how long did it take, and why did it fail. Fastify logs JSON through `pino`, which any log platform can index.

```typescript
// src/app.ts (excerpt)
import Fastify, { type FastifyInstance } from "fastify";
import { randomUUID } from "node:crypto";

export function buildApp(): FastifyInstance {
  const app = Fastify({
    genReqId: () => randomUUID(),
    logger: {
      level: config.NODE_ENV === "production" ? "info" : "debug",
      redact: ["req.headers.authorization", 'req.headers["x-api-key"]'],
    },
  });
  // register plugins and routes here
  return app;
}
```

Log these fields per LLM call: request ID, client ID, model, input tokens, output tokens, latency, and outcome. Do not log full prompts or completions by default. They often contain personal data. If you need them for debugging, sample a small share, redact, and set a short retention period.

For tracing across the API, queue, and worker, use OpenTelemetry. Its Node.js SDK instruments HTTP, PostgreSQL, and Redis automatically. LLM-specific platforms such as Langfuse add prompt versions, token cost dashboards, and evaluation scores. Pick one early. Retrofitting traces is harder than adding them at the start.

Expose two health endpoints: `/health/live` returns `200` if the process runs, and `/health/ready` checks Redis and PostgreSQL. Never call the LLM provider in a health check. It costs money and couples your uptime to theirs.

---

## Deployment-Ready Patterns

### Graceful Shutdown

Streaming responses can run for a minute. On `SIGTERM`, stop accepting new connections and let open streams finish.

```typescript
// src/server.ts
import { buildApp } from "./app.js";
import { config } from "./config.js";
import { worker } from "./queue/worker.js"; // only import in the worker entrypoint in production

const app = buildApp();

async function shutdown(signal: string): Promise<void> {
  app.log.info({ signal }, "shutting down");
  const timer = setTimeout(() => process.exit(1), 25_000); // hard stop
  timer.unref();
  await app.close(); // waits for in-flight requests
  process.exit(0);
}

process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));

await app.listen({ port: config.PORT, host: "0.0.0.0" });
```

In practice, split `server.ts` and a separate `worker-main.ts` so the API and workers deploy and scale independently. Remove the worker import from the API entrypoint.

### Dockerfile

```dockerfile
FROM node:22-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production
COPY package*.json ./
RUN npm ci --omit=dev
COPY --from=build /app/dist ./dist
USER node
EXPOSE 3000
CMD ["node", "dist/server.js"]
```

The multi-stage build keeps compilers and dev dependencies out of the runtime image. Running as `node` instead of root limits the damage of a compromise. Deployment targets are covered in **Chapter 12: Deployment** and **Chapter 17: AWS Cloud + AI**.

---

## Production Notes

- **Keep the event loop free.** Parse large documents, run tokenizers on big inputs, and compute embeddings locally in worker threads or separate processes. Monitor event loop lag.
- **Set explicit timeouts everywhere.** LLM client, database, Redis, and the HTTP server. Defaults are rarely right for AI traffic.
- **Limit concurrency to the provider.** Upstream rate limits apply across all instances. Use a shared limiter or queue to stay under them.
- **Mind proxy timeouts.** Load balancers often close idle connections after 60 seconds. For long streams, send a comment line such as `: ping` every 15 seconds.
- **Version prompts.** Store prompts in files with version identifiers and log the version with each request.
- **Plan for provider changes.** Models are deprecated on a schedule. Keep model names in configuration, not code.

---

## Common Mistakes

1. **Calling the LLM provider from the client.** The API key ends up in the browser or app. Always proxy through your backend, as discussed in **Chapter 11: Flutter + AI**.
2. **Ignoring client disconnects.** Without an abort signal, you pay for completions nobody reads.
3. **Sending errors as HTTP statuses after streaming starts.** Headers are already sent. Use an SSE `error` event.
4. **Trusting structured output.** Schema-valid output can still be wrong or malicious. Validate and sanitize before using it.
5. **Rate limiting by request count only.** One large request can cost more than a hundred small ones. Add token budgets.
6. **Running workers inside the API process.** A heavy job degrades latency for every user.
7. **Logging full prompts.** Prompts contain personal data and secrets. Log metadata by default.
8. **Mocking nothing in tests.** Tests that call real providers are slow, costly, and flaky.

---

## Best Practices

1. Validate all input and environment variables with `zod` at the boundary.
2. Wrap provider SDKs in your own functions so tests and provider swaps touch one place.
3. Stream any response longer than a sentence, and cancel upstream work on disconnect.
4. Use structured output with a schema for every machine-consumed result.
5. Apply authentication, request rate limits, and token budgets together.
6. Move work longer than a few seconds to a queue with retries and a dead-letter strategy.
7. Return generic errors to clients and detailed errors to logs.
8. Track tokens and latency per request, per client, and per endpoint.
9. Shut down gracefully so deployments do not cut off streams.
10. Pin dependency majors and review provider changelogs on a schedule.

---

## Hands-On Exercises

### Exercise 1: Streaming Endpoint with Cancellation

**Difficulty:** ⭐⭐
**Time:** 30 minutes

Build `POST /v1/chat/stream` with Fastify and SSE.

1. Stream tokens as `data:` events and finish with `[DONE]`.
2. Abort the upstream request when the client disconnects.
3. Verify with `curl -N` that tokens arrive incrementally.
4. Verify cancellation: start a request, press Ctrl+C, and confirm in your logs that the abort fired.

**Expected outcome:** Tokens appear one at a time in the terminal, and the abort log line appears within a second of disconnecting.

### Exercise 2: Structured Ticket Classifier

**Difficulty:** ⭐⭐
**Time:** 25 minutes

Implement `POST /v1/tickets/classify`.

1. Define a `zod` schema with an enum category, enum priority, and a nullable email.
2. Derive the JSON Schema and use strict structured output.
3. Return `422` when the model refuses.
4. Test with five sample tickets, including one with no email.

**Expected outcome:** Every response matches the schema, and the no-email ticket returns `customerEmail: null`.

### Exercise 3: Token Budget

**Difficulty:** ⭐⭐
**Time:** 25 minutes

Add per-client daily token budgets backed by Redis.

1. Record usage from the final streaming chunk.
2. Reject requests with `429` once the budget is spent.
3. Set a budget of 500 tokens and confirm the third request is rejected.
4. Reset the budget by deleting the Redis key.

**Expected outcome:** Requests succeed until the budget is exhausted, then return `429`.

### Exercise 4: Mocked Provider Tests

**Difficulty:** ⭐⭐
**Time:** 25 minutes

Write Vitest tests for the chat endpoint without any network access.

1. Mock `streamCompletion` to emit fixed tokens.
2. Assert `401` without a key and a valid SSE body with one.
3. Add a test where the mock throws and assert that an `error` event is sent.
4. Run `npx vitest run` with your network disabled.

**Expected outcome:** All tests pass offline in under five seconds.

---

## Mini Project

**Title:** Document Q&A API
**Difficulty:** ⭐⭐⭐
**Time:** 4 hours
**What you'll build:** A production-shaped Node.js API that ingests documents in the background and answers questions with cited sources over a streaming endpoint.

### Requirements

1. **Ingestion endpoint:** `POST /v1/documents` accepts text, returns `202` and a job ID.
2. **Worker:** A BullMQ worker splits text into chunks, embeds them, and stores them in pgvector with a tenant ID.
3. **Ask endpoint:** `POST /v1/ask/stream` retrieves chunks, streams the answer over SSE, and sends a final `sources` event.
4. **Auth and limits:** API key authentication, per-minute rate limit, and daily token budget.
5. **Errors:** Provider failures map to correct statuses or SSE error events.
6. **Tests:** At least five Vitest tests with mocked providers.
7. **Operations:** Structured logs with token usage, health endpoints, graceful shutdown, and a Dockerfile.

### Deliverables

```
doc-qa-api/
├── src/
│   ├── routes/
│   ├── llm/
│   ├── rag/
│   ├── queue/
│   └── plugins/
├── tests/
├── docker-compose.yml     # api, worker, postgres (pgvector), redis
├── Dockerfile
├── .env.example
└── README.md
```

---

## Challenge Project

**Title:** Multi-Tenant AI Gateway
**Difficulty:** ⭐⭐⭐⭐
**Time:** 8 hours
**What you'll build:** A gateway service that sits between your applications and several LLM providers. It enforces per-tenant limits, routes by task, and reports cost.

### Requirements

1. **Unified endpoint:** One request format that routes to at least two providers.
2. **Routing rules:** Choose model by task type and input size, configured without redeploying.
3. **Failover:** Retry on the secondary provider when the primary fails, behind a circuit breaker.
4. **Tenant controls:** Per-tenant API keys, rate limits, token budgets, and allowed models.
5. **Caching:** Exact-match cache with hit-rate metrics.
6. **Cost reporting:** An endpoint that returns spend per tenant per day.
7. **Observability:** OpenTelemetry traces covering the gateway, provider call, and queue.
8. **Load test:** A script that shows behavior under 200 concurrent streams.

---

## Portfolio Milestone

After this chapter, you should have:

- A Node.js AI API with streaming, structured output, and RAG
- Authentication, rate limiting, and token budgets in place
- A background job pipeline with retries
- A test suite that runs without network access
- A containerized service you can deploy, which we do in **Chapter 17: AWS Cloud + AI**

> 🚀 **Portfolio Tip**
>
> Publish the repository with a README that shows a `curl -N` streaming demo, an architecture diagram, and a short section on cost controls. Reviewers look for evidence that you considered failure and spend, not only the happy path.

---

## Chapter Summary

- Node.js suits AI API layers because LLM calls are network-bound and its event loop handles many open connections.
- Create one SDK client with explicit timeouts and retries, and wrap it behind your own functions.
- Stream with SSE, send mid-stream errors as events, and abort upstream requests when clients disconnect.
- Structured output with a JSON Schema constrains shape. Validate the parsed result again in code.
- Combine authentication, request rate limits, and token budgets. Request counts alone do not control cost.
- A RAG endpoint is retrieval plus a grounded prompt. Filter by tenant inside the SQL query and treat retrieved text as untrusted.
- Queue long-running AI work with BullMQ and run workers in a separate process.
- Map provider errors to stable client responses and log details server-side.
- Test with mocked providers, and run model evaluations separately.
- Log tokens, latency, and cost per request, and shut down gracefully.

---

## Cheat Sheet

| Task | Tool or Pattern | Notes |
|------|----------------|-------|
| HTTP framework | Fastify 5 | Schema validation, `pino` logging |
| Environment validation | `zod` | Fail at startup |
| LLM client | `openai` | Set `maxRetries` and `timeout` |
| Streaming | SSE via `reply.hijack()` | Abort on `close` |
| Structured output | `response_format: json_schema` | Strict mode, validate again |
| Vector search | `pg` + `pgvector` | Parameterized queries, tenant filter |
| Background jobs | BullMQ + Redis | `maxRetriesPerRequest: null` |
| Rate limiting | `@fastify/rate-limit` + Redis | Key by client ID |
| Testing | Vitest + `app.inject` | Mock provider modules |
| Tracing | OpenTelemetry | Add early |

| Failure | Client Status |
|---------|--------------|
| Invalid input | 400 |
| Missing or wrong key | 401 |
| Budget exceeded | 429 |
| Provider rate limit | 503 + `Retry-After` |
| Provider timeout | 504 |
| Provider error | 502 |

---

## Glossary

| Term | Definition |
|------|-----------|
| SSE | Server-Sent Events. A one-way HTTP streaming protocol used to deliver tokens as they are generated. |
| Event loop | The Node.js mechanism that runs callbacks for I/O. Blocking it delays every request. |
| Structured output | LLM output constrained to a JSON Schema. |
| JSON Schema | A standard format for describing the shape of JSON data. |
| RAG | Retrieval-Augmented Generation. Supplying retrieved documents to an LLM so it answers from them. |
| pgvector | A PostgreSQL extension that stores embeddings and runs similarity search. |
| BullMQ | A Redis-backed job queue library for Node.js. |
| Backoff | Increasing the wait between retries, usually exponentially. |
| Circuit breaker | A pattern that rejects requests quickly after repeated failures, to give a dependency time to recover. |
| Token budget | A cap on the number of tokens a client may consume in a period. |
| Graceful shutdown | Finishing in-flight requests before a process exits. |
| OpenTelemetry | A vendor-neutral standard for traces, metrics, and logs. |

---

## References

- [Node.js Documentation](https://nodejs.org/docs/latest/api/) — Runtime API reference.
- [Fastify Documentation](https://fastify.dev/docs/latest/) — Routing, hooks, plugins, and testing with `inject`.
- [OpenAI Node.js SDK](https://github.com/openai/openai-node) — Official SDK, streaming and retry behavior.
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) — Supported JSON Schema subset.
- [Anthropic TypeScript SDK](https://github.com/anthropics/anthropic-sdk-typescript) — Official SDK for Claude.
- [Zod Documentation](https://zod.dev/) — Schema validation and JSON Schema conversion.
- [BullMQ Documentation](https://docs.bullmq.io/) — Queues, workers, retries, and rate limiting.
- [pgvector](https://github.com/pgvector/pgvector) — Vector similarity search for PostgreSQL.
- [Vitest](https://vitest.dev/) — Test runner and mocking.

## Further Reading

- [Vercel AI SDK](https://ai-sdk.dev/docs) — Multi-provider TypeScript toolkit for AI applications.
- [OpenTelemetry for JavaScript](https://opentelemetry.io/docs/languages/js/) — Tracing Node.js services.
- [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices) — Community guide to production Node.js.
- [MDN: Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) — Protocol details and browser API.

---

## Next Chapter Preview

In **Chapter 17: AWS Cloud + AI**, we move this kind of backend onto managed cloud services. You will call models through Amazon Bedrock, run endpoints on AWS Lambda, queue work with Amazon SQS, store vectors in a managed database, and define everything as code with the AWS CDK. The patterns from this chapter, such as streaming, budgets, and retries, carry over with different building blocks.
