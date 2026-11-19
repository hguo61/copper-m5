// @vitest-environment node
import { convertArrayToReadableStream, MockLanguageModelV4 } from "ai/test";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { callStream, callText } from "@/lib/ai/call";
import {
  flushLangfuse,
  isLangfuseConfigured,
  startLangfuse,
  stopLangfuse,
  traced,
} from "@/lib/ai/langfuse";
import { costDetails, costUsd } from "@/lib/ai/pricing";

// Google hands back whatever mock model the test sets.
let model: MockLanguageModelV4;
vi.mock("@ai-sdk/google", () => ({
  createGoogleGenerativeAI: () => () => model,
}));

const usage = (input: number, output: number) => ({
  inputTokens: {
    total: input,
    noCache: input,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: { total: output, text: output, reasoning: undefined },
});

function replyWith(text: string, input = 1000, output = 200) {
  return new MockLanguageModelV4({
    modelId: "gemini-3.5-flash-lite",
    doGenerate: async () => ({
      content: [{ type: "text", text }],
      finishReason: { unified: "stop", raw: undefined },
      usage: usage(input, output),
      warnings: [],
    }),
    doStream: async () => ({
      stream: convertArrayToReadableStream([
        { type: "text-start", id: "t" },
        { type: "text-delta", id: "t", delta: text },
        { type: "text-end", id: "t" },
        {
          type: "finish",
          finishReason: { unified: "stop", raw: undefined },
          usage: usage(input, output),
        },
      ]),
    }),
  });
}

/** The spans that would have been sent to Langfuse. */
async function sentSpans(exporter: InMemorySpanExporter) {
  await flushLangfuse();
  return exporter.getFinishedSpans();
}

beforeEach(() => {
  vi.stubEnv("GOOGLE_GENERATIVE_AI_API_KEY", "AIza-test-key");
  vi.spyOn(console, "info").mockImplementation(() => {});
});
afterEach(async () => {
  await stopLangfuse();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("costDetails", () => {
  it("splits a call's cost into input and output, matching costUsd", () => {
    const cost = costDetails("gemini-3.5-flash-lite", 1000, 200);
    expect(cost).toEqual({
      input: costUsd("google/gemini-3.5-flash-lite", 1000, 0),
      output: costUsd("google/gemini-3.5-flash-lite", 0, 200),
      total: costUsd("google/gemini-3.5-flash-lite", 1000, 200),
    });
  });

  it("knows a model by Google's own name for it", () => {
    expect(costDetails("gemini-3-flash-preview", 1_000_000, 0)?.total).toBe(
      0.5,
    );
  });

  it("gives no cost for an unknown model or missing token counts", () => {
    expect(costDetails("some-other-model", 10, 10)).toBeNull();
    expect(costDetails("gemini-3.5-flash-lite", undefined, 10)).toBeNull();
  });
});

describe("Langfuse tracing", () => {
  it("is off without the Langfuse keys: calls still work and nothing starts", async () => {
    vi.stubEnv("LANGFUSE_PUBLIC_KEY", "");
    vi.stubEnv("LANGFUSE_SECRET_KEY", "");
    model = replyWith("hello");
    expect(isLangfuseConfigured()).toBe(false);
    expect(startLangfuse()).toBe(false);
    const { text } = await callText(
      { module: 5, feature: "untraced" },
      { prompt: "hi" },
    );
    expect(text).toBe("hello");
  });

  it("needs both keys", () => {
    vi.stubEnv("LANGFUSE_PUBLIC_KEY", "pk-lf-test");
    vi.stubEnv("LANGFUSE_SECRET_KEY", "");
    expect(isLangfuseConfigured()).toBe(false);
    vi.stubEnv("LANGFUSE_SECRET_KEY", "sk-lf-test");
    expect(isLangfuseConfigured()).toBe(true);
  });

  it("traces a call with its tokens, the app's own cost, and feature and module tags", async () => {
    const exporter = new InMemorySpanExporter();
    startLangfuse(exporter);
    model = replyWith("hello", 1000, 200);
    await callText({ module: 5, feature: "judge" }, { prompt: "hi" });

    const spans = await sentSpans(exporter);
    const chat = spans.find(
      (s) => s.attributes["gen_ai.operation.name"] === "chat",
    )!;
    expect(chat.attributes["gen_ai.usage.input_tokens"]).toBe(1000);
    expect(chat.attributes["gen_ai.usage.output_tokens"]).toBe(200);
    expect(
      JSON.parse(chat.attributes["langfuse.observation.cost_details"] as string)
        .total,
    ).toBe(costUsd("google/gemini-3.5-flash-lite", 1000, 200));
    expect(chat.attributes["langfuse.trace.name"]).toBe("judge");
    expect(chat.attributes["langfuse.trace.tags"]).toEqual([
      "judge",
      "module-5",
    ]);
    // Latency: every span has a start and an end.
    expect(chat.endTime).not.toEqual([0, 0]);
  });

  it("puts two agents' calls in one trace for the turn", async () => {
    const exporter = new InMemorySpanExporter();
    startLangfuse(exporter);
    model = replyWith("hello");
    await traced(
      "ask-copper-turn",
      "Can I get a refund?",
      async (setOutput) => {
        for (const feature of ["ask-copper", "refund-desk"]) {
          const result = callStream({ module: 4, feature }, { prompt: "hi" });
          setOutput(await result.text);
        }
      },
    );

    const spans = await sentSpans(exporter);
    const traceIds = new Set(spans.map((s) => s.spanContext().traceId));
    expect(traceIds.size).toBe(1);
    const turn = spans.find((s) => s.name === "ask-copper-turn")!;
    expect(turn.attributes["langfuse.observation.input"]).toContain("refund");
    expect(turn.attributes["langfuse.observation.output"]).toContain("hello");
    const agents = spans
      .map((s) => s.attributes["gen_ai.agent.name"])
      .filter(Boolean);
    expect(agents).toEqual(
      expect.arrayContaining(["ask-copper", "refund-desk"]),
    );
    // The turn keeps its own name, not the name of the last agent.
    for (const span of spans)
      expect(span.attributes["langfuse.trace.name"]).toBe("ask-copper-turn");
  });
});
