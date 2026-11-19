import { LangfuseSpanProcessor } from "@langfuse/otel";
import { propagateAttributes, startActiveObservation } from "@langfuse/tracing";
import { LangfuseVercelAiSdkIntegration } from "@langfuse/vercel-ai-sdk";
import { trace } from "@opentelemetry/api";
import type {
  ReadableSpan,
  SpanExporter,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { registerTelemetry } from "ai";
import { costDetails } from "@/lib/ai/pricing";

// Module 5: observability with Langfuse. Every model call already goes
// through lib/ai/call.ts; that file calls into this one, so each call also
// becomes a Langfuse trace with its steps, tool calls, tokens, cost and
// latency. Without the Langfuse keys nothing here runs and nothing is sent.

export function isLangfuseConfigured(): boolean {
  return Boolean(
    process.env.LANGFUSE_PUBLIC_KEY?.trim() &&
    process.env.LANGFUSE_SECRET_KEY?.trim(),
  );
}

/**
 * Puts the app's own price for a model call on its span, so the cost in
 * Langfuse always matches lib/ai/pricing.ts (and the [ai_calls] log), even
 * for a model Langfuse has no price for.
 */
class CostSpanProcessor implements SpanProcessor {
  onStart() {}
  onEnd(span: ReadableSpan) {
    const a = span.attributes;
    if (a["gen_ai.operation.name"] !== "chat") return;
    const cost = costDetails(
      String(a["gen_ai.response.model"] ?? a["gen_ai.request.model"]),
      a["gen_ai.usage.input_tokens"],
      a["gen_ai.usage.output_tokens"],
    );
    if (cost) a["langfuse.observation.cost_details"] = JSON.stringify(cost);
  }
  async forceFlush() {}
  async shutdown() {}
}

// Kept on globalThis so a dev-server reload doesn't set tracing up twice.
const globals = globalThis as { copperLangfuse?: NodeTracerProvider };

/**
 * Starts sending traces to Langfuse, once. Returns false (and does nothing)
 * when the keys aren't set. Tests pass their own `exporter`.
 */
export function startLangfuse(exporter?: SpanExporter): boolean {
  if (globals.copperLangfuse) return true;
  if (!exporter && !isLangfuseConfigured()) return false;
  const provider = new NodeTracerProvider({
    spanProcessors: [
      new CostSpanProcessor(),
      new LangfuseSpanProcessor({
        exporter,
        // On Vercel a function can be frozen right after it answers, so
        // don't wait for a batch to fill up.
        exportMode: process.env.VERCEL ? "immediate" : "batched",
      }),
    ],
  });
  provider.register();
  registerTelemetry(new LangfuseVercelAiSdkIntegration());
  globals.copperLangfuse = provider;
  return true;
}

/** Sends any traces still waiting. Never throws: tracing must not break a reply. */
export async function flushLangfuse(): Promise<void> {
  try {
    await globals.copperLangfuse?.forceFlush();
  } catch {
    // Langfuse being unreachable is not the customer's problem.
  }
}

/** Stops tracing (tests use this to start clean). */
export async function stopLangfuse(): Promise<void> {
  const provider = globals.copperLangfuse;
  globals.copperLangfuse = undefined;
  await provider?.shutdown().catch(() => {});
  trace.disable();
}

/** True when this call is the start of a trace, not a step inside one. */
export function isTraceRoot(): boolean {
  return !trace.getActiveSpan();
}

/**
 * Labels the model call made inside `run` for Langfuse: the trace is named
 * after the feature (unless the call is part of a bigger trace), and tagged
 * with the feature and module so cost and latency can be filtered by either.
 */
export function labelled<T>(
  info: { module: number; feature: string },
  run: () => T,
): T {
  if (!startLangfuse()) return run();
  return propagateAttributes(
    {
      ...(isTraceRoot() && { traceName: info.feature }),
      tags: [info.feature, `module-${info.module}`],
    },
    run,
  );
}

/**
 * Groups every model call made inside `run` into one Langfuse trace, so a
 * turn that uses two agents shows up once, with its total cost and latency.
 * Call `setOutput` with the final answer.
 */
export async function traced<T>(
  name: string,
  input: unknown,
  run: (setOutput: (output: unknown) => void) => Promise<T>,
): Promise<T> {
  if (!startLangfuse()) return run(() => {});
  try {
    return await startActiveObservation(name, (span) => {
      span.update({ input });
      return propagateAttributes({ traceName: name }, () =>
        run((output) => span.update({ output })),
      );
    });
  } finally {
    await flushLangfuse();
  }
}
