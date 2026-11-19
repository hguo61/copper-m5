import { MODELS, type ModelId } from "@/lib/ai/models";

// US dollars per million tokens: Google's list prices as of 2026-10.
// Billing may differ slightly; these are for cost estimates.
export const PRICING: Record<ModelId, { input: number; output: number }> = {
  // As of 2026-11 (https://ai.google.dev/gemini-api/docs/pricing).
  "google/gemini-3.5-flash-lite": { input: 0.3, output: 2.5 },
  "google/gemini-2.5-flash": { input: 0.3, output: 2.5 },
  "google/gemini-3-flash": { input: 0.5, output: 3 },
};

export function costUsd(
  model: ModelId,
  inputTokens: number,
  outputTokens: number,
): number {
  const price = PRICING[model];
  return (inputTokens * price.input + outputTokens * price.output) / 1_000_000;
}

/**
 * The cost of one model call, split into input and output, for Langfuse.
 * `googleModel` is the name Google reports (for example "gemini-2.5-flash").
 * Returns null for a model that isn't approved or a call with no token counts.
 */
export function costDetails(
  googleModel: string,
  inputTokens: unknown,
  outputTokens: unknown,
): { input: number; output: number; total: number } | null {
  const model = MODELS.find(
    (m) => m.googleId === googleModel || m.id === googleModel,
  )?.id;
  if (!model) return null;
  if (typeof inputTokens !== "number" || typeof outputTokens !== "number")
    return null;
  const input = costUsd(model, inputTokens, 0);
  const output = costUsd(model, 0, outputTokens);
  return { input, output, total: costUsd(model, inputTokens, outputTokens) };
}
