import { describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import {
	collapseBuiltVariants,
	collapseVariants,
	deriveThinkingPairFamilies,
	getVariantAliasSources,
	isCollapsedVariantSpec,
	resolveBareVariantSelector,
	resolveVariantSelector,
	reviewedCollapseTable,
	type VariantCollapseTable,
} from "@oh-my-pi/pi-catalog/compat/collapse";
import { stripThinkingVariantSuffix } from "@oh-my-pi/pi-catalog/compat/taxonomy";
import {
	ANTIGRAVITY_PRIMARY_ENDPOINT,
	fetchAntigravityDiscoveryModels,
} from "@oh-my-pi/pi-catalog/discovery/antigravity";
import { Effort } from "@oh-my-pi/pi-catalog/effort";
import { resolveProviderModels } from "@oh-my-pi/pi-catalog/model-manager";
import {
	defaultSupportedEffort,
	mapEffortToGoogleThinkingLevel,
	resolveWireModelId,
} from "@oh-my-pi/pi-catalog/model-thinking";
import { getBundledModel, getBundledModels } from "@oh-my-pi/pi-catalog/models";
import {
	googleAntigravityModelManagerOptions,
	googleGeminiCliModelManagerOptions,
} from "@oh-my-pi/pi-catalog/provider-models/google";
import type { ModelSpec } from "@oh-my-pi/pi-catalog/types";

function requireReviewedTable(provider: string): VariantCollapseTable {
	const table = reviewedCollapseTable(provider);
	if (table === undefined) throw new Error(`missing reviewed collapse table for ${provider}`);
	return table;
}

const antigravityTable = requireReviewedTable("google-antigravity");
const cursorTable = requireReviewedTable("cursor");
const devinTable = requireReviewedTable("devin");
const geminiCliTable = requireReviewedTable("google-gemini-cli");

function memberSpec(
	id: string,
	overrides: Partial<ModelSpec<"google-gemini-cli">> = {},
): ModelSpec<"google-gemini-cli"> {
	return {
		id,
		name: id,
		api: "google-gemini-cli",
		provider: "google-antigravity",
		baseUrl: "https://daily-cloudcode-pa.googleapis.com",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_048_576,
		maxTokens: 65_535,
		...overrides,
	};
}

function pairSpec(
	id: string,
	overrides: Partial<ModelSpec<"openai-completions">> = {},
): ModelSpec<"openai-completions"> {
	return {
		id,
		name: id,
		api: "openai-completions",
		provider: "venice",
		baseUrl: "https://api.venice.ai/api/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 8_192,
		...overrides,
	};
}

function cursorMemberSpec(id: string, overrides: Partial<ModelSpec<"cursor-agent">> = {}): ModelSpec<"cursor-agent"> {
	return {
		id,
		name: id,
		api: "cursor-agent",
		provider: "cursor",
		baseUrl: "https://api2.cursor.sh",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 64_000,
		...overrides,
	};
}

function devinMemberSpec(id: string, overrides: Partial<ModelSpec<"devin-agent">> = {}): ModelSpec<"devin-agent"> {
	return {
		id,
		name: id,
		api: "devin-agent",
		provider: "devin",
		baseUrl: "https://server.codeium.com",
		reasoning: true,
		input: ["text"],
		supportsTools: true,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 64_000,
		...overrides,
	};
}
const PAIR_THINKING = {
	mode: "budget",
	efforts: [Effort.Minimal, Effort.Low, Effort.Medium, Effort.High],
} as const;

const FLASH_TIERSET = () => [
	memberSpec("gemini-3.8-flash-low", { maxTokens: 32_000 }),
	memberSpec("gemini-3.8-flash-medium"),
	memberSpec("gemini-3.8-flash-high"),
	memberSpec("gemini-3.8-flash-tiered"),
];

const FLASH_TRIPLET = () => [
	memberSpec("gemini-3.5-flash-extra-low", { maxTokens: 32_000 }),
	memberSpec("gemini-3.5-flash-low"),
	memberSpec("gemini-3-flash-agent", { input: ["text"] }),
];

describe("collapseVariants with a reviewed table", () => {
	it("collapses the Sonnet 5.5 tier SKUs into one routed logical spec", () => {
		const out = collapseVariants(
			[
				memberSpec("claude-sonnet-5-5-high"),
				memberSpec("claude-sonnet-5-5-low", { maxTokens: 64_000 }),
				memberSpec("claude-sonnet-5-5-medium"),
				memberSpec("gemini-2.5-flash-lite"),
			],
			{ table: antigravityTable },
		);

		expect(out.map(m => m.id).sort()).toEqual(["claude-sonnet-5-5", "gemini-2.5-flash-lite"]);
		// Non-family specs pass through by reference.
		expect(out.find(m => m.id === "gemini-2.5-flash-lite")?.thinking).toBeUndefined();
		const sonnet = out.find(m => m.id === "claude-sonnet-5-5");
		expect(sonnet?.name).toBe("Claude Sonnet 5.5");
		// First live member in priority order becomes the default wire id.
		expect(sonnet?.requestModelId).toBe("claude-sonnet-5-5-low");
		// Capability union: max caps, image support from any member.
		expect(sonnet?.maxTokens).toBe(65_535);
		expect(sonnet?.input).toEqual(["text", "image"]);
		// Every 5.5 SKU is a thinking tier: mandatory ladder, no off route.
		expect(sonnet?.thinking?.mode).toBe("budget");
		expect(sonnet?.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High]);
		expect(sonnet?.thinking?.requiresEffort).toBe(true);
		expect(sonnet?.thinking?.effortRouting).toEqual({
			low: "claude-sonnet-5-5-low",
			medium: "claude-sonnet-5-5-medium",
			high: "claude-sonnet-5-5-high",
		});
	});

	it("collapses the Opus 5.5 tier SKUs into one routed logical spec", () => {
		const out = collapseVariants(
			[memberSpec("claude-opus-5-5-high"), memberSpec("claude-opus-5-5-low"), memberSpec("claude-opus-5-5-medium")],
			{ table: antigravityTable },
		);

		expect(out).toHaveLength(1);
		const opus = out[0];
		expect(opus?.id).toBe("claude-opus-5-5");
		expect(opus?.name).toBe("Claude Opus 5.5");
		expect(opus?.requestModelId).toBe("claude-opus-5-5-low");
		expect(opus?.thinking).toEqual({
			mode: "budget",
			efforts: [Effort.Low, Effort.Medium, Effort.High],
			requiresEffort: true,
			effortRouting: {
				low: "claude-opus-5-5-low",
				medium: "claude-opus-5-5-medium",
				high: "claude-opus-5-5-high",
			},
		});
		const model = buildModel(opus as ModelSpec<"google-gemini-cli">);
		expect(resolveWireModelId(model, Effort.Low)).toBe("claude-opus-5-5-low");
		expect(resolveWireModelId(model, Effort.Medium)).toBe("claude-opus-5-5-medium");
		expect(resolveWireModelId(model, Effort.High)).toBe("claude-opus-5-5-high");
	});

	it("folds the gemini-3.8-flash-tiered alias into gemini-3.8-flash so :minimal sends LOW not MINIMAL (#10016)", () => {
		const out = collapseVariants(FLASH_TIERSET(), { table: antigravityTable });

		expect(out).toHaveLength(1);
		const flash = out[0];
		expect(flash?.id).toBe("gemini-3.8-flash");
		expect(flash?.thinking?.effortRouting).toEqual({
			minimal: "gemini-3.8-flash-low",
			low: "gemini-3.8-flash-low",
			medium: "gemini-3.8-flash-medium",
			high: "gemini-3.8-flash-high",
		});

		// The actual regression: Cloud Code Assist rejects thinkingLevel MINIMAL on
		// this SKU with HTTP 400, so the collapsed routing must downgrade both :off
		// (floored to minimal by requiresEffort) and :minimal to LOW on the wire.
		const model = buildModel(flash as ModelSpec<"google-gemini-cli">);
		expect(mapEffortToGoogleThinkingLevel(Effort.Minimal, model)).toBe("LOW");
		expect(mapEffortToGoogleThinkingLevel(Effort.Low, model)).toBe("LOW");
		expect(resolveWireModelId(model, Effort.Minimal)).toBe("gemini-3.8-flash-low");
	});

	it("dedupes a stale standalone gemini-3.8-flash-tiered snapshot into the collapsed gemini-3.8-flash (#10016)", () => {
		// Mirrors the bundled catalog snapshot: the already-collapsed logical id
		// plus a raw -tiered alias that older snapshots emitted as its own
		// MINIMAL-sending model. Runtime collapse must fold the alias away.
		const collapsedFlash = memberSpec("gemini-3.8-flash", {
			requestModelId: "gemini-3.8-flash-low",
			thinking: {
				mode: "google-level",
				efforts: [Effort.Minimal, Effort.Low, Effort.Medium, Effort.High],
				requiresEffort: true,
				effortRouting: {
					minimal: "gemini-3.8-flash-low",
					low: "gemini-3.8-flash-low",
					medium: "gemini-3.8-flash-medium",
					high: "gemini-3.8-flash-high",
				},
			},
		});
		const staleTiered = memberSpec("gemini-3.8-flash-tiered", {
			thinking: {
				mode: "google-level",
				efforts: [Effort.Minimal, Effort.Low, Effort.Medium, Effort.High],
				requiresEffort: true,
			},
		});

		const out = collapseVariants([collapsedFlash, staleTiered], { table: antigravityTable });
		expect(out.map(m => m.id)).toEqual(["gemini-3.8-flash"]);
		expect(out[0]?.thinking?.effortRouting?.minimal).toBe("gemini-3.8-flash-low");
	});

	it("bundles only the routed Gemini flash and Claude 5.5 logical models after generation", () => {
		const ids = getBundledModels("google-antigravity").map(model => model.id);
		expect(ids).not.toContain("gemini-3.5-flash");
		expect(ids).not.toContain("gemini-3.6-flash");
		expect(ids).not.toContain("gemini-3.7-flash");
		expect(ids).not.toContain("claude-sonnet-4-6");
		expect(ids).not.toContain("claude-opus-4-6");
		expect(ids.some(id => id.startsWith("gemini-3.8-flash-"))).toBe(false);
		expect(ids.some(id => id.startsWith("claude-sonnet-5-5-") || id.startsWith("claude-opus-5-5-"))).toBe(false);

		const flash = getBundledModel("google-antigravity", "gemini-3.8-flash");
		expect(flash?.thinking?.effortRouting?.minimal).toBe("gemini-3.8-flash-low");
		expect(flash ? mapEffortToGoogleThinkingLevel(Effort.Minimal, flash) : undefined).toBe("LOW");

		const sonnet = getBundledModel("google-antigravity", "claude-sonnet-5-5");
		expect(sonnet?.thinking?.requiresEffort).toBe(true);
		expect(sonnet ? resolveWireModelId(sonnet, Effort.High) : undefined).toBe("claude-sonnet-5-5-high");
		const opus = getBundledModel("google-antigravity", "claude-opus-5-5");
		expect(opus ? resolveWireModelId(opus, Effort.Medium) : undefined).toBe("claude-opus-5-5-medium");
	});

	it("instantiates the gemini-{rev}-flash template for a generation no reviewed entry names", () => {
		// Discovery advertises a brand-new Flash revision: the template must
		// collapse it like 3.8 without a per-revision KDL entry, while retired
		// flash lanes (outside `revision=">=3.8"`) stay expanded.
		const out = collapseVariants(
			[
				memberSpec("gemini-3.9-flash-high"),
				memberSpec("gemini-3.9-flash-low"),
				memberSpec("gemini-3.9-flash-medium"),
				memberSpec("gemini-3.9-flash-tiered"),
				memberSpec("gemini-2.9-flash-low"),
				memberSpec("gemini-2.9-flash-high"),
				memberSpec("gemini-3.7-flash-low"),
			],
			{ table: antigravityTable },
		);

		expect(out.map(m => m.id)).toEqual([
			"gemini-3.9-flash",
			"gemini-2.9-flash-low",
			"gemini-2.9-flash-high",
			"gemini-3.7-flash-low",
		]);
		const flash = out[0];
		expect(flash?.name).toBe("Gemini 3.9 Flash");
		expect(flash?.requestModelId).toBe("gemini-3.9-flash-low");
		expect(flash?.thinking?.effortRouting).toEqual({
			minimal: "gemini-3.9-flash-low",
			low: "gemini-3.9-flash-low",
			medium: "gemini-3.9-flash-medium",
			high: "gemini-3.9-flash-high",
		});

		const model = buildModel(flash as ModelSpec<"google-gemini-cli">);
		expect(model.thinking?.mode).toBe("google-level");
		expect(resolveWireModelId(model, Effort.Minimal)).toBe("gemini-3.9-flash-low");
		expect(resolveVariantSelector("google-antigravity", "gemini-3.9-flash-tiered")).toBe("gemini-3.9-flash");
		expect(resolveVariantSelector("google-antigravity", "gemini-2.9-flash-low")).toBeUndefined();
	});

	it("drops routes whose target member is absent", () => {
		const out = collapseVariants([memberSpec("gemini-3.8-flash-low")], { table: antigravityTable });

		expect(out).toHaveLength(1);
		expect(out[0]?.id).toBe("gemini-3.8-flash");
		expect(out[0]?.requestModelId).toBe("gemini-3.8-flash-low");
		// minimal+low route to -low (present); medium (-medium) and high (-high)
		// targets are absent and drop.
		expect(out[0]?.thinking?.effortRouting).toEqual({
			minimal: "gemini-3.8-flash-low",
			low: "gemini-3.8-flash-low",
		});
	});

	it("routes off to the bare wire id and efforts to the -thinking twin on the 2.5 pair", () => {
		const out = collapseVariants(
			[
				memberSpec("gemini-2.5-flash", { maxTokens: 64_000 }),
				memberSpec("gemini-2.5-flash-thinking", { maxTokens: 64_000 }),
			],
			{ table: antigravityTable },
		);

		expect(out).toHaveLength(1);
		const spec = out[0];
		expect(spec?.id).toBe("gemini-2.5-flash");
		// Default wire id equals the logical id — requestModelId is omitted.
		expect(spec?.requestModelId).toBeUndefined();
		expect(spec?.thinking?.mode).toBe("budget");
		expect(spec?.thinking?.effortRouting).toEqual({
			off: "gemini-2.5-flash",
			minimal: "gemini-2.5-flash-thinking",
			low: "gemini-2.5-flash-thinking",
			medium: "gemini-2.5-flash-thinking",
			high: "gemini-2.5-flash-thinking",
		});

		const model = buildModel(spec as ModelSpec<"google-gemini-cli">);
		expect(resolveWireModelId(model, undefined)).toBe("gemini-2.5-flash");
		expect(resolveWireModelId(model, Effort.High)).toBe("gemini-2.5-flash-thinking");
	});

	it("preserves -thinking effort routes on a bare-only 2.5 discovery", () => {
		// `preserve-absent-effort-routes`: upstream accepts the `-thinking` wire
		// id even when the model-list endpoint advertises only the bare id.
		const out = collapseVariants([memberSpec("gemini-2.5-flash", { maxTokens: 64_000 })], {
			table: antigravityTable,
		});

		expect(out).toHaveLength(1);
		const spec = out[0];
		expect(spec?.id).toBe("gemini-2.5-flash");
		// Preserved-absent routes pin the bare wire id explicitly even though it
		// equals the logical id.
		expect(spec?.requestModelId).toBe("gemini-2.5-flash");
		expect(spec?.thinking?.effortRouting?.[Effort.High]).toBe("gemini-2.5-flash-thinking");

		const model = buildModel(spec as ModelSpec<"google-gemini-cli">);
		expect(resolveWireModelId(model, Effort.High)).toBe("gemini-2.5-flash-thinking");
		expect(resolveWireModelId(model, undefined)).toBe("gemini-2.5-flash");
	});

	it("drops absent 5.5 tier routes and falls back through requestModelId", () => {
		const out = collapseVariants([memberSpec("claude-opus-5-5-high")], { table: antigravityTable });

		expect(out).toHaveLength(1);
		const spec = out[0];
		expect(spec?.id).toBe("claude-opus-5-5");
		expect(spec?.requestModelId).toBe("claude-opus-5-5-high");
		expect(spec?.thinking?.effortRouting).toEqual({ high: "claude-opus-5-5-high" });

		const model = buildModel(spec as ModelSpec<"google-gemini-cli">);
		// Thinking-off and efforts without a live member fall back through
		// requestModelId to the only wire id the backend actually serves.
		expect(resolveWireModelId(model, undefined)).toBe("claude-opus-5-5-high");
		expect(resolveWireModelId(model, Effort.Medium)).toBe("claude-opus-5-5-high");
		expect(resolveWireModelId(model, Effort.High)).toBe("claude-opus-5-5-high");
	});

	it("reconciles a stale 3.1-pro snapshot whose routing still targets the retired -high wire id", () => {
		// Snapshots written while `gemini-3.1-pro-high` was live keep routes and
		// the default wire id pointing at it. The `retiredMembers` entry triggers
		// `reconcileRetiredRouting`, which re-points retired targets to the
		// family's live wire ids.
		const stale: ModelSpec<"google-gemini-cli"> = {
			...memberSpec("gemini-3.1-pro"),
			requestModelId: "gemini-3.1-pro-high",
			thinking: {
				mode: "budget",
				efforts: [Effort.Low, Effort.High],
				effortRouting: {
					off: "gemini-3.1-pro-low",
					[Effort.Low]: "gemini-3.1-pro-low",
					[Effort.High]: "gemini-3.1-pro-high",
				},
			},
		};
		const out = collapseVariants([stale], { table: antigravityTable });

		expect(out).toHaveLength(1);
		const spec = out[0];
		expect(spec?.id).toBe("gemini-3.1-pro");
		expect(spec?.requestModelId).toBe("gemini-3.1-pro-low");
		expect(spec?.thinking?.effortRouting).toEqual({
			off: "gemini-3.1-pro-low",
			low: "gemini-3.1-pro-low",
			high: "gemini-pro-agent",
		});

		const model = buildModel(spec as ModelSpec<"google-gemini-cli">);
		expect(resolveWireModelId(model, Effort.High)).toBe("gemini-pro-agent");
	});

	it("renames single-member families through requestModelId with no routing", () => {
		const out = collapseVariants([memberSpec("gpt-oss-120b-medium", { input: ["text"] })], {
			table: antigravityTable,
		});

		expect(out[0]?.id).toBe("gpt-oss-120b");
		expect(out[0]?.requestModelId).toBe("gpt-oss-120b-medium");
		expect(out[0]?.thinking?.effortRouting).toBeUndefined();
		expect(out[0]?.thinking?.mode).toBe("budget");
	});

	it("is idempotent and dedupes mixed raw+collapsed input", () => {
		const once = collapseVariants(FLASH_TIERSET(), { table: antigravityTable });
		expect(collapseVariants(once, { table: antigravityTable })).toEqual(once);

		// Stale raw members beside the live collapsed entry dedupe away; the
		// collapsed entry wins verbatim.
		const mixed = [...once, memberSpec("gemini-3.8-flash-medium"), memberSpec("gemini-3.8-flash-tiered")];
		const deduped = collapseVariants(mixed, { table: antigravityTable });
		expect(deduped).toEqual(once);
	});

	it("keeps gemini-cli flash on the level transport with the original routing", () => {
		const out = collapseVariants(FLASH_TRIPLET(), { table: geminiCliTable });
		const flash = out.find(m => m.id === "gemini-3.5-flash");
		expect(flash?.thinking?.mode).toBe("google-level");
		expect(flash?.thinking?.effortBudgets).toBeUndefined();
		expect(flash?.thinking?.effortRouting).toEqual({
			off: "gemini-3.5-flash-extra-low",
			minimal: "gemini-3-flash-agent",
			low: "gemini-3.5-flash-extra-low",
			medium: "gemini-3.5-flash-extra-low",
			high: "gemini-3.5-flash-low",
		});
	});

	it("collapses the 3.1-pro family on the budget transport with the +1 budgets", () => {
		const out = collapseVariants([memberSpec("gemini-3.1-pro-low"), memberSpec("gemini-pro-agent")], {
			table: antigravityTable,
		});
		const pro = out.find(m => m.id === "gemini-3.1-pro");
		expect(pro?.thinking?.mode).toBe("budget");
		expect(pro?.thinking?.effortBudgets).toEqual({ low: 1001, high: 10001 });
		expect(pro?.thinking?.effortRouting).toEqual({
			off: "gemini-3.1-pro-low",
			low: "gemini-3.1-pro-low",
			high: "gemini-pro-agent",
		});
	});

	it("refreshes a stale family.id-keyed 3.1-pro snapshot in place to the budget contract", () => {
		// Pass-through branch: a bundled collapsed `gemini-3.1-pro` on the old level
		// transport with no live members refreshes from the hand table.
		const stale: ModelSpec<"google-gemini-cli"> = {
			...memberSpec("gemini-3.1-pro"),
			reasoning: true,
			requestModelId: "gemini-3.1-pro-low",
			thinking: {
				mode: "google-level",
				efforts: [Effort.Low, Effort.High],
				effortRouting: { off: "gemini-3.1-pro-low", low: "gemini-3.1-pro-low", high: "gemini-pro-agent" },
				suppressWhenOff: true,
			},
		};
		const out = collapseVariants([stale], { table: antigravityTable });
		const pro = out.find(m => m.id === "gemini-3.1-pro");
		expect(pro?.thinking?.mode).toBe("budget");
		expect(pro?.thinking?.effortBudgets).toEqual({ low: 1001, high: 10001 });
	});
});

describe("stripThinkingVariantSuffix", () => {
	it("strips trailing and infix tokens case-insensitively", () => {
		expect(stripThinkingVariantSuffix("kimi-k2-thinking")).toBe("kimi-k2");
		expect(stripThinkingVariantSuffix("hf:moonshotai/Kimi-K2-Thinking")).toBe("hf:moonshotai/Kimi-K2");
		expect(stripThinkingVariantSuffix("xiaomi/mimo-v2-flash-thinking-original")).toBe(
			"xiaomi/mimo-v2-flash-original",
		);
		expect(stripThinkingVariantSuffix("[Kiro] claude-opus-4-8-thinking [X]")).toBe("[Kiro] claude-opus-4-8 [X]");
		// Infix `reasoning` pairs live siblings (perplexity/sonar-reasoning-pro
		// beside sonar-pro) and dated `-thinking-2507` forms keep pairing.
		expect(stripThinkingVariantSuffix("perplexity/sonar-reasoning-pro")).toBe("perplexity/sonar-pro");
		expect(stripThinkingVariantSuffix("qwen3-235b-a22b-thinking-2507")).toBe("qwen3-235b-a22b-2507");
	});

	it("ignores ids without a token and negated tokens", () => {
		expect(stripThinkingVariantSuffix("kimi-k2")).toBeUndefined();
		// OpenRouter route variants use `:thinking` — a different mechanism.
		expect(stripThinkingVariantSuffix("anthropic/claude-3.7-sonnet:thinking")).toBeUndefined();
		expect(stripThinkingVariantSuffix("thinkingcap-1")).toBeUndefined();
		// `non-thinking` names the NON-thinking SKU.
		expect(stripThinkingVariantSuffix("deepseek-non-thinking-v3.2-exp")).toBeUndefined();
	});
});

describe("deriveThinkingPairFamilies", () => {
	it("derives a pair family routing off to bare and efforts to -thinking", () => {
		const specs = [pairSpec("kimi-k2"), pairSpec("kimi-k2-thinking", { reasoning: true, thinking: PAIR_THINKING })];
		const families = deriveThinkingPairFamilies(specs);

		expect(families).toHaveLength(1);
		expect(families[0]?.id).toBe("kimi-k2");
		expect(families[0]?.members).toEqual(["kimi-k2", "kimi-k2-thinking"]);
		expect(families[0]?.routing).toEqual({
			off: "kimi-k2",
			minimal: "kimi-k2-thinking",
			low: "kimi-k2-thinking",
			medium: "kimi-k2-thinking",
			high: "kimi-k2-thinking",
		});

		const out = collapseVariants(specs, { table: { families } });
		expect(out).toHaveLength(1);
		expect(out[0]?.id).toBe("kimi-k2");
		expect(out[0]?.requestModelId).toBeUndefined();
		expect(out[0]?.reasoning).toBe(true);
		expect(out[0]?.thinking?.effortRouting?.[Effort.High]).toBe("kimi-k2-thinking");
		expect(out[0]?.thinking?.effortRouting?.off).toBe("kimi-k2");
	});

	it("pairs infix -thinking tokens", () => {
		const specs = [
			pairSpec("xiaomi/mimo-v2-flash-original"),
			pairSpec("xiaomi/mimo-v2-flash-thinking-original", { reasoning: true, thinking: PAIR_THINKING }),
		];
		const families = deriveThinkingPairFamilies(specs);
		expect(families).toHaveLength(1);
		expect(families[0]?.id).toBe("xiaomi/mimo-v2-flash-original");
	});

	it("collapses metadata-poor twins using the bare member's surface", () => {
		// Aggregators routinely ship the twin with reasoning:false, no
		// thinking config, and zero (unknown) pricing — name wins.
		const base = pairSpec("xiaomi/mimo-v2-flash", {
			reasoning: true,
			cost: { input: 0.09, output: 0.29, cacheRead: 0.045, cacheWrite: 0 },
			thinking: { mode: "effort", efforts: [Effort.Minimal, Effort.Low, Effort.Medium, Effort.High, Effort.XHigh] },
		});
		const twin = pairSpec("xiaomi/mimo-v2-flash-thinking", {
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		});
		const families = deriveThinkingPairFamilies([base, twin]);
		expect(families).toHaveLength(1);
		expect(families[0]?.thinking?.mode).toBe("effort");

		const out = collapseVariants([base, twin], { table: { families } });
		expect(out).toHaveLength(1);
		expect(out[0]?.reasoning).toBe(true);
		expect(out[0]?.cost.input).toBe(0.09);
		expect(out[0]?.thinking?.effortRouting?.[Effort.XHigh]).toBe("xiaomi/mimo-v2-flash-thinking");
		expect(out[0]?.thinking?.effortRouting?.off).toBe("xiaomi/mimo-v2-flash");
	});

	it("collapses zero-cost metadata-less twins with a derived surface", () => {
		const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
		const base = pairSpec("TEE/kimi-k2.5", { cost: zeroCost });
		const twin = pairSpec("TEE/kimi-k2.5-thinking", { cost: zeroCost });
		const families = deriveThinkingPairFamilies([base, twin]);
		expect(families).toHaveLength(1);
		expect(families[0]?.thinking?.efforts.length).toBeGreaterThan(0);

		const out = collapseVariants([base, twin], { table: { families } });
		expect(out.map(m => m.id)).toEqual(["TEE/kimi-k2.5"]);
		// Effort routing to a live thinking id forces reasoning even though
		// upstream marked neither member.
		expect(out[0]?.reasoning).toBe(true);
		expect(out[0]?.thinking?.effortRouting?.off).toBe("TEE/kimi-k2.5");
	});

	it("never merges price-divergent twins or orphan thinking ids", () => {
		// Different pricing — distinct SKUs.
		expect(
			deriveThinkingPairFamilies([
				pairSpec("kimi-k2"),
				pairSpec("kimi-k2-thinking", {
					reasoning: true,
					thinking: PAIR_THINKING,
					cost: { input: 3, output: 6, cacheRead: 0.1, cacheWrite: 0 },
				}),
			]),
		).toEqual([]);
		// No bare twin.
		expect(
			deriveThinkingPairFamilies([
				pairSpec("moonshot.kimi-k2-thinking", { reasoning: true, thinking: PAIR_THINKING }),
			]),
		).toEqual([]);
		// Api mismatch.
		expect(
			deriveThinkingPairFamilies([
				pairSpec("kimi-k2"),
				{
					...pairSpec("kimi-k2-thinking", { reasoning: true, thinking: PAIR_THINKING }),
					api: "anthropic-messages",
				} as unknown as ModelSpec<"openai-completions">,
			]),
		).toEqual([]);
	});

	it("defers to hand-table families for claimed ids", () => {
		const specs = [
			memberSpec("gemini-2.5-flash"),
			memberSpec("gemini-2.5-flash-thinking", { thinking: PAIR_THINKING }),
		];
		expect(deriveThinkingPairFamilies(specs, antigravityTable)).toEqual([]);
	});
});

describe("collapseVariants across providers", () => {
	it("applies hand tables and derived pairs per provider", () => {
		const out = collapseVariants([
			memberSpec("gemini-3.8-flash-medium"),
			pairSpec("kimi-k2"),
			pairSpec("kimi-k2-thinking", { reasoning: true, thinking: PAIR_THINKING }),
			// Same ids on a provider without a table or a live bare twin stay.
			pairSpec("qwen3-vl-32b-thinking", { provider: "aimlapi", reasoning: true, thinking: PAIR_THINKING }),
		]);

		expect(out.map(m => `${m.provider}/${m.id}`).sort()).toEqual([
			"aimlapi/qwen3-vl-32b-thinking",
			"google-antigravity/gemini-3.8-flash",
			"venice/kimi-k2",
		]);
	});
});

describe("Devin tier routing", () => {
	const family = (id: string) => {
		const found = devinTable.families.find(f => f.id === id);
		if (!found) throw new Error(`Devin family ${id} missing`);
		return found;
	};

	it("routes user efforts 1:1 onto per-tier siblings including max", () => {
		const opus = family("claude-opus-4-8");
		expect(opus.routing).toEqual({
			[Effort.Low]: "claude-opus-4-8-low",
			[Effort.Medium]: "claude-opus-4-8-medium",
			[Effort.High]: "claude-opus-4-8-high",
			[Effort.XHigh]: "claude-opus-4-8-xhigh",
			[Effort.Max]: "claude-opus-4-8-max",
		});
		expect(opus.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh, Effort.Max]);
		expect(opus.thinking?.requiresEffort).toBe(true);

		const sol = family("gpt-5-6-sol");
		expect(sol.routing[Effort.Max]).toBe("gpt-5-6-sol-max");
		expect(sol.routing[Effort.Low]).toBe("gpt-5-6-sol-low");
		expect(sol.routing.off).toBe("gpt-5-6-sol-none");
		expect(sol.routing[Effort.Minimal]).toBeUndefined();

		const solFast = family("gpt-5-6-sol-fast");
		expect(solFast.routing[Effort.Max]).toBe("gpt-5-6-sol-max-priority");
		expect(solFast.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh, Effort.Max]);
	});

	it("keeps pre-5.6 families without a -max sibling on the xhigh ceiling", () => {
		const gpt55 = family("gpt-5-5");
		expect(gpt55.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh]);
		expect(gpt55.routing[Effort.Minimal]).toBeUndefined();
		expect(gpt55.routing[Effort.Max]).toBeUndefined();
	});

	it("routes current Devin families onto their account-visible wire variants", () => {
		const fable = family("claude-fable-5");
		expect(fable.routing[Effort.Low]).toBe("claude-5-fable-low");
		expect(fable.routing[Effort.Max]).toBe("claude-5-fable-max");

		const swe = family("swe-1-7");
		expect(swe.thinking?.efforts).toEqual([Effort.Medium, Effort.Max]);
		expect(swe.routing[Effort.Medium]).toBe("swe-1-7-medium");
		expect(swe.routing[Effort.Max]).toBe("swe-1-7");

		const gemini = family("gemini-3-6-flash");
		expect(gemini.routing[Effort.Minimal]).toBe("gemini-3-6-flash-minimal");
		expect(gemini.routing[Effort.High]).toBe("gemini-3-6-flash-high");

		const inkling = family("inkling");
		expect(inkling.routing.off).toBe("inkling-none");
		expect(inkling.routing[Effort.Max]).toBe("inkling-max");
	});

	it("collapses entitled current variants and resolves the selected effort to the wire UID", () => {
		const rawIds = [
			"claude-5-fable-low",
			"claude-5-fable-medium",
			"claude-5-fable-high",
			"claude-5-fable-xhigh",
			"claude-5-fable-max",
			"swe-1-7-medium",
			"swe-1-7",
		];
		const specs = rawIds.map((id): ModelSpec<"devin-agent"> => ({
			id,
			name: id,
			api: "devin-agent",
			provider: "devin",
			baseUrl: "https://server.codeium.com",
			reasoning: true,
			input: ["text"],
			supportsTools: true,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 200_000,
			maxTokens: 64_000,
		}));

		const collapsed = collapseVariants(specs, { table: devinTable });
		expect(collapsed.map(model => model.id).sort()).toEqual(["claude-fable-5", "swe-1-7"]);

		const fable = collapsed.find(model => model.id === "claude-fable-5");
		const swe = collapsed.find(model => model.id === "swe-1-7");
		if (!fable || !swe) throw new Error("Current Devin families did not collapse");
		expect(resolveWireModelId(buildModel(fable), Effort.XHigh)).toBe("claude-5-fable-xhigh");
		expect(resolveWireModelId(buildModel(swe), Effort.Medium)).toBe("swe-1-7-medium");
		expect(resolveWireModelId(buildModel(swe), Effort.Max)).toBe("swe-1-7");
	});

	it("routes the added fallback families onto their native tiers and default wire UIDs", () => {
		const gemini = family("gemini-3-7-flash");
		expect(gemini.routing).toEqual({
			[Effort.Minimal]: "gemini-3-7-flash-minimal",
			[Effort.Low]: "gemini-3-7-flash-low",
			[Effort.Medium]: "gemini-3-7-flash-medium",
			[Effort.High]: "gemini-3-7-flash-high",
		});
		// The native default tier leads `members`, so it is also the collapsed
		// spec's `requestModelId`.
		expect(gemini.members[0]).toBe("gemini-3-7-flash-medium");
		expect(gemini.thinking?.defaultLevel).toBe(Effort.Medium);

		const lightning = family("swe-1-7-lightning");
		expect(lightning.routing).toEqual({
			[Effort.Medium]: "swe-1-7-lightning-medium",
			[Effort.Max]: "swe-1-7-lightning",
		});
		expect(lightning.members[0]).toBe("swe-1-7-lightning-medium");

		const grok = family("grok-4-6");
		expect(grok.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh]);
		expect(grok.routing[Effort.XHigh]).toBe("grok-4-6-xhigh");
		expect(grok.members[0]).toBe("grok-4-6-medium");

		for (const id of ["deepseek-v4-flash", "deepseek-v4-pro"]) {
			const deepseek = family(id);
			expect(deepseek.thinking?.efforts).toEqual([Effort.Low, Effort.High, Effort.Max]);
			expect(deepseek.routing[Effort.Max]).toBe(`${id}-max`);
			expect(deepseek.members[0]).toBe(`${id}-high`);
		}

		const nemotron = family("nemotron-3-ultra");
		expect(nemotron.routing.off).toBe("nemotron-3-ultra-none");
		expect(nemotron.thinking?.efforts).toEqual([Effort.Medium, Effort.High]);
		expect(nemotron.thinking?.requiresEffort).toBeUndefined();
		expect(nemotron.members[0]).toBe("nemotron-3-ultra-high");
	});

	it("renames the single-uid Haiku config without inventing an effort surface", () => {
		const haiku = family("claude-haiku-4-5");
		expect(haiku.members).toEqual(["MODEL_PRIVATE_11"]);
		expect(haiku.thinking).toBeUndefined();

		const collapsed = collapseVariants([devinMemberSpec("MODEL_PRIVATE_11")], { table: devinTable });
		const spec = collapsed[0];
		if (!spec) throw new Error("Haiku did not collapse");
		expect(spec.id).toBe("claude-haiku-4-5");
		expect(spec.requestModelId).toBe("MODEL_PRIVATE_11");
		// Devin encodes effort in the uid: one uid means no controllable
		// surface, but the model still reasons.
		expect(spec.reasoning).toBe(true);
		expect(spec.thinking).toBeUndefined();
		expect(buildModel(spec).thinking).toBeUndefined();
	});

	it("collapses the added families and defaults the wire UID to the native tier", () => {
		const collapsed = collapseVariants(
			[
				"gemini-3-7-flash-minimal",
				"gemini-3-7-flash-low",
				"gemini-3-7-flash-medium",
				"gemini-3-7-flash-high",
				"swe-1-7-lightning-medium",
				"swe-1-7-lightning",
			].map(id => devinMemberSpec(id)),
			{ table: devinTable },
		);
		expect(collapsed.map(spec => spec.id).sort()).toEqual(["gemini-3-7-flash", "swe-1-7-lightning"]);

		const gemini = collapsed.find(spec => spec.id === "gemini-3-7-flash");
		const lightning = collapsed.find(spec => spec.id === "swe-1-7-lightning");
		if (!gemini || !lightning) throw new Error("Added Devin families did not collapse");
		expect(gemini.requestModelId).toBe("gemini-3-7-flash-medium");
		const geminiModel = buildModel(gemini);
		expect(geminiModel.thinking?.defaultLevel).toBe(Effort.Medium);
		expect(resolveWireModelId(geminiModel, Effort.High)).toBe("gemini-3-7-flash-high");
		// No `off` route: thinking-off falls back to the default wire uid.
		expect(resolveWireModelId(geminiModel, undefined)).toBe("gemini-3-7-flash-medium");
		expect(resolveWireModelId(buildModel(lightning), Effort.Max)).toBe("swe-1-7-lightning");
	});
});

describe("Cursor Grok tier routing (issue #8803)", () => {
	const RAW_SIBLINGS = [
		"cursor-grok-4.5-high",
		"cursor-grok-4.5-high-fast",
		"cursor-grok-4.5-low",
		"cursor-grok-4.5-low-fast",
		"cursor-grok-4.5-medium",
		"cursor-grok-4.5-medium-fast",
		"cursor-grok-4.6-high",
		"cursor-grok-4.6-high-fast",
		"cursor-grok-4.6-low",
		"cursor-grok-4.6-low-fast",
		"cursor-grok-4.6-medium",
		"cursor-grok-4.6-medium-fast",
		"cursor-grok-4.6-xhigh",
		"cursor-grok-4.6-xhigh-fast",
	];

	it("collapses the 14 effort siblings into four logical models, split by the -fast lane", () => {
		const collapsed = collapseVariants(
			RAW_SIBLINGS.map(id => cursorMemberSpec(id)),
			{ table: cursorTable },
		);
		expect(collapsed.map(model => model.id).sort()).toEqual([
			"cursor-grok-4.5",
			"cursor-grok-4.5-fast",
			"cursor-grok-4.6",
			"cursor-grok-4.6-fast",
		]);

		const g46 = collapsed.find(model => model.id === "cursor-grok-4.6");
		if (!g46) throw new Error("cursor-grok-4.6 did not collapse");
		expect(g46.name).toBe("Grok 4.6");
		// Effort route forces reasoning even though every member said false.
		expect(g46.reasoning).toBe(true);
		expect(g46.thinking?.mode).toBe("effort");
		expect(g46.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh]);
		expect(g46.thinking?.requiresEffort).toBe(true);
	});

	it("routes each user effort onto the live sibling wire id per service-tier lane", () => {
		const collapsed = collapseVariants(
			RAW_SIBLINGS.map(id => cursorMemberSpec(id)),
			{ table: cursorTable },
		);
		const model = (id: string) => {
			const found = collapsed.find(m => m.id === id);
			if (!found) throw new Error(`${id} did not collapse`);
			return buildModel(found as ModelSpec<"cursor-agent">);
		};

		expect(resolveWireModelId(model("cursor-grok-4.6"), Effort.XHigh)).toBe("cursor-grok-4.6-xhigh");
		expect(resolveWireModelId(model("cursor-grok-4.6"), Effort.Low)).toBe("cursor-grok-4.6-low");
		expect(resolveWireModelId(model("cursor-grok-4.6-fast"), Effort.High)).toBe("cursor-grok-4.6-high-fast");
		expect(resolveWireModelId(model("cursor-grok-4.5"), Effort.Medium)).toBe("cursor-grok-4.5-medium");
		// 4.5 has no xhigh sibling: the ceiling stays at high.
		expect(model("cursor-grok-4.5").thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High]);
	});

	it("collapses built Grok 4.7 rows per service-tier lane before their per-row effort ladders leak (#12773)", () => {
		const tiers = ["low", "medium", "high", "xhigh"];
		const collapsed = collapseBuiltVariants(
			[...tiers.map(tier => `grok-4.7-${tier}`), ...tiers.map(tier => `grok-4.7-${tier}-fast`)].map(id =>
				buildModel({ ...cursorMemberSpec(id), reasoning: true }),
			),
		);
		expect(collapsed.map(model => model.id)).toEqual(["grok-4.7", "grok-4.7-fast"]);

		const [grok, fast] = collapsed;
		if (!grok || !fast) throw new Error("grok-4.7 lanes did not collapse");
		expect(grok.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh]);
		expect(fast.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh]);
		expect(resolveWireModelId(grok, Effort.XHigh)).toBe("grok-4.7-xhigh");
		expect(resolveWireModelId(fast, Effort.XHigh)).toBe("grok-4.7-xhigh-fast");
		expect(resolveWireModelId(fast, Effort.Low)).toBe("grok-4.7-low-fast");
	});

	it("defaults the collapsed row to -medium and clamps effort-less to -medium (issue #9478)", () => {
		const collapsed = collapseVariants(
			RAW_SIBLINGS.map(id => cursorMemberSpec(id)),
			{ table: cursorTable },
		);
		const defaults = [
			["cursor-grok-4.5", "cursor-grok-4.5-medium"],
			["cursor-grok-4.5-fast", "cursor-grok-4.5-medium-fast"],
			["cursor-grok-4.6", "cursor-grok-4.6-medium"],
			["cursor-grok-4.6-fast", "cursor-grok-4.6-medium-fast"],
		] as const;
		for (const [id, requestModelId] of defaults) {
			const spec = collapsed.find(model => model.id === id);
			if (!spec) throw new Error(`${id} did not collapse`);
			// The Start plan refuses the -low floor; the collapsed default and
			// effort-less clamp target must both be the fixed-settings tier.
			expect(spec.requestModelId).toBe(requestModelId);
			const model = buildModel(spec as ModelSpec<"cursor-agent">);
			expect(defaultSupportedEffort(model)).toBe(Effort.Medium);
			expect(resolveWireModelId(model, defaultSupportedEffort(model))).toBe(requestModelId);
		}
	});

	it("re-points a stale collapsed snapshot pinned to -low back to -medium (issue #9478)", () => {
		const stale: ModelSpec<"cursor-agent"> = {
			...cursorMemberSpec("cursor-grok-4.6"),
			name: "Grok 4.6",
			reasoning: true,
			requestModelId: "cursor-grok-4.6-low",
			thinking: {
				mode: "effort",
				efforts: [Effort.Low, Effort.Medium, Effort.High, Effort.XHigh],
				requiresEffort: true,
				effortRouting: {
					[Effort.Low]: "cursor-grok-4.6-low",
					[Effort.Medium]: "cursor-grok-4.6-medium",
					[Effort.High]: "cursor-grok-4.6-high",
					[Effort.XHigh]: "cursor-grok-4.6-xhigh",
				},
			},
		};
		// Bundled/cache row (no live siblings) — the offline pass-through path.
		const offline = collapseBuiltVariants([buildModel(stale)]);
		expect(offline.find(m => m.id === "cursor-grok-4.6")?.requestModelId).toBe("cursor-grok-4.6-medium");
		// Bundled row merged with live siblings — the online discovery path.
		const online = collapseBuiltVariants([
			buildModel(stale),
			...RAW_SIBLINGS.filter(id => id.startsWith("cursor-grok-4.6-") && !id.endsWith("-fast")).map(id =>
				buildModel(cursorMemberSpec(id)),
			),
		]);
		expect(online.find(m => m.id === "cursor-grok-4.6")?.requestModelId).toBe("cursor-grok-4.6-medium");

		// Account-specific discovery can omit the preferred tier. Do not route
		// effort-less requests to a sibling the account did not advertise.
		const withoutMedium = collapseBuiltVariants([
			buildModel({ ...stale, requestModelId: "cursor-grok-4.6-medium" }),
			buildModel(cursorMemberSpec("cursor-grok-4.6-low")),
			buildModel(cursorMemberSpec("cursor-grok-4.6-high")),
		]);
		expect(withoutMedium.find(m => m.id === "cursor-grok-4.6")?.requestModelId).toBe("cursor-grok-4.6-low");
	});
});

describe("Cursor GPT-5.6 tier routing (issue #9025)", () => {
	const TIERS = ["none", "low", "medium", "high", "xhigh", "max"];
	const RAW_SIBLINGS = ["luna", "sol", "terra"].flatMap(variant =>
		TIERS.flatMap(tier => [`gpt-5.6-${variant}-${tier}`, `gpt-5.6-${variant}-${tier}-fast`]),
	);

	it("collapses the 36 effort siblings into six logical models, split by the -fast lane", () => {
		expect(RAW_SIBLINGS).toHaveLength(36);
		const collapsed = collapseVariants(
			RAW_SIBLINGS.map(id => cursorMemberSpec(id)),
			{ table: cursorTable },
		);
		expect(collapsed.map(model => model.id).sort()).toEqual([
			"gpt-5.6-luna",
			"gpt-5.6-luna-fast",
			"gpt-5.6-sol",
			"gpt-5.6-sol-fast",
			"gpt-5.6-terra",
			"gpt-5.6-terra-fast",
		]);

		const luna = collapsed.find(model => model.id === "gpt-5.6-luna");
		if (!luna) throw new Error("gpt-5.6-luna did not collapse");
		expect(luna.name).toBe("GPT-5.6 Luna");
		// Effort route forces reasoning even though every member said false.
		expect(luna.reasoning).toBe(true);
		expect(luna.thinking?.mode).toBe("effort");
		expect(luna.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh, Effort.Max]);
	});

	it("routes each user effort onto the live sibling wire id per service-tier lane", () => {
		const collapsed = collapseVariants(
			RAW_SIBLINGS.map(id => cursorMemberSpec(id)),
			{ table: cursorTable },
		);
		const model = (id: string) => {
			const found = collapsed.find(m => m.id === id);
			if (!found) throw new Error(`${id} did not collapse`);
			return buildModel(found as ModelSpec<"cursor-agent">);
		};

		// `-none` is the thinking-off tier; efforts route onto the standard lane.
		expect(resolveWireModelId(model("gpt-5.6-sol"), undefined)).toBe("gpt-5.6-sol-none");
		expect(resolveWireModelId(model("gpt-5.6-sol"), Effort.Max)).toBe("gpt-5.6-sol-max");
		expect(resolveWireModelId(model("gpt-5.6-sol"), Effort.Low)).toBe("gpt-5.6-sol-low");
		// The -fast lane is a distinct SKU routing onto its own sibling ids.
		expect(resolveWireModelId(model("gpt-5.6-terra-fast"), Effort.High)).toBe("gpt-5.6-terra-high-fast");
		expect(resolveWireModelId(model("gpt-5.6-terra-fast"), Effort.XHigh)).toBe("gpt-5.6-terra-xhigh-fast");
	});

	it("keeps the max-mode flag on the collapsed row when only the premium tiers carry it", () => {
		// Cursor marks the extended-context tiers `maxMode` and leaves the cheap
		// ones unmarked, so the flag disagrees across a reviewed family whose
		// first member is the `-none` tier. The collapsed row is what the
		// cursor-agent transport reads for the `max_mode` request flag, while the
		// wire id comes from effort routing — dropping the flag sends
		// `max_mode: false` on a max-mode-only wire id.
		const maxModeTier = /-(xhigh|max)(-fast)?$/;
		const collapsed = collapseVariants(
			RAW_SIBLINGS.map(id => cursorMemberSpec(id, { cursorMaxMode: maxModeTier.test(id) })),
			{ table: cursorTable },
		);
		const luna = collapsed.find(model => model.id === "gpt-5.6-luna");
		if (!luna) throw new Error("gpt-5.6-luna did not collapse");

		expect(luna.cursorMaxMode).toBe(true);
		expect(resolveWireModelId(buildModel(luna as ModelSpec<"cursor-agent">), Effort.Max)).toBe("gpt-5.6-luna-max");
		// The OR cannot say which tier needed max mode, so the transport reads the
		// members' own markers per wire id.
		expect(luna.cursorMaxModeRoutes).toEqual({
			"gpt-5.6-luna-none": false,
			"gpt-5.6-luna-low": false,
			"gpt-5.6-luna-medium": false,
			"gpt-5.6-luna-high": false,
			"gpt-5.6-luna-xhigh": true,
			"gpt-5.6-luna-max": true,
		});
	});

	it("lifts the live max-mode flag onto a bundled collapsed row (existing-collapsed merge)", () => {
		// The committed bundled row froze `cursorMaxMode: false` from the `-none`
		// tier. Merging it with live `GetUsableModels` tiers takes the
		// existing-collapsed pass-through, which keeps the snapshot verbatim — so
		// the live `-xhigh`/`-max` marks have to be lifted onto the row the
		// cursor-agent transport reads for the `max_mode` request flag.
		const bundled: ModelSpec<"cursor-agent"> = {
			...cursorMemberSpec("gpt-5.6-luna"),
			name: "GPT-5.6 Luna",
			reasoning: true,
			cursorMaxMode: false,
			requestModelId: "gpt-5.6-luna-none",
			thinking: {
				mode: "effort",
				efforts: [Effort.Low, Effort.Medium, Effort.High, Effort.XHigh, Effort.Max],
				requiresEffort: true,
				effortRouting: {
					off: "gpt-5.6-luna-none",
					[Effort.Low]: "gpt-5.6-luna-low",
					[Effort.Medium]: "gpt-5.6-luna-medium",
					[Effort.High]: "gpt-5.6-luna-high",
					[Effort.XHigh]: "gpt-5.6-luna-xhigh",
					[Effort.Max]: "gpt-5.6-luna-max",
				},
			},
		};
		const markedTiers = TIERS.map(tier =>
			buildModel(cursorMemberSpec(`gpt-5.6-luna-${tier}`, { cursorMaxMode: tier === "xhigh" || tier === "max" })),
		);
		const unmarkedTiers = TIERS.map(tier => buildModel(cursorMemberSpec(`gpt-5.6-luna-${tier}`)));

		const merged = collapseBuiltVariants([buildModel(bundled), ...markedTiers]);
		const luna = merged.find(model => model.id === "gpt-5.6-luna");
		if (!luna) throw new Error("gpt-5.6-luna did not survive the merge");
		expect(luna.cursorMaxMode).toBe(true);
		expect(resolveWireModelId(luna, Effort.Max)).toBe("gpt-5.6-luna-max");
		expect(luna.cursorMaxModeRoutes?.["gpt-5.6-luna-max"]).toBe(true);
		expect(luna.cursorMaxModeRoutes?.["gpt-5.6-luna-low"]).toBe(false);

		// A roster that marks no tier leaves the snapshot's own flag alone.
		const unmarked = collapseBuiltVariants([buildModel(bundled), ...unmarkedTiers]);
		expect(unmarked.find(model => model.id === "gpt-5.6-luna")?.cursorMaxMode).toBe(false);
	});

	it("derives max mode from bundled routing when Cursor discovery is unavailable", () => {
		const bundled: ModelSpec<"cursor-agent"> = {
			...cursorMemberSpec("gpt-5.6-luna"),
			name: "GPT-5.6 Luna",
			cursorMaxMode: false,
			requestModelId: "gpt-5.6-luna-none",
			thinking: {
				mode: "effort",
				efforts: [Effort.Low, Effort.Medium, Effort.High, Effort.XHigh, Effort.Max],
				effortRouting: {
					off: "gpt-5.6-luna-none",
					[Effort.Low]: "gpt-5.6-luna-low",
					[Effort.Medium]: "gpt-5.6-luna-medium",
					[Effort.High]: "gpt-5.6-luna-high",
					[Effort.XHigh]: "gpt-5.6-luna-xhigh",
					[Effort.Max]: "gpt-5.6-luna-max",
				},
			},
		};
		const [luna] = collapseVariants([bundled], { table: cursorTable });
		if (!luna) throw new Error("bundled GPT-5.6 row disappeared");

		expect(luna.cursorMaxMode).toBe(true);
	});
});

describe("Cursor generic tier routing (issue #9237)", () => {
	const TIERS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

	it("collapses live pure-effort siblings into logical standard and fast lanes", () => {
		const raw = TIERS.flatMap(tier => [
			cursorMemberSpec(`gpt-5.5-${tier}`, { name: `GPT-5.5 ${tier}` }),
			cursorMemberSpec(`gpt-5.5-${tier}-fast`, { name: `GPT-5.5 ${tier} Fast` }),
		]);
		const collapsed = collapseVariants(raw);
		expect(collapsed.map(model => model.id).sort()).toEqual(["gpt-5.5", "gpt-5.5-fast"]);

		const standard = collapsed.find(model => model.id === "gpt-5.5");
		const fast = collapsed.find(model => model.id === "gpt-5.5-fast");
		if (!standard || !fast) throw new Error("GPT-5.5 lanes did not collapse");
		expect(standard.name).toBe("GPT-5.5");
		expect(fast.name).toBe("GPT-5.5 Fast");
		expect(standard.thinking?.efforts).toEqual([
			Effort.Minimal,
			Effort.Low,
			Effort.Medium,
			Effort.High,
			Effort.XHigh,
			Effort.Max,
		]);
		expect(resolveWireModelId(buildModel(standard), undefined)).toBe("gpt-5.5-none");
		expect(resolveWireModelId(buildModel(standard), Effort.XHigh)).toBe("gpt-5.5-xhigh");
		expect(resolveWireModelId(buildModel(fast), Effort.Max)).toBe("gpt-5.5-max-fast");
	});

	it("maps Cursor's extra-high suffix to xhigh", () => {
		const raw = ["none", "low", "extra-high"].map(tier =>
			cursorMemberSpec(`gpt-5.5-${tier}`, { name: `GPT-5.5 ${tier}` }),
		);
		const collapsed = collapseVariants(raw);

		expect(collapsed.map(model => model.id)).toEqual(["gpt-5.5"]);
		const model = collapsed[0];
		if (!model) throw new Error("GPT-5.5 did not collapse");
		expect(model.thinking?.efforts).toEqual([Effort.Low, Effort.XHigh]);
		expect(resolveWireModelId(buildModel(model), Effort.XHigh)).toBe("gpt-5.5-extra-high");
	});

	it("keeps a reference-backed base and its existing thinking ladders intact", () => {
		const raw = [
			cursorMemberSpec("gpt-5.2", {
				reasoning: true,
				contextWindow: 400_000,
				thinking: { mode: "effort", efforts: [Effort.Low, Effort.Medium, Effort.High, Effort.XHigh] },
			}),
			cursorMemberSpec("gpt-5.2-low", { contextWindow: 400_000 }),
			cursorMemberSpec("gpt-5.2-high", {
				reasoning: true,
				contextWindow: 400_000,
				thinking: { mode: "effort", efforts: [Effort.Low, Effort.Medium, Effort.High, Effort.XHigh] },
			}),
		];
		const collapsed = collapseVariants(raw);

		expect(collapsed.map(model => model.id)).toEqual(raw.map(model => model.id));
		expect(collapsed[0]?.thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High, Effort.XHigh]);
		expect(collapsed[0]?.contextWindow).toBe(400_000);
	});

	it("keeps tier-looking siblings separate when their context SKUs differ", () => {
		const raw = [
			cursorMemberSpec("gemini-3.6-flash-low", { contextWindow: 200_000 }),
			cursorMemberSpec("gemini-3.6-flash-high", { contextWindow: 1_000_000 }),
		];

		expect(collapseVariants(raw).map(model => model.id)).toEqual(raw.map(model => model.id));
	});

	it("keeps Cursor max-mode SKUs out of effort-only families", () => {
		const raw = [
			cursorMemberSpec("gemini-3.7-flash-low", { cursorMaxMode: false }),
			cursorMemberSpec("gemini-3.7-flash-max", { cursorMaxMode: true }),
		];

		expect(collapseVariants(raw).map(model => model.id)).toEqual(raw.map(model => model.id));
	});

	it("does not treat product -max or split -thinking names as effort-only families", () => {
		const ids = [
			"gpt-5.1-codex-low",
			"gpt-5.1-codex-max",
			"gpt-5.1-codex-max-high",
			"gpt-5.1-codex-max-xhigh",
			"claude-opus-5-low",
			"claude-opus-5-medium",
			"claude-opus-5-high",
			"claude-opus-5-thinking-xhigh",
			"claude-opus-5-thinking-max",
		];

		expect(collapseVariants(ids.map(id => cursorMemberSpec(id))).map(model => model.id)).toEqual(ids);
	});

	it("dedupes live tiers beside matching collapsed static rows", () => {
		const raw = ["low", "high"].flatMap(tier => [
			buildModel(cursorMemberSpec(`review-merged-9237-${tier}`)),
			buildModel(cursorMemberSpec(`review-merged-9237-${tier}-fast`)),
		]);
		const staticModels = collapseBuiltVariants(raw);
		const merged = collapseBuiltVariants([...staticModels, ...raw]);

		expect(merged.map(model => model.id).sort()).toEqual(["review-merged-9237", "review-merged-9237-fast"]);
		const standard = merged.find(model => model.id === "review-merged-9237");
		if (!standard) throw new Error("Standard merged family did not collapse");
		expect(resolveWireModelId(standard, Effort.High)).toBe("review-merged-9237-high");
	});

	it("retargets internal model references whose destination tier collapses", () => {
		const sourceOverrides = {
			contextPromotionTarget: "cursor/review-promotion-target-low",
			compactionModel: "review-promotion-target-high",
		};
		const collapsed = collapseVariants([
			cursorMemberSpec("review-promotion-target-low"),
			cursorMemberSpec("review-promotion-target-high"),
			cursorMemberSpec("review-promotion-source-none", sourceOverrides),
			cursorMemberSpec("review-promotion-source-high", sourceOverrides),
		]);
		const source = collapsed.find(model => model.id === "review-promotion-source");

		expect(source?.contextPromotionTarget).toBe("cursor/review-promotion-target");
		expect(source?.compactionModel).toBe("review-promotion-target");
	});

	it("preserves references to live tiers after an earlier family becomes unsafe", () => {
		collapseVariants([
			cursorMemberSpec("review-changing-target-low"),
			cursorMemberSpec("review-changing-target-high"),
		]);
		const collapsed = collapseVariants([
			cursorMemberSpec("review-changing-target-low", { contextWindow: 200_000 }),
			cursorMemberSpec("review-changing-target-high", { contextWindow: 1_000_000 }),
			cursorMemberSpec("review-changing-source", {
				contextPromotionTarget: "cursor/review-changing-target-low",
				compactionModel: "review-changing-target-high",
			}),
		]);
		const source = collapsed.find(model => model.id === "review-changing-source");

		expect(source?.contextPromotionTarget).toBe("cursor/review-changing-target-low");
		expect(source?.compactionModel).toBe("review-changing-target-high");
	});
});

describe("variant aliases", () => {
	it("resolves members and recycled ids per provider", () => {
		expect(resolveVariantSelector("google-antigravity", "claude-sonnet-5-5-high")).toBe("claude-sonnet-5-5");
		expect(resolveVariantSelector("google-antigravity", "claude-opus-5-5-low")).toBe("claude-opus-5-5");
		expect(resolveVariantSelector("google-gemini-cli", "gemini-pro-agent")).toBe("gemini-3.1-pro");
		expect(resolveVariantSelector("google-antigravity", "gemini-2.5-flash-thinking")).toBe("gemini-2.5-flash");
		expect(resolveVariantSelector("google-antigravity", "gemini-2.5-flash-lite")).toBeUndefined();
		expect(resolveVariantSelector("anthropic", "claude-sonnet-4-6-thinking")).toBeUndefined();
	});

	it("names the declaring providers in bare-id lookups", () => {
		const hit = resolveBareVariantSelector("GEMINI-2.5-FLASH-THINKING");
		expect(hit?.id).toBe("gemini-2.5-flash");
		expect(hit?.providers).toContain("google-antigravity");
		expect(hit?.providers).toContain("google-gemini-cli");
		expect(resolveBareVariantSelector("gpt-4o")).toBeUndefined();
	});

	it("reverse sources cover members and recycled ids", () => {
		const sources = getVariantAliasSources("google-antigravity", "claude-sonnet-5-5");
		expect(sources).toContain("claude-sonnet-5-5-low");
		expect(sources).toContain("claude-sonnet-5-5-medium");
		expect(sources).toContain("claude-sonnet-5-5-high");
		expect(getVariantAliasSources("openai", "gpt-4o")).toEqual([]);
	});

	it("registers selector aliases for dynamically collapsed Cursor tier families", () => {
		const ids = [
			"review-model-9237-low",
			"review-model-9237-high",
			"review-model-9237-low-fast",
			"review-model-9237-high-fast",
		];
		const collapsed = collapseVariants(ids.map(id => cursorMemberSpec(id)));
		expect(collapsed.map(model => model.id).sort()).toEqual(["review-model-9237", "review-model-9237-fast"]);

		expect(resolveVariantSelector("cursor", "review-model-9237-high")).toBe("review-model-9237");
		expect(resolveVariantSelector("CURSOR", "review-model-9237-low-fast")).toBe("review-model-9237-fast");
		expect(resolveBareVariantSelector("review-model-9237-high")).toEqual({
			id: "review-model-9237",
			providers: ["cursor"],
		});
		expect(getVariantAliasSources("cursor", "review-model-9237")).toEqual([
			"review-model-9237-low",
			"review-model-9237-high",
		]);
		expect(getVariantAliasSources("cursor", "review-model-9237-fast")).toEqual([
			"review-model-9237-low-fast",
			"review-model-9237-high-fast",
		]);
	});

	it("keeps provider-scoped aliases out of bare-id and reverse lookups", () => {
		expect(resolveVariantSelector("devin", "opus")).toBe("claude-opus-5");
		expect(resolveVariantSelector("devin", "claude")).toBe("claude-sonnet-5");
		expect(resolveVariantSelector("devin", "sonnet")).toBe("claude-sonnet-5");
		expect(resolveVariantSelector("devin", "haiku")).toBe("claude-haiku-4-5");
		expect(resolveVariantSelector("devin", "gemini")).toBe("gemini-3-7-flash");
		expect(resolveVariantSelector("devin", "gpt")).toBe("gpt-5-6-terra");
		expect(resolveVariantSelector("devin", "codex")).toBe("gpt-5-3-codex");
		expect(resolveVariantSelector("devin", "SWE")).toBe("swe-1-7-lightning");

		// Generic labels must stay meaningless without a provider.
		for (const alias of ["opus", "claude", "sonnet", "haiku", "gemini", "gpt", "codex", "swe"]) {
			expect(resolveBareVariantSelector(alias)).toBeUndefined();
			expect(resolveVariantSelector("google-antigravity", alias)).toBeUndefined();
		}
		// ...and must never re-key config off the collapsed model.
		expect(getVariantAliasSources("devin", "claude-opus-5")).not.toContain("opus");
	});

	it("resolves the dotted native Devin spellings to hyphenated logical ids", () => {
		expect(resolveVariantSelector("devin", "gpt-5.6-terra")).toBe("gpt-5-6-terra");
		expect(resolveVariantSelector("devin", "gpt-5.6-sol")).toBe("gpt-5-6-sol");
		expect(resolveVariantSelector("devin", "gpt-5.6-luna")).toBe("gpt-5-6-luna");
		expect(resolveVariantSelector("devin", "gemini-3.7-flash")).toBe("gemini-3-7-flash");
		expect(resolveVariantSelector("devin", "swe-1.7")).toBe("swe-1-7");
		expect(resolveVariantSelector("devin", "swe-1.7-lightning")).toBe("swe-1-7-lightning");
		expect(resolveVariantSelector("devin", "grok-4.6")).toBe("grok-4-6");
		expect(resolveVariantSelector("devin", "glm-5.2")).toBe("glm-5-2");
		expect(resolveVariantSelector("devin", "claude-haiku-4.5")).toBe("claude-haiku-4-5");
		// Members still win over provider aliases and stay bare-resolvable.
		expect(resolveVariantSelector("devin", "claude-opus-5-max")).toBe("claude-opus-5");
		expect(resolveBareVariantSelector("claude-opus-5-max")?.id).toBe("claude-opus-5");
	});
	it("scopes collapsed-spec detection to routing and hand-table families", () => {
		const collapsed = collapseVariants([memberSpec("gemini-3.8-flash-low")], { table: antigravityTable })[0];
		expect(collapsed && isCollapsedVariantSpec(collapsed)).toBe(true);
		expect(isCollapsedVariantSpec(memberSpec("gemini-3.8-flash-low"))).toBe(false);
		// Copilot long-context variants carry requestModelId but are NOT
		// collapsed specs — the generator rebake must not skip them.
		expect(
			isCollapsedVariantSpec(
				memberSpec("claude-opus-4.7-1m", { provider: "github-copilot", requestModelId: "claude-opus-4.7" }),
			),
		).toBe(false);
	});
});

describe("resolveWireModelId", () => {
	it("survives buildModel and routes per effort with requestModelId fallback", () => {
		const collapsed = collapseVariants(FLASH_TIERSET(), { table: antigravityTable })[0];
		const model = buildModel(collapsed as ModelSpec<"google-gemini-cli">);

		expect(model.thinking?.effortRouting).toEqual(collapsed?.thinking?.effortRouting);
		expect(model.thinking?.requiresEffort).toBe(true);
		expect(resolveWireModelId(model, Effort.High)).toBe("gemini-3.8-flash-high");
		expect(resolveWireModelId(model, Effort.Medium)).toBe("gemini-3.8-flash-medium");
		expect(resolveWireModelId(model, Effort.Minimal)).toBe("gemini-3.8-flash-low");
		expect(resolveWireModelId(model, undefined)).toBe("gemini-3.8-flash-low");

		// Dropped route (partial family) falls back to requestModelId.
		const partial = collapseVariants([memberSpec("gemini-3.8-flash-low")], { table: antigravityTable })[0];
		const partialModel = buildModel(partial as ModelSpec<"google-gemini-cli">);
		expect(resolveWireModelId(partialModel, Effort.High)).toBe("gemini-3.8-flash-low");

		// Models without routing serialize their own id.
		expect(resolveWireModelId(buildModel(memberSpec("gemini-2.5-flash-lite")), Effort.High)).toBe(
			"gemini-2.5-flash-lite",
		);
	});
});

describe("merge-point collapsing (resolveProviderModels)", () => {
	async function tempDb(): Promise<string> {
		const dir = await fs.mkdtemp(path.join(os.tmpdir(), "variant-collapse-"));
		return path.join(dir, "models.db");
	}

	it("converges stale static raw ids with collapsed dynamic results", async () => {
		const dbPath = await tempDb();
		const staleStatic = [
			memberSpec("gemini-3.1-pro-low"),
			memberSpec("gemini-3.1-pro-high"),
			memberSpec("gemini-2.5-flash-lite"),
		];
		const liveCollapsed = collapseVariants([memberSpec("gemini-3.1-pro-low"), memberSpec("gemini-3.1-pro-high")], {
			table: antigravityTable,
		});

		const result = await resolveProviderModels(
			{
				providerId: "google-antigravity",
				staticModels: staleStatic,
				fetchDynamicModels: () => Promise.resolve(liveCollapsed),
				cacheDbPath: dbPath,
			},
			"online",
		);
		expect(result.models.map(m => m.id).sort()).toEqual(["gemini-2.5-flash-lite", "gemini-3.1-pro"]);

		// The cache snapshot written above is collapsed too: a later resolve
		// whose dynamic fetch fails must not resurrect raw ids.
		const offline = await resolveProviderModels(
			{
				providerId: "google-antigravity",
				staticModels: staleStatic,
				fetchDynamicModels: () => Promise.resolve(null),
				cacheDbPath: dbPath,
			},
			"online",
		);
		expect(offline.models.filter(m => m.id.includes("gemini-3.1-pro")).map(m => m.id)).toEqual(["gemini-3.1-pro"]);
	});

	it("collapses X/X-thinking pairs for providers without a hand table", async () => {
		const dbPath = await tempDb();
		const result = await resolveProviderModels(
			{
				providerId: "venice",
				staticModels: [
					pairSpec("kimi-k2"),
					pairSpec("kimi-k2-thinking", { reasoning: true, thinking: PAIR_THINKING }),
				],
				cacheDbPath: dbPath,
			},
			"offline",
		);

		expect(result.models.map(m => m.id)).toEqual(["kimi-k2"]);
		const model = result.models[0];
		expect(model?.reasoning).toBe(true);
		expect(model && resolveWireModelId(model, Effort.High)).toBe("kimi-k2-thinking");
		expect(model && resolveWireModelId(model, undefined)).toBe("kimi-k2");
	});
});

describe("antigravity discovery collapsing", () => {
	const payload = {
		models: {
			"gemini-3.8-flash-low": {
				displayName: "Gemini 3.8 Flash (Low)",
				supportsThinking: true,
				supportsImages: true,
				maxTokens: 1_048_576,
				maxOutputTokens: 65_536,
			},
			"gemini-3.8-flash-medium": {
				displayName: "Gemini 3.8 Flash (Medium)",
				supportsThinking: true,
				supportsImages: true,
				maxTokens: 1_048_576,
				maxOutputTokens: 65_536,
			},
			"gemini-3.8-flash-high": {
				displayName: "Gemini 3.8 Flash (High)",
				supportsThinking: true,
				supportsImages: true,
				maxTokens: 1_048_576,
				maxOutputTokens: 65_536,
			},
			// Retired lane: rule-owned exclusion must drop it before collapsing.
			"gemini-3.7-flash-low": {
				displayName: "Gemini 3.7 Flash Low",
				supportsThinking: true,
				supportsImages: true,
				maxTokens: 1_048_576,
				maxOutputTokens: 65_536,
			},
			"claude-sonnet-5-5-low": {
				displayName: "Claude Sonnet 5.5 (Low)",
				supportsThinking: true,
				supportsImages: true,
			},
			"claude-sonnet-5-5-medium": {
				displayName: "Claude Sonnet 5.5 (Medium)",
				supportsThinking: true,
				supportsImages: true,
			},
			"claude-sonnet-5-5-high": {
				displayName: "Claude Sonnet 5.5 (High)",
				supportsThinking: true,
				supportsImages: true,
			},
			// Retired pair: both ids are excluded.
			"claude-sonnet-4-6": { displayName: "Claude Sonnet 4.6", supportsThinking: true, supportsImages: true },
			"claude-sonnet-4-6-thinking": {
				displayName: "Claude Sonnet 4.6 Thinking",
				supportsThinking: true,
				supportsImages: true,
			},
			"gemini-2.5-flash": { displayName: "Gemini 2.5 Flash", supportsThinking: true, supportsImages: true },
			"gemini-2.5-flash-thinking": { displayName: "Gemini 2.5 Flash Thinking", supportsThinking: true },
			chat_20706: { displayName: "Chat Internal" },
			"internal-model": { displayName: "Internal", isInternal: true },
		},
	};
	const fetcher = Object.assign(
		(_input: Parameters<typeof fetch>[0], _init?: Parameters<typeof fetch>[1]) =>
			Promise.resolve(new Response(JSON.stringify(payload), { status: 200 })),
		{ preconnect: fetch.preconnect },
	);

	it("returns collapsed logical entries and keeps the denylist", async () => {
		const models = await fetchAntigravityDiscoveryModels({ token: "t", endpoint: "https://cca.test", fetcher });

		expect(models?.map(m => m.id).sort()).toEqual(["claude-sonnet-5-5", "gemini-2.5-flash", "gemini-3.8-flash"]);
		// Retired lanes (3.7-flash, sonnet-4-6) are rule-excluded pre-collapse.
		const sonnet = models?.find(m => m.id === "claude-sonnet-5-5");
		expect(sonnet?.requestModelId).toBe("claude-sonnet-5-5-low");
		expect(sonnet?.thinking?.effortRouting?.[Effort.High]).toBe("claude-sonnet-5-5-high");
		expect(sonnet?.thinking?.requiresEffort).toBe(true);
		// The 2.5 pair collapses instead of denylisting the -thinking twin.
		const flash25 = models?.find(m => m.id === "gemini-2.5-flash");
		expect(flash25?.thinking?.effortRouting?.[Effort.High]).toBe("gemini-2.5-flash-thinking");
		expect(flash25?.thinking?.effortRouting?.off).toBe("gemini-2.5-flash");
		const flash38 = models?.find(m => m.id === "gemini-3.8-flash");
		expect(flash38?.requestModelId).toBe("gemini-3.8-flash-low");
		expect(flash38?.thinking).toEqual({
			mode: "google-level",
			efforts: [Effort.Minimal, Effort.Low, Effort.Medium, Effort.High],
			requiresEffort: true,
			effortRouting: {
				minimal: "gemini-3.8-flash-low",
				low: "gemini-3.8-flash-low",
				medium: "gemini-3.8-flash-medium",
				high: "gemini-3.8-flash-high",
			},
		});
	});

	it("discovers Gemini models through Antigravity before provisioning Cloud Code Assist", async () => {
		const requestedUrls: string[] = [];
		const geminiCliFetcher = Object.assign(
			(input: string | URL | Request, _init?: RequestInit) => {
				const url = String(input);
				requestedUrls.push(url);
				if (!url.startsWith(ANTIGRAVITY_PRIMARY_ENDPOINT)) {
					return Promise.resolve(new Response("Forbidden", { status: 403 }));
				}
				return Promise.resolve(new Response(JSON.stringify(payload), { status: 200 }));
			},
			{ preconnect: fetch.preconnect },
		);
		const options = googleGeminiCliModelManagerOptions({
			oauthToken: "t",
			endpoint: "https://cca.test",
			fetch: geminiCliFetcher,
		});
		const models = await options.fetchDynamicModels?.();

		expect(requestedUrls).toContain(`${ANTIGRAVITY_PRIMARY_ENDPOINT}/v1internal:fetchAvailableModels`);
		expect(models?.some(m => m.id === "claude-sonnet-5-5")).toBe(false);
		expect(models?.some(m => m.id.startsWith("claude-"))).toBe(false);
		expect(models?.every(m => m.baseUrl === "https://cca.test")).toBe(true);
		const flash = models?.find(m => m.id === "gemini-3.8-flash");
		expect(flash?.provider).toBe("google-gemini-cli");
		expect(flash?.requestModelId).toBe("gemini-3.8-flash-low");
		expect(flash?.thinking?.requiresEffort).toBe(true);
		expect(flash?.thinking?.effortRouting?.[Effort.High]).toBe("gemini-3.8-flash-high");
	});

	it("uses the primary daily endpoint by default", async () => {
		const requestedUrls: string[] = [];
		const defaultFetcher = Object.assign(
			(input: string | URL | Request, _init?: RequestInit) => {
				requestedUrls.push(String(input));
				return Promise.resolve(new Response(JSON.stringify(payload), { status: 200 }));
			},
			{ preconnect: fetch.preconnect },
		);

		const models = await fetchAntigravityDiscoveryModels({
			token: "t",
			fetcher: defaultFetcher,
		});

		const discoveryUrl = requestedUrls.find(url => url.includes("/v1internal:fetchAvailableModels"));
		expect(discoveryUrl).toBeDefined();
		expect(discoveryUrl).toContain(ANTIGRAVITY_PRIMARY_ENDPOINT);
		expect(models?.[0]?.baseUrl).toBe(ANTIGRAVITY_PRIMARY_ENDPOINT);
	});

	it("drops bundled rows the account roster does not serve, and keeps them when discovery fails", async () => {
		const roster = {
			models: {
				"claude-sonnet-4-5": { displayName: "Claude Sonnet 4.5", supportsThinking: true, maxTokens: 250_000 },
				"gemini-3.1-pro-low": { displayName: "Gemini 3.1 Pro (Low)", supportsThinking: true },
			},
		};
		const resolve = async (discovery: () => Response) => {
			const dir = await fs.mkdtemp(path.join(os.tmpdir(), "antigravity-roster-"));
			const fetcher = Object.assign(
				(input: string | URL | Request, _init?: RequestInit) =>
					Promise.resolve(
						String(input).includes(":fetchAvailableModels") ? discovery() : new Response("version: 2.19.1\n"),
					),
				{ preconnect: fetch.preconnect },
			);
			const options = googleAntigravityModelManagerOptions({ oauthToken: "t", fetch: fetcher });
			const result = await resolveProviderModels({ ...options, cacheDbPath: path.join(dir, "models.db") }, "online");
			return result.models.map(m => m.id);
		};

		// Bundled Claude 5.5 rows 404 on accounts whose roster omits them (#14328).
		const served = await resolve(() => Response.json(roster));
		expect(served).toContain("claude-sonnet-4-5");
		expect(served).toContain("gemini-3.1-pro");
		expect(served).not.toContain("claude-sonnet-5-5");
		expect(served).not.toContain("claude-opus-5-5");
		// Image SKUs are not chat rows and stay available to the image role.
		expect(served).toContain("gemini-3-pro-image");

		const unreachable = await resolve(() => new Response("Forbidden", { status: 403 }));
		expect(unreachable).toContain("claude-sonnet-5-5");
		expect(unreachable).toContain("claude-opus-5-5");
	});
});

describe("Devin GLM-5.2 collapse", () => {
	it("collapses the three 200K GLM-5.2 variants into one logical entry routing all efforts to the free glm-5-2 wire UID", () => {
		const out = collapseVariants(
			[
				devinMemberSpec("glm-5-2"),
				devinMemberSpec("glm-5-2-max"),
				devinMemberSpec("glm-5-2-none", { reasoning: false }),
			],
			{ table: devinTable },
		);

		expect(out).toHaveLength(1);
		const spec = out[0];
		expect(spec?.id).toBe("glm-5-2");
		expect(spec?.thinking?.effortRouting).toEqual({
			high: "glm-5-2",
			xhigh: "glm-5-2",
		});
	});

	it("collapses the three 1M GLM-5.2 variants into one paid entry with proper effort routing", () => {
		const out = collapseVariants(
			[
				devinMemberSpec("glm-5-2-1m", { contextWindow: 1_000_000 }),
				devinMemberSpec("glm-5-2-max-1m", { contextWindow: 1_000_000 }),
				devinMemberSpec("glm-5-2-none-1m", { contextWindow: 1_000_000, reasoning: false }),
			],
			{ table: devinTable },
		);

		expect(out).toHaveLength(1);
		const spec = out[0];
		expect(spec?.id).toBe("glm-5-2-1m");
		expect(spec?.contextWindow).toBe(1_000_000);
		expect(spec?.thinking?.effortRouting).toEqual({
			off: "glm-5-2-none-1m",
			high: "glm-5-2-1m",
			xhigh: "glm-5-2-max-1m",
		});
	});
});
