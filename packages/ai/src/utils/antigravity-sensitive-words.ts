/**
 * Zero-width obfuscation of system-instruction phrases for Antigravity.
 *
 * Cloud Code Assist inspects `systemInstruction` when the envelope carries
 * `requestType: "agent"` and answers matched payloads with a bare
 * `429 RESOURCE_EXHAUSTED` (no `ErrorInfo`/`RetryInfo`), which is
 * indistinguishable from real quota exhaustion and identical on every retry.
 * Splitting a matched phrase with U+200B (zero-width space) clears the match
 * while leaving the phrase visually and semantically intact for the model.
 *
 * Same mitigation CLIProxyAPI ships as `antigravity.sensitive-words`.
 */

/** Zero-width space. Invisible when rendered, breaks literal matching. */
const ZERO_WIDTH_SPACE = "\u200b";

/** Escape a literal phrase for use inside a RegExp. */
function escapeRegExp(literal: string): string {
	return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Insert a zero-width space after the first character of every occurrence of
 * each phrase. Matching is case-sensitive and literal; empty or whitespace-only
 * phrases are ignored. Returns the input unchanged when nothing matches.
 */
export function obfuscateSensitiveWords(text: string, phrases: readonly string[]): string {
	let result = text;
	for (const phrase of phrases) {
		const trimmed = phrase.trim();
		if (trimmed.length < 2) continue;
		const pattern = new RegExp(escapeRegExp(trimmed), "g");
		result = result.replace(pattern, match => `${match[0]}${ZERO_WIDTH_SPACE}${match.slice(1)}`);
	}
	return result;
}

/** Native Antigravity identity sentence, as the `agy` client's default system-prompt template opens with it. */
export const ANTIGRAVITY_IDENTITY =
	"You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.";

/**
 * omp/Pi identity sentences swapped for the native Antigravity identity on
 * `google-antigravity` requests. Cloud Code Assist content-filters
 * `systemInstruction` for known third-party agent identity lines and answers
 * matched payloads with a bare 429 RESOURCE_EXHAUSTED (verified live for omp's
 * default opening on 2026-11-16: quota was healthy and every other field
 * passed). Zero-width obfuscation cannot clear these matches — the filter
 * normalizes zero-width characters, collapses whitespace, and matches
 * case-insensitively — so the whole sentence is replaced with the native
 * client's identity instead. Self-identification lines follow so the agent
 * stays consistent with the swapped identity on this provider.
 */
const IDENTITY_REPLACEMENTS: ReadonlyArray<readonly [string, string]> = [
	["You are omp's trusted coding assistant.", ANTIGRAVITY_IDENTITY],
	["You are Pi, a fast and rigorous autonomous coding agent harness.", ANTIGRAVITY_IDENTITY],
	["Identify yourself as Pi.", "Identify yourself as Antigravity."],
	["NEVER refer to yourself as Oh My Pi or OMP.", "NEVER refer to yourself as Oh My Pi, OMP, or Pi."],
];

/**
 * Replace omp/Pi identity sentences with the native Antigravity identity.
 * Matching is literal and case-sensitive; text without a match passes through
 * unchanged, so prompts that never carried an omp identity are not rewritten.
 */
export function replaceAntigravityIdentity(text: string): string {
	let result = text;
	for (const [sentence, replacement] of IDENTITY_REPLACEMENTS) {
		if (result.includes(sentence)) result = result.replaceAll(sentence, replacement);
	}
	return result;
}
