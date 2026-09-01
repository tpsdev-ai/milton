/**
 * Cases whose Q4 / F16 goldens must come from llama.cpp GGUF forward on
 * the exact HF token IDs in tokens.json — never from llama-embedding's
 * text tokenizer (accents → [UNK]) or `llama-embedding -f` (newline split).
 *
 * The reference stays independent of Milton: token IDs are the #7 HF pin;
 * the forward is the pinned llama.cpp commit in pin.json.
 */
export const TOKEN_ID_ORACLE_CASES = ["unicode-nfd", "newlines-tabs"];

export const TOKEN_ID_ORACLE_SOURCE = "harness/goldens/tokens.json";

export const TOKEN_ID_ORACLE_TOOL = "harness/tools/embed-from-token-ids.cpp";
