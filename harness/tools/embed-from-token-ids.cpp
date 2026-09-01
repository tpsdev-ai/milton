// Reference oracle: llama.cpp GGUF forward on EXACT token IDs.
// Harness-only. Never linked into src/ or the shipped package.
//
// Usage:
//   embed-from-token-ids <gguf> --ids 101,3945,...
//
// Runs llama_decode on the given token IDs (no text tokenizer),
// mean-pools (LLAMA_POOLING_TYPE_MEAN), L2-normalizes (embd-normalize 2),
// and writes one JSON object to stdout. This is how goldens are produced
// when llama.cpp's *text* tokenizer is the wrong oracle (HF token IDs
// already pinned in harness/goldens/tokens.json).

#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die(const char * msg) {
    std::fprintf(stderr, "embed-from-token-ids: %s\n", msg);
    std::exit(1);
}

static void quiet_log(enum ggml_log_level level, const char * text, void * /*ud*/) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

static std::vector<llama_token> parse_ids(const char * spec) {
    std::vector<llama_token> ids;
    const char * p = spec;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
        if (!*p) break;
        char * end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) die("invalid token-id list (expected comma-separated integers)");
        ids.push_back(static_cast<llama_token>(v));
        p = end;
    }
    if (ids.empty()) die("token-id list is empty");
    return ids;
}

static void l2_normalize(const float * in, float * out, int n) {
    // Match llama.cpp common_embd_normalize(..., embd_norm=2): Euclidean, then scale.
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += static_cast<double>(in[i]) * static_cast<double>(in[i]);
    sum = std::sqrt(sum);
    if (sum == 0.0) sum = 1.0;
    const float d = 1.0f / static_cast<float>(sum);
    for (int i = 0; i < n; i++) out[i] = in[i] * d;
}

int main(int argc, char ** argv) {
    const char * gguf = nullptr;
    const char * ids_spec = nullptr;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--ids") == 0) {
            if (i + 1 >= argc) die("--ids requires a comma-separated list");
            ids_spec = argv[++i];
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::fprintf(stderr, "usage: embed-from-token-ids <gguf> --ids 101,3945,...\n");
            return 2;
        } else if (argv[i][0] == '-') {
            die("unknown flag (expected <gguf> --ids <list>)");
        } else if (!gguf) {
            gguf = argv[i];
        } else {
            die("unexpected positional argument");
        }
    }
    if (!gguf || !ids_spec) {
        std::fprintf(stderr, "usage: embed-from-token-ids <gguf> --ids 101,3945,...\n");
        return 2;
    }

    const std::vector<llama_token> ids = parse_ids(ids_spec);
    const int n_tokens = static_cast<int>(ids.size());

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(gguf, model_params);
    if (!model) die("llama_model_load_from_file failed");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    ctx_params.n_ubatch = 2048; // encoder / non-causal: batch == ubatch
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    ctx_params.kv_unified = true;
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        die("llama_init_from_model failed");
    }

    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_MEAN) {
        llama_free(ctx);
        llama_model_free(model);
        die("fail-closed: pooling_type is not MEAN");
    }

    const int n_embd = llama_model_n_embd(model);
    if (n_embd <= 0) {
        llama_free(ctx);
        llama_model_free(model);
        die("fail-closed: n_embd <= 0");
    }
    if (n_tokens > static_cast<int>(ctx_params.n_batch)) {
        llama_free(ctx);
        llama_model_free(model);
        die("fail-closed: n_tokens exceeds n_batch");
    }

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = ids[static_cast<size_t>(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true; // embeddings for every token (mean pool)
    }

    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        std::fprintf(stderr, "embed-from-token-ids: llama_decode failed (%d)\n", rc);
        return 1;
    }

    const float * embd = llama_get_embeddings_seq(ctx, 0);
    if (!embd) {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        die("llama_get_embeddings_seq returned NULL");
    }

    std::vector<float> out(static_cast<size_t>(n_embd));
    l2_normalize(embd, out.data(), n_embd);

    // Provenance + vector on stdout. Logs stay on stderr (errors only).
    std::printf("{\n");
    std::printf("  \"schema\": \"milton.embed-from-token-ids/1\",\n");
    std::printf("  \"n_ids\": %d,\n", n_tokens);
    std::printf("  \"ids\": [");
    for (int i = 0; i < n_tokens; i++) {
        if (i) std::printf(",");
        std::printf("%d", static_cast<int>(ids[static_cast<size_t>(i)]));
    }
    std::printf("],\n");
    std::printf("  \"dims\": %d,\n", n_embd);
    std::printf("  \"pooling\": \"mean\",\n");
    std::printf("  \"embd_normalize\": 2,\n");
    std::printf("  \"embedding\": [");
    for (int i = 0; i < n_embd; i++) {
        if (i) std::printf(",");
        std::printf("%.9g", static_cast<double>(out[static_cast<size_t>(i)]));
    }
    std::printf("]\n}\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
