// Reference oracle: dequantize pinned GGUF tensors via llama.cpp ggml.
// Harness-only. Never linked into src/ or the shipped package.
//
// Usage:
//   dump-dequant <gguf> <outdir>
// Writes:
//   outdir/meta.json          metadata + type census (as-read)
//   outdir/tensors/<name>.f32 little-endian f32 dump
//   outdir/tensors/<name>.wire raw quantized bytes
//   outdir/kernels/<id>.{wire,f32,type}  synthetic Q8_0 / F16 + first blocks

#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void die(const char * msg) {
    std::fprintf(stderr, "dump-dequant: %s\n", msg);
    std::exit(1);
}

static void write_bytes(const fs::path & p, const void * data, size_t n) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    if (!out) die(("cannot write " + p.string()).c_str());
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n));
}

static const char * type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        default:             return ggml_type_name(t);
    }
}

static void json_escape(FILE * f, const char * s) {
    fputc('"', f);
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(s); *p; ++p) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', f);
            fputc(*p, f);
        } else if (*p < 0x20) {
            std::fprintf(f, "\\u%04x", *p);
        } else {
            fputc(*p, f);
        }
    }
    fputc('"', f);
}

static void dump_kv(FILE * f, const gguf_context * gguf, int64_t id) {
    const char * key = gguf_get_key(gguf, id);
    json_escape(f, key);
    fputc(':', f);
    const gguf_type ty = gguf_get_kv_type(gguf, id);
    switch (ty) {
        case GGUF_TYPE_UINT8:  std::fprintf(f, "%u", gguf_get_val_u8(gguf, id)); break;
        case GGUF_TYPE_INT8:   std::fprintf(f, "%d", gguf_get_val_i8(gguf, id)); break;
        case GGUF_TYPE_UINT16: std::fprintf(f, "%u", gguf_get_val_u16(gguf, id)); break;
        case GGUF_TYPE_INT16:  std::fprintf(f, "%d", gguf_get_val_i16(gguf, id)); break;
        case GGUF_TYPE_UINT32: std::fprintf(f, "%u", gguf_get_val_u32(gguf, id)); break;
        case GGUF_TYPE_INT32:  std::fprintf(f, "%d", gguf_get_val_i32(gguf, id)); break;
        case GGUF_TYPE_UINT64: std::fprintf(f, "%llu", (unsigned long long)gguf_get_val_u64(gguf, id)); break;
        case GGUF_TYPE_INT64:  std::fprintf(f, "%lld", (long long)gguf_get_val_i64(gguf, id)); break;
        case GGUF_TYPE_FLOAT32: std::fprintf(f, "%.9g", gguf_get_val_f32(gguf, id)); break;
        case GGUF_TYPE_FLOAT64: std::fprintf(f, "%.17g", gguf_get_val_f64(gguf, id)); break;
        case GGUF_TYPE_BOOL:   std::fputs(gguf_get_val_bool(gguf, id) ? "true" : "false", f); break;
        case GGUF_TYPE_STRING: json_escape(f, gguf_get_val_str(gguf, id)); break;
        case GGUF_TYPE_ARRAY: {
            const size_t n = gguf_get_arr_n(gguf, id);
            std::fprintf(f, "{\"type\":\"array\",\"n\":%zu}", n);
            break;
        }
        default:
            json_escape(f, "unknown");
            break;
    }
}

static void emit_kernel(const fs::path & dir, const char * id, ggml_type ty, const void * wire, size_t nbytes, const float * vals, size_t n) {
    write_bytes(dir / (std::string(id) + ".wire"), wire, nbytes);
    write_bytes(dir / (std::string(id) + ".f32"), vals, n * sizeof(float));
    write_bytes(dir / (std::string(id) + ".type"), type_name(ty), std::strlen(type_name(ty)));
    char nbuf[64];
    std::snprintf(nbuf, sizeof(nbuf), "%zu", n);
    write_bytes(dir / (std::string(id) + ".n"), nbuf, std::strlen(nbuf));
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dump-dequant <gguf> <outdir>\n");
        return 2;
    }
    const char * gguf_path = argv[1];
    const fs::path outdir = argv[2];
    fs::create_directories(outdir / "tensors");
    fs::create_directories(outdir / "kernels");

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = { /*no_alloc=*/false, /*ctx=*/&ctx };
    struct gguf_context * gguf = gguf_init_from_file(gguf_path, params);
    if (!gguf || !ctx) die("gguf_init_from_file failed");

    FILE * meta = std::fopen((outdir / "meta.json").c_str(), "w");
    if (!meta) die("cannot write meta.json");
    std::fputs("{\n  \"schema\": \"milton.dequant.dump/1\",\n  \"metadata\": {\n", meta);
    const int64_t n_kv = gguf_get_n_kv(gguf);
    int emitted = 0;
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(gguf, i);
        // skip huge tokenizer arrays in the kv dump; they are not dequant facts
        if (std::strncmp(key, "tokenizer.ggml.tokens", 21) == 0) continue;
        if (std::strncmp(key, "tokenizer.ggml.scores", 21) == 0) continue;
        if (std::strncmp(key, "tokenizer.ggml.token_type", 25) == 0) continue;
        if (emitted++) std::fputs(",\n", meta);
        std::fputs("    ", meta);
        dump_kv(meta, gguf, i);
    }
    std::fputs("\n  },\n  \"quant_types_present\": {\n", meta);

    int counts[GGML_TYPE_COUNT] = {};
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        ggml_type t = gguf_get_tensor_type(gguf, i);
        if (t >= 0 && t < GGML_TYPE_COUNT) counts[t]++;
    }
    emitted = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (!counts[t]) continue;
        if (emitted++) std::fputs(",\n", meta);
        std::fprintf(meta, "    \"%s\": %d", type_name(static_cast<ggml_type>(t)), counts[t]);
    }
    std::fputs("\n  },\n  \"tensors\": [\n", meta);

    static const char * kSelected[] = {
        "token_embd_norm.weight",
        "token_embd_norm.bias",
        "blk.0.attn_output_norm.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ffn_down.weight",
        nullptr,
    };

    emitted = 0;
    float global_r2r = 0.f;
    bool emitted_kernel_type[GGML_TYPE_COUNT] = {};
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        const char * name = ggml_get_name(t);
        if (!name || !*name) continue;
        bool selected = false;
        for (int i = 0; kSelected[i]; ++i) {
            if (std::strcmp(name, kSelected[i]) == 0) selected = true;
        }
        const int64_t ne = ggml_nelements(t);
        if (gguf_find_tensor(gguf, name) < 0) {
            continue; // internal ggml bookkeeping tensors (e.g. data blob)
        }
        const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (!traits) {
            std::fprintf(stderr, "fail-closed: no type traits for %s type %s\n", name, type_name(t->type));
            return 1;
        }
        if (!traits->to_float && t->type != GGML_TYPE_F32) {
            std::fprintf(stderr, "fail-closed: no to_float for %s type %s\n", name, type_name(t->type));
            return 1;
        }
        std::string fname = name;
        for (char & c : fname) if (c == '/' || c == '\\') c = '_';

        const bool want_kernel = (t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K
                                  || t->type == GGML_TYPE_Q6_K)
                                 && t->type < GGML_TYPE_COUNT && !emitted_kernel_type[t->type];
        if (!selected && !want_kernel) {
            continue;
        }

        const int64_t dequant_n = selected ? ne : traits->blck_size;
        if (dequant_n <= 0 || dequant_n > ne) continue;
        std::vector<float> dst(static_cast<size_t>(dequant_n));
        std::vector<float> dst_b(static_cast<size_t>(dequant_n));
        if (t->type == GGML_TYPE_F32) {
            std::memcpy(dst.data(), t->data, static_cast<size_t>(dequant_n) * sizeof(float));
            std::memcpy(dst_b.data(), t->data, static_cast<size_t>(dequant_n) * sizeof(float));
        } else {
            traits->to_float(t->data, dst.data(), dequant_n);
            traits->to_float(t->data, dst_b.data(), dequant_n);
        }
        float max_abs = 0.f;
        for (int64_t i = 0; i < dequant_n; ++i) {
            float d = dst[i] - dst_b[i];
            if (d < 0) d = -d;
            if (d > max_abs) max_abs = d;
        }
        if (max_abs > global_r2r) global_r2r = max_abs;

        if (selected) {
            write_bytes(outdir / "tensors" / (fname + ".f32"), dst.data(), dst.size() * sizeof(float));
            write_bytes(outdir / "tensors" / (fname + ".wire"), t->data, ggml_nbytes(t));
            if (emitted++) std::fputs(",\n", meta);
            std::fprintf(meta, "    {\"name\":");
            json_escape(meta, name);
            std::fprintf(meta, ",\"type\":\"%s\",\"n_elements\":%lld,\"n_bytes\":%zu,\"shape\":[",
                         type_name(t->type), (long long)ne, ggml_nbytes(t));
            const int nd = ggml_n_dims(t);
            for (int d = 0; d < nd; ++d) {
                if (d) std::fputc(',', meta);
                std::fprintf(meta, "%lld", (long long)t->ne[d]);
            }
            std::fputs("]}", meta);
        }

        if (want_kernel) {
            const size_t blk = static_cast<size_t>(traits->type_size);
            const int64_t blk_ne = traits->blck_size;
            if (ggml_nbytes(t) >= blk && ne >= blk_ne) {
                std::string kid = std::string("file-") + fname + "-block0";
                emit_kernel(outdir / "kernels", kid.c_str(), t->type, t->data, blk, dst.data(), static_cast<size_t>(blk_ne));
                emitted_kernel_type[t->type] = true;
            }
        }
    }
    std::fputs("\n  ],\n", meta);
    std::fprintf(meta, "  \"run_to_run_max_abs\": %.9g\n}\n", global_r2r);
    std::fclose(meta);

    // Synthetic Q8_0 + F16 via llama.cpp from_float_ref / to_float so those
    // kernels (required by the issue, absent from this Q4_K_M file) still
    // have an oracle-pinned fixture.
    {
        const struct ggml_type_traits * q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);
        const struct ggml_type_traits * f16 = ggml_get_type_traits(GGML_TYPE_F16);
        if (!q8 || !q8->from_float_ref || !q8->to_float) die("Q8_0 traits missing");
        if (!f16 || !f16->from_float_ref || !f16->to_float) die("F16 traits missing");

        float src[32];
        for (int i = 0; i < 32; ++i) {
            src[i] = (i - 15.5f) * 0.17f;
        }
        std::vector<uint8_t> q8wire(static_cast<size_t>(q8->type_size));
        q8->from_float_ref(src, q8wire.data(), 32);
        float q8out[32];
        q8->to_float(q8wire.data(), q8out, 32);
        emit_kernel(outdir / "kernels", "synth-q8_0-ramp32", GGML_TYPE_Q8_0, q8wire.data(), q8wire.size(), q8out, 32);

        std::vector<uint8_t> f16wire(32 * static_cast<size_t>(f16->type_size));
        f16->from_float_ref(src, f16wire.data(), 32);
        float f16out[32];
        f16->to_float(f16wire.data(), f16out, 32);
        emit_kernel(outdir / "kernels", "synth-f16-ramp32", GGML_TYPE_F16, f16wire.data(), f16wire.size(), f16out, 32);

        // second Q8_0 / F16 pass for run-to-run epsilon derivation
        float q8out_b[32];
        q8->to_float(q8wire.data(), q8out_b, 32);
        write_bytes(outdir / "kernels" / "synth-q8_0-ramp32.f32.b", q8out_b, sizeof(q8out_b));
        float f16out_b[32];
        f16->to_float(f16wire.data(), f16out_b, 32);
        write_bytes(outdir / "kernels" / "synth-f16-ramp32.f32.b", f16out_b, sizeof(f16out_b));
    }

    gguf_free(gguf);
    ggml_free(ctx);
    std::fprintf(stderr, "dump-dequant: wrote %s\n", outdir.c_str());
    return 0;
}
