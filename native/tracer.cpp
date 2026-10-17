#include "llama.h"
#include "llama-model.h"
#include "chat.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
static constexpr const char * revision = "9d1878a4354e743529fbe09e7350a957f5e39e9a";
static void emit(const json & value) { std::cout << value.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl; }
static double rounded(double x) { return std::isfinite(x) ? std::round(x * 1e8) / 1e8 : 0; }
static void progress(const std::string & stage, int step = -1) { emit({{"type", "progress"}, {"stage", stage}, {"step", step}}); }
static std::string piece(const llama_vocab * vocab, llama_token id) {
    std::vector<char> out(128);
    int n = llama_token_to_piece(vocab, id, out.data(), out.size(), 0, true);
    if (n < 0) { out.resize(-n); n = llama_token_to_piece(vocab, id, out.data(), out.size(), 0, true); }
    return n > 0 ? std::string(out.data(), n) : "";
}
static std::vector<float> read_row(ggml_tensor * t, int64_t query, int64_t head = 0, int64_t length = -1) {
    if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float)) return {};
    const int64_t n = length < 0 ? t->ne[0] : std::min(length, t->ne[0]);
    std::vector<float> row(n);
    ggml_backend_tensor_get(t, row.data(), query * t->nb[1] + head * t->nb[2], n * sizeof(float));
    for (float v : row) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite values in the model trace.");
    return row;
}
struct capture {
    int context_length = 0;
    std::map<int, std::vector<float>> hidden;
    std::map<int, std::vector<std::vector<float>>> attention;
    std::string error;
};
static int layer_number(const std::string & name, const std::string & prefix) {
    if (name.compare(0, prefix.size(), prefix) != 0) return -1;
    const std::string suffix = name.substr(prefix.size());
    if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](char c) { return c >= '0' && c <= '9'; })) return -1;
    return std::stoi(suffix);
}
static bool callback(ggml_tensor * t, bool ask, void * user) {
    auto & data = *static_cast<capture *>(user);
    const std::string name = t->name;
    const int hidden_layer = layer_number(name, "l_out-");
    const int attn_layer = layer_number(name, "kq_soft_max-");
    if (ask) return t->type == GGML_TYPE_F32 && (hidden_layer >= 0 || attn_layer >= 0);
    try {
        if (hidden_layer >= 0) data.hidden[hidden_layer] = read_row(t, t->ne[1] - 1);
        if (attn_layer >= 0) {
            auto & heads = data.attention[attn_layer];
            heads.clear();
            for (int64_t h = 0; h < t->ne[2]; h++) heads.push_back(read_row(t, t->ne[1] - 1, h, data.context_length));
        }
    } catch (const std::exception & e) { data.error = e.what(); return false; }
    return true;
}
static json candidates(const float * logits, int size, const llama_vocab * vocab, int count = 8) {
    const float maximum = *std::max_element(logits, logits + size);
    double sum = 0;
    for (int i = 0; i < size; i++) sum += std::exp(double(logits[i] - maximum));
    std::vector<int> ids(size);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + std::min(count, size), ids.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    json result = json::array();
    for (int i = 0; i < std::min(count, size); i++) {
        const int id = ids[i];
        result.push_back({{"id", id}, {"text", piece(vocab, id)}, {"probability", rounded(std::exp(double(logits[id] - maximum)) / sum)}});
    }
    return result;
}
// Apply the model's own final RMS normalization and output head to each recorded
// residual state. These are untuned logit-lens projections, not early decisions.
static json lens(llama_model * model, const capture & data, const llama_vocab * vocab, int threads, const float * final_logits) {
    const bool supported = model->arch == LLM_ARCH_LLAMA || model->arch == LLM_ARCH_GEMMA || model->arch == LLM_ARCH_GEMMA2 || model->arch == LLM_ARCH_GEMMA3 || model->arch == LLM_ARCH_GEMMA4 || model->arch == LLM_ARCH_QWEN2 || model->arch == LLM_ARCH_QWEN3;
    if (!supported || !model->output || !model->output_norm || model->output_s || model->output_in_s || data.hidden.empty()) return {{"available", false}, {"layers", json::array()}, {"reason", "This architecture's final readout is not supported by the logit-lens adapter."}};
    const int width = model->output->ne[0], n_vocab = model->output->ne[1], depth = data.hidden.size();
    std::vector<float> states;
    for (const auto & [layer, hidden] : data.hidden) {
        if (hidden.size() != size_t(width)) return {{"available", false}, {"layers", json::array()}, {"reason", "Hidden-state width does not match the output head."}};
        states.insert(states.end(), hidden.begin(), hidden.end());
    }
    ggml_init_params params = {ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, depth);
    ggml_tensor * norm = ggml_rms_norm(ctx, input, model->hparams.f_norm_rms_eps);
    norm = ggml_mul(ctx, norm, model->output_norm);
    if (model->output_norm_b) norm = ggml_add(ctx, norm, model->output_norm_b);
    ggml_tensor * output = ggml_mul_mat(ctx, model->output, norm);
    if (model->output_b) output = ggml_add(ctx, output, model->output_b);
    const float cap = model->hparams.f_final_logit_softcapping;
    if (cap > 0) output = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, output, 1.0f / cap)), cap);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_backend_free(backend); ggml_free(ctx); return {{"available", false}, {"layers", json::array()}, {"reason", "Could not allocate the diagnostic output graph."}}; }
    ggml_backend_tensor_set(input, states.data(), 0, states.size() * sizeof(float));
    const auto status = ggml_backend_graph_compute(backend, graph);
    json result = {{"available", status == GGML_STATUS_SUCCESS}, {"layers", json::array()}};
    if (status == GGML_STATUS_SUCCESS) {
        std::vector<float> row(n_vocab);
        int column = 0;
        for (const auto & [layer, hidden] : data.hidden) {
            ggml_backend_tensor_get(output, row.data(), column * output->nb[1], n_vocab * sizeof(float));
            result["layers"].push_back({{"layer", layer}, {"candidates", candidates(row.data(), n_vocab, vocab, 5)}});
            if (column == depth - 1) {
                double error = 0;
                for (int v = 0; v < n_vocab; v++) error = std::max(error, std::abs(double(row[v] - final_logits[v])));
                result["finalReadoutMaxLogitError"] = rounded(error);
            }
            column++;
        }
    } else result["reason"] = "The diagnostic output graph failed.";
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return result;
}
static json activation(const std::vector<float> & values, const std::vector<float> * previous) {
    double sum = 0, squares = 0, dot = 0, previous_squares = 0;
    std::vector<double> raw, bins;
    for (size_t i = 0; i < values.size(); i++) {
        sum += values[i]; squares += double(values[i]) * values[i]; raw.push_back(rounded(values[i]));
        if (previous && previous->size() == values.size()) { dot += double(values[i]) * (*previous)[i]; previous_squares += double((*previous)[i]) * (*previous)[i]; }
    }
    const int count = std::min<int>(64, values.size());
    for (int b = 0; b < count; b++) {
        const size_t start = b * values.size() / count, end = (b + 1) * values.size() / count;
        double value = 0; for (size_t i = start; i < end; i++) value += values[i];
        bins.push_back(rounded(value / (end - start)));
    }
    return {{"values", raw}, {"bins", bins}, {"rms", rounded(std::sqrt(squares / values.size()))}, {"mean", rounded(sum / values.size())}, {"min", *std::min_element(values.begin(), values.end())}, {"max", *std::max_element(values.begin(), values.end())}, {"cosineToPrevious", previous_squares > 0 && squares > 0 ? json(rounded(dot / std::sqrt(squares * previous_squares))) : json(nullptr)}};
}
int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") { emit({{"engine", "llama.cpp"}, {"revision", revision}, {"protocol", 1}}); return 0; }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    try {
        json request; std::cin >> request;
        const std::string path = request.at("modelPath"), prompt = request.at("prompt");
        const int steps = request.value("maxTokens", 4);
        if (steps < 1 || steps > 8 || prompt.empty() || prompt.size() > 5000) throw std::runtime_error("Trace request exceeds its limits.");
        const int threads = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
        const auto start = std::chrono::steady_clock::now();
        llama_log_set([](ggml_log_level level, const char * text, void *) { if (level == GGML_LOG_LEVEL_ERROR) std::cerr << text; }, nullptr);
        llama_backend_init();
        progress("Loading your existing GGUF weights");
        auto model_params = llama_model_default_params(); model_params.n_gpu_layers = 0;
        model = llama_model_load_from_file(path.c_str(), model_params);
        if (!model) throw std::runtime_error("llama.cpp could not load this GGUF model. Check model compatibility and available memory.");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        std::string override_template, template_source = "GGUF embedded chat template";
        if (!llama_model_chat_template(model, nullptr)) {
            if (model->arch != LLM_ARCH_GEMMA4) throw std::runtime_error("This GGUF has no embedded chat template, and its architecture has no tracing template adapter yet.");
            if (model->hparams.n_embd_per_layer > 0) {
                // E2B/E4B use a different non-thinking prefix from 12B/26B/31B.
                // https://ai.google.dev/gemma/docs/capabilities/thinking
                // This adapter handles our single user message, text-only protocol.
                override_template = R"jinja({{ bos_token }}{{ '<|turn>user\n' }}{{ messages[0]['content'] | trim }}{{ '<turn|>\n<|turn>model\n' }})jinja";
                template_source = "Gemma 4 E2B/E4B documented single-turn text format";
            } else {
                std::ifstream file(std::string(TOKENSCOPE_LLAMA_SOURCE_DIR) + "/models/templates/google-gemma-4-31B-it.jinja");
                if (!file) throw std::runtime_error("The canonical Gemma 4 template is missing. Run npm run setup:tracer again.");
                override_template.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
                template_source = "Canonical Gemma 4 template from pinned llama.cpp source";
            }
        }
        auto templates = common_chat_templates_init(model, override_template);
        common_chat_templates_inputs chat;
        common_chat_msg message; message.role = "user"; message.content = prompt;
        chat.messages = {message}; chat.enable_thinking = false;
        const std::string formatted = common_chat_templates_apply(templates.get(), chat).prompt.str();
        const std::string bos = piece(vocab, llama_vocab_bos(vocab));
        const bool add_special = bos.empty() || formatted.compare(0, bos.size(), bos) != 0;
        int n = -llama_tokenize(vocab, formatted.data(), formatted.size(), nullptr, 0, add_special, true);
        if (n <= 0 || n > 256) throw std::runtime_error("Keep the question shorter: the formatted prompt must fit within 256 tokens.");
        std::vector<llama_token> tokens(n);
        llama_tokenize(vocab, formatted.data(), formatted.size(), tokens.data(), tokens.size(), add_special, true);
        json context_tokens = json::array();
        for (int i = 0; i < n; i++) context_tokens.push_back({{"id", tokens[i]}, {"text", piece(vocab, tokens[i])}, {"position", i}, {"source", "prompt"}});
        capture data;
        auto context_params = llama_context_default_params();
        context_params.n_ctx = 512; context_params.n_batch = 512; context_params.n_ubatch = 512;
        context_params.n_threads = threads; context_params.n_threads_batch = threads;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.cb_eval = callback; context_params.cb_eval_user_data = &data;
        ctx = llama_init_from_model(model, context_params);
        if (!ctx) throw std::runtime_error("Could not create the tracing context.");
        json result = {{"engine", "llama.cpp"}, {"revision", revision}, {"mode", "separate-instrumented-run"}, {"sampling", "greedy"}, {"layerCount", llama_model_n_layer(model)}, {"hiddenSize", llama_model_n_embd(model)}, {"promptTokenCount", n}, {"steps", json::array()}, {"text", ""}};
        result["templateSource"] = template_source;
        int position = n;
        auto batch = llama_batch_get_one(tokens.data(), tokens.size());
        std::string text;
        for (int step = 0; step < steps; step++) {
            progress("Measuring attention and activations", step);
            data.context_length = position; data.hidden.clear(); data.attention.clear(); data.error.clear();
            if (llama_decode(ctx, batch) != 0 || !data.error.empty()) throw std::runtime_error(data.error.empty() ? "The tracing evaluation failed." : data.error);
            if (data.hidden.size() != size_t(llama_model_n_layer(model))) throw std::runtime_error("This model did not expose all layer outputs through the evaluation callback.");
            const float * logits = llama_get_logits_ith(ctx, -1);
            const int n_vocab = llama_vocab_n_tokens(vocab);
            const auto chosen = std::max_element(logits, logits + n_vocab) - logits;
            json top = candidates(logits, n_vocab, vocab);
            progress("Projecting intermediate token predictions", step);
            json projected = lens(model, data, vocab, threads, logits);
            json layers = json::array();
            const std::vector<float> * previous = nullptr;
            for (const auto & [layer, hidden] : data.hidden) {
                json heads = json::array();
                if (data.attention.count(layer)) for (const auto & head : data.attention.at(layer)) { std::vector<double> row; for (float p : head) row.push_back(rounded(p)); heads.push_back(row); }
                layers.push_back({{"index", layer}, {"activation", activation(hidden, previous)}, {"attention", heads}});
                previous = &hidden;
            }
            const bool stop = llama_vocab_is_eog(vocab, chosen);
            const std::string token_text = piece(vocab, chosen);
            result["steps"].push_back({{"index", step}, {"contextLength", position}, {"token", {{"id", chosen}, {"text", token_text}, {"probability", top[0]["probability"]}, {"isStop", stop}}}, {"candidates", top}, {"layers", layers}, {"lens", projected}});
            if (stop) { result["finishReason"] = "stop"; break; }
            text += token_text;
            context_tokens.push_back({{"id", chosen}, {"text", token_text}, {"position", position}, {"source", "generated"}});
            tokens = {llama_token(chosen)}; batch = llama_batch_get_one(tokens.data(), 1); position++;
        }
        result["text"] = text; result["contextTokens"] = context_tokens;
        if (!result.contains("finishReason")) result["finishReason"] = "limit";
        result["duration"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        emit({{"type", "result"}, {"data", result}});
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        emit({{"type", "error"}, {"error", e.what()}});
        if (ctx) llama_free(ctx); if (model) llama_model_free(model); llama_backend_free(); return 1;
    }
}
