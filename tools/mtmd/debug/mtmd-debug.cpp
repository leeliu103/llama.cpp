#include "mtmd-debug.h"

#include "arg.h"
#include "debug.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include <cinttypes>
#include <clocale>
#include <map>
#include <string>
#include <unordered_set>

// INTERNAL TOOL FOR DEBUGGING PURPOSES ONLY
// NOT INTENDED FOR PUBLIC USE

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Internal debugging tool for mtmd; See mtmd-debug.md for the pytorch equivalent code\n"
        "Note: we repurpose some args from other examples, they will have different meaning here\n"
        "\n"
        "Usage: %s -m <model> --mmproj <mmproj> -p <mode> -n <size> --image <image> --audio <audio>\n"
        "\n"
        "    -n <size>: number of pixels per edge for image (always square image), or number of samples for audio\n"
        "\n"
        "    -p \"encode\" (debugging encode pass, default case):\n"
        "        --image can be:\n"
        "          \"white\", \"black\", \"gray\": filled 1.0f, 0.0f and 0.5f respectively\n"
        "          \"red\", \"green\", \"blue\": filled with respective colors\n"
        "          \"cb\": checkerboard pattern, alternate 1.0f and 0.0f\n"
        "          \"rainbow\": raspberry-pi-like rainbow pattern\n"
        "        --audio can be:\n"
        "          \"one\", \"zero\", \"half\": filled 1.0f, 0.0f and 0.5f respectively\n"
        "          \"1010\": checkerboard pattern, alternate 1.0f and 0.0f\n"
        "\n"
        "    -p \"preproc\" (debugging preprocessing pass):\n"
        "        --image can be:\n"
        "          \"white\", \"black\", \"gray\": filled image with respective colors\n"
        "          \"cb\": checkerboard pattern\n"
        "          or a path to an image file\n"
        "        --audio can be:\n"
        "          \"one\", \"zero\", \"half\": filled 1.0f, 0.0f and 0.5f respectively\n"
        "          \"440\": sine wave with 440 Hz frequency\n"
        "\n",
        argv[0]
    );
}

static std::string mtmd_debug_join_path(const std::string & dir, const std::string & name) {
    if (dir.empty() || dir.back() == '/' || dir.back() == '\\') {
        return dir + name;
    }
    return dir + "/" + name;
}

static std::vector<std::string> mtmd_debug_split_csv(const char * csv) {
    std::vector<std::string> names;
    if (!csv || csv[0] == '\0') {
        return names;
    }

    std::string cur;
    for (const char * p = csv; *p; ++p) {
        if (*p == ',') {
            if (!cur.empty()) {
                names.push_back(cur);
                cur.clear();
            }
        } else if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
            cur.push_back(*p);
        }
    }
    if (!cur.empty()) {
        names.push_back(cur);
    }
    return names;
}

static std::string mtmd_debug_safe_file_stem(const std::string & name) {
    std::string stem = name;
    for (char & ch : stem) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
        if (!ok) {
            ch = '_';
        }
    }
    return stem;
}

struct phi4mm_layer_dump_cb_data {
    std::string dump_dir;
    std::unordered_set<std::string> wanted;
    std::unordered_set<std::string> dumped;
    std::map<std::string, std::string> manifest_entries;
    int64_t n_patches = 1024;
};

static std::string mtmd_debug_shape_json(const std::vector<int64_t> & shape) {
    std::string shape_json = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            shape_json += ", ";
        }
        shape_json += std::to_string(shape[i]);
    }
    shape_json += "]";
    return shape_json;
}

static void phi4mm_layer_dump_write_manifest(phi4mm_layer_dump_cb_data * cb_data) {
    const std::string manifest_path = mtmd_debug_join_path(cb_data->dump_dir, "phi4mm_layer_dumps.json");
    std::ofstream manifest(manifest_path);
    if (!manifest) {
        LOG_ERR("%s: failed to open %s for writing\n", __func__, manifest_path.c_str());
        return;
    }

    manifest << "{\n  \"tensors\": {\n";
    size_t i = 0;
    for (const auto & entry : cb_data->manifest_entries) {
        manifest << "    \"" << entry.first << "\": " << entry.second;
        manifest << (++i < cb_data->manifest_entries.size() ? "," : "") << "\n";
    }
    manifest << "  }\n}\n";
}

static bool phi4mm_layer_dump_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = static_cast<phi4mm_layer_dump_cb_data *>(user_data);
    const std::string name(t->name);

    if (cb_data->wanted.find(name) == cb_data->wanted.end() ||
            cb_data->dumped.find(name) != cb_data->dumped.end()) {
        return !ask;
    }

    if (ask) {
        return true;
    }

    cb_data->dumped.insert(name);
    std::vector<int64_t> shape;
    const int n_dims = ggml_n_dims(t);
    if (n_dims == 2 && cb_data->n_patches > 0 && t->ne[1] % cb_data->n_patches == 0) {
        shape = { t->ne[1] / cb_data->n_patches, cb_data->n_patches, t->ne[0] };
    } else if (n_dims == 3) {
        shape = { t->ne[2], t->ne[1], t->ne[0] };
    } else if (n_dims == 4) {
        shape = { t->ne[3], t->ne[2], t->ne[1], t->ne[0] };
    } else {
        shape.assign(t->ne, t->ne + n_dims);
    }

    const std::string file_name = mtmd_debug_safe_file_stem(name) + ".f32";
    const std::string data_path = mtmd_debug_join_path(cb_data->dump_dir, file_name);
    std::vector<float> data(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ggml_nelements(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(tmp.data(), data.data(), tmp.size());
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> tmp(ggml_nelements(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
        ggml_bf16_to_fp32_row(tmp.data(), data.data(), tmp.size());
    } else {
        LOG_WRN("%s: requested tensor %s is %s, only F32/F16/BF16 dumps are supported\n",
                __func__, name.c_str(), ggml_type_name(t->type));
        return true;
    }

    std::ofstream out(data_path, std::ios::binary);
    if (!out) {
        LOG_ERR("%s: failed to open %s for writing\n", __func__, data_path.c_str());
        return true;
    }
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    if (!out) {
        LOG_ERR("%s: failed to write %s\n", __func__, data_path.c_str());
        return true;
    }

    cb_data->manifest_entries[name] =
        "{\"file\": \"" + file_name + "\", \"dtype\": \"float32\", \"shape\": " +
        mtmd_debug_shape_json(shape) + "}";
    phi4mm_layer_dump_write_manifest(cb_data);

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    ggml_backend_load_all();

    LOG_INF("%s: loading model: %s\n", __func__, params.model.path.c_str());

    mtmd::context_ptr ctx_mtmd;
    common_init_result_ptr llama_init;
    common_debug_cb_user_data cb_data;
    phi4mm_layer_dump_cb_data phi4mm_layer_cb_data;

    llama_init = common_init_from_params(params);
    {
        auto * model = llama_init->model();
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        const char * phi4mm_layer_dump_dir = std::getenv("MTMD_DEBUG_PHI4MM_LAYER_DUMP");
        if (phi4mm_layer_dump_dir != nullptr && phi4mm_layer_dump_dir[0] != '\0') {
            phi4mm_layer_cb_data.dump_dir = phi4mm_layer_dump_dir;
            for (const std::string & name : mtmd_debug_split_csv(std::getenv("MTMD_DEBUG_PHI4MM_LAYER_DUMP_NAMES"))) {
                phi4mm_layer_cb_data.wanted.insert(name);
            }
            if (const char * n_patches = std::getenv("MTMD_DEBUG_PHI4MM_LAYER_DUMP_N_PATCHES")) {
                phi4mm_layer_cb_data.n_patches = std::strtoll(n_patches, nullptr, 10);
            }
            mparams.cb_eval_user_data = &phi4mm_layer_cb_data;
            mparams.cb_eval = phi4mm_layer_dump_cb_eval;
        } else if (std::getenv("MTMD_DEBUG_PHI4MM_HIDDEN_STATES_DUMP") == nullptr
                && std::getenv("MTMD_DEBUG_PHI4MM_PROJECTED_EMBEDDINGS_DUMP") == nullptr) {
            // always enable debug callback
            mparams.cb_eval_user_data = &cb_data;
            mparams.cb_eval = common_debug_cb_eval;
        }
        ctx_mtmd.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_mtmd.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }

    std::string input;
    int32_t inp_size = params.n_predict;
    if (params.image.empty()) {
        LOG_ERR("ERR: At least one of --image or --audio must be specified\n");
        return 1;
    }
    if (inp_size <= 0) {
        if (params.prompt.empty() || params.prompt == "encode") {
            LOG_ERR("ERR: Invalid size specified with -n, must be greater than 0\n");
            return 1;
        }
        inp_size = 1;
    }
    input = params.image[0];

    if (params.prompt.empty() || params.prompt == "encode") {
        std::vector<std::vector<float>> image;
        std::vector<float> samples;

        if (input == "black") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.0f);
                image.push_back(row);
            }
        } else if (input == "white") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 1.0f);
                image.push_back(row);
            }
        } else if (input == "gray") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.5f);
                image.push_back(row);
            }
        } else if (input == "cb") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.0f);
                image.push_back(row);
            }
            for (int y = 0; y < inp_size; ++y) {
                for (int x = 0; x < inp_size; ++x) {
                    float v = ((x + y) % 2) ? 0.0f : 1.0f;
                    image[y][x * 3 + 0] = v;
                    image[y][x * 3 + 1] = v;
                    image[y][x * 3 + 2] = v;
                }
            }
        } else if (input == "red") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.0f);
                for (int j = 0; j < inp_size; ++j) {
                    row[j * 3 + 0] = 1.0f;  // R channel
                }
                image.push_back(row);
            }
        } else if (input == "green") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.0f);
                for (int j = 0; j < inp_size; ++j) {
                    row[j * 3 + 1] = 1.0f;  // G channel
                }
                image.push_back(row);
            }
        } else if (input == "blue") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.0f);
                for (int j = 0; j < inp_size; ++j) {
                    row[j * 3 + 2] = 1.0f;  // B channel
                }
                image.push_back(row);
            }
        } else if (input == "rainbow") {
            for (int i = 0; i < inp_size; ++i) {
                image.push_back(std::vector<float>(inp_size * 3, 0.0f));
            }
            float cx = inp_size / 2.0f;
            float cy = inp_size / 2.0f;
            float max_dist = std::sqrt(cx * cx + cy * cy);
            for (int y = 0; y < inp_size; ++y) {
                for (int x = 0; x < inp_size; ++x) {
                    float dx = x - cx;
                    float dy = y - cy;
                    float hue = std::atan2(dy, dx) / (2.0f * 3.14159265f);
                    if (hue < 0) hue += 1.0f;
                    float sat = std::sqrt(dx * dx + dy * dy) / max_dist;
                    if (sat > 1.0f) sat = 1.0f;
                    float h6 = hue * 6.0f;
                    int i6 = (int)h6;
                    float f = h6 - i6;
                    float p = 1.0f - sat;
                    float q = 1.0f - sat * f;
                    float t = 1.0f - sat * (1.0f - f);
                    float r, g, b;
                    switch (i6 % 6) {
                        case 0: r=1; g=t; b=p; break;
                        case 1: r=q; g=1; b=p; break;
                        case 2: r=p; g=1; b=t; break;
                        case 3: r=p; g=q; b=1; break;
                        case 4: r=t; g=p; b=1; break;
                        default: r=1; g=p; b=q; break;
                    }
                    image[y][x * 3 + 0] = r;
                    image[y][x * 3 + 1] = g;
                    image[y][x * 3 + 2] = b;
                }
            }
        } else if (input == "one") {
            samples = std::vector<float>(inp_size, 1.0f);
        } else if (input == "zero") {
            samples = std::vector<float>(inp_size, 0.0f);
        } else if (input == "half") {
            samples = std::vector<float>(inp_size, 0.5f);
        } else if (input == "1010") {
            samples.resize(inp_size);
            for (int i = 0; i < inp_size; ++i) {
                samples[i] = (i % 2) ? 0.0f : 1.0f;
            }
        } else {
            LOG_ERR("ERR: Invalid input specified with --image/--audio\n");
            show_additional_info(argc, argv);
            return 1;
        }

        // run encode pass
        LOG_INF("Running encode pass for input type: %s\n", input.c_str());
        if (samples.size() > 0) {
            LOG_INF("Input audio with %zu samples, type: %s\n", samples.size(), input.c_str());
            mtmd_debug_encode_audio(ctx_mtmd.get(), samples);
        } else {
            LOG_INF("Input image with dimensions %d x %d, type: %s\n", inp_size, inp_size, input.c_str());
            mtmd_debug_encode_image(ctx_mtmd.get(), image);
        }

    } else if (params.prompt == "preproc") {
        std::vector<uint8_t> rgb_values;
        std::vector<float> pcm_samples;

        if (input == "black") {
            rgb_values = std::vector<uint8_t>(inp_size * inp_size * 3, 0);
        } else if (input == "white") {
            rgb_values = std::vector<uint8_t>(inp_size * inp_size * 3, 255);
        } else if (input == "gray") {
            rgb_values = std::vector<uint8_t>(inp_size * inp_size * 3, 128);
        } else if (input == "cb") {
            rgb_values.resize(inp_size * inp_size * 3);
            for (int y = 0; y < inp_size; ++y) {
                for (int x = 0; x < inp_size; ++x) {
                    uint8_t v = ((x + y) % 2) ? 0 : 255;
                    rgb_values[(y * inp_size + x) * 3 + 0] = v;
                    rgb_values[(y * inp_size + x) * 3 + 1] = v;
                    rgb_values[(y * inp_size + x) * 3 + 2] = v;
                }
            }
        } else if (input == "one") {
            pcm_samples = std::vector<float>(inp_size, 1.0f);
        } else if (input == "zero") {
            pcm_samples = std::vector<float>(inp_size, 0.0f);
        } else if (input == "half") {
            pcm_samples = std::vector<float>(inp_size, 0.5f);
        } else if (input == "440") {
            pcm_samples.resize(inp_size);
            float freq = 440.0f;
            float sample_rate = mtmd_get_audio_sample_rate(ctx_mtmd.get());
            float pi = 3.14159265f;
            for (int i = 0; i < inp_size; ++i) {
                pcm_samples[i] = sinf(2 * pi * freq * i / sample_rate);
            }
        } else {
            auto loaded = mtmd_helper_bitmap_init_from_file(ctx_mtmd.get(), input.c_str(), false);
            if (!loaded.bitmap || loaded.video_ctx || mtmd_bitmap_is_audio(loaded.bitmap)) {
                if (loaded.bitmap) {
                    mtmd_bitmap_free(loaded.bitmap);
                }
                if (loaded.video_ctx) {
                    mtmd_helper_video_free(loaded.video_ctx);
                }
                LOG_ERR("ERR: Invalid input specified with --image/--audio\n");
                show_additional_info(argc, argv);
                return 1;
            }

            const unsigned char * data = mtmd_bitmap_get_data(loaded.bitmap);
            const size_t n_bytes = mtmd_bitmap_get_n_bytes(loaded.bitmap);
            rgb_values.assign(data, data + n_bytes);
            inp_size = (int) mtmd_bitmap_get_nx(loaded.bitmap);
            const int inp_height = (int) mtmd_bitmap_get_ny(loaded.bitmap);
            mtmd_bitmap_free(loaded.bitmap);
            if (loaded.video_ctx) {
                mtmd_helper_video_free(loaded.video_ctx);
            }

            LOG_INF("Running preprocessing pass for input file: %s\n", input.c_str());
            LOG_INF("Input image with dimensions %d x %d\n", inp_size, inp_height);
            mtmd_debug_preprocess_image(ctx_mtmd.get(), rgb_values, inp_size, inp_height);
            return 0;
        }

        // run preprocessing pass
        LOG_INF("Running preprocessing pass for input type: %s\n", input.c_str());
        if (pcm_samples.size() > 0) {
            LOG_INF("Input audio with %zu samples, type: %s\n", pcm_samples.size(), input.c_str());
            mtmd_debug_preprocess_audio(ctx_mtmd.get(), pcm_samples);
        } else {
            LOG_INF("Input image with dimensions %d x %d, type: %s\n", inp_size, inp_size, input.c_str());
            mtmd_debug_preprocess_image(ctx_mtmd.get(), rgb_values, inp_size, inp_size);
        }

    } else {
        LOG_ERR("ERR: Invalid mode specified with -p\n");
        show_additional_info(argc, argv);
        return 1;
    }

    return 0;
}
