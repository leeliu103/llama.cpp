#include "hip-aot-loader.h"

#include "llama-impl.h"

#include <cstddef>

llama_hip_aot_loader::llama_hip_aot_loader(
        const char *            root,
        const hipDeviceProp_t & properties) :
    directory_(root) {
    std::string architecture = properties.gcnArchName;

    const size_t feature_separator = architecture.find(':');
    if (feature_separator != std::string::npos) {
        architecture.resize(feature_separator);
    }

    if (!directory_.empty() && directory_.back() != '/') {
        directory_ += '/';
    }
    directory_ += architecture;
}

bool llama_hip_aot_loader::load(
        hipModule_t * module,
        const char *  file_name) const {
    const std::string path = directory_ + "/" + file_name;
    const hipError_t error = hipModuleLoad(module, path.c_str());

    if (error == hipSuccess) {
        return true;
    }

    LLAMA_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path.c_str(), hipGetErrorString(error));
    return false;
}
