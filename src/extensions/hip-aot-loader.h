#pragma once

#include <hip/hip_runtime_api.h>

#include <string>

class llama_hip_aot_loader {
public:
    llama_hip_aot_loader(const char * root, const hipDeviceProp_t & properties);

    bool load(hipModule_t * module, const char * file_name) const;

private:
    std::string directory_;
};
