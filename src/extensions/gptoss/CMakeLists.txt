if (NOT CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR "CMAKE_HIP_ARCHITECTURES must be set for GPT-OSS AOT kernels")
endif()

set(LLAMA_GPTOSS_HIP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-kernel-hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-repack-hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/decode_megakernel_hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/embedding_q8_0_hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/lm_head_mmvq_hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/moe_combine_hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/ogs_build_routes_hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/qkv_rope_cache_hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/rms_norm_hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/topk_hip.cpp"
)

set_source_files_properties(
    ${LLAMA_GPTOSS_HIP_SOURCES}
    TARGET_DIRECTORY llama
    PROPERTIES LANGUAGE HIP
)

target_sources(llama PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-kernel-aot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-buffers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-extension.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-kv.cpp"
    ${LLAMA_GPTOSS_HIP_SOURCES}
)

target_link_libraries(llama PRIVATE roc::hipblas)

set(GPTOSS_AOT_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

target_compile_definitions(llama PRIVATE
    "GPTOSS_AOT_ROOT=\"${GPTOSS_AOT_ROOT}\""
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LLAMA_GPTOSS_AOT_SUPPORTED_ARCHITECTURES
    gfx1201
)

set(LLAMA_GPTOSS_AOT_ARCHITECTURES)
foreach(architecture IN LISTS CMAKE_HIP_ARCHITECTURES)
    string(REGEX REPLACE ":.*$" "" architecture_base "${architecture}")
    if (architecture_base IN_LIST LLAMA_GPTOSS_AOT_SUPPORTED_ARCHITECTURES)
        list(APPEND LLAMA_GPTOSS_AOT_ARCHITECTURES "${architecture_base}")
    endif()
endforeach()
list(REMOVE_DUPLICATES LLAMA_GPTOSS_AOT_ARCHITECTURES)

if (NOT LLAMA_GPTOSS_AOT_ARCHITECTURES)
    message(FATAL_ERROR "GPT-OSS AOT supports ${LLAMA_GPTOSS_AOT_SUPPORTED_ARCHITECTURES}, but CMAKE_HIP_ARCHITECTURES is ${CMAKE_HIP_ARCHITECTURES}")
endif()

set(LLAMA_GPTOSS_AOT_KERNELS
    q8_qkv.hsaco
    q8_attn_out.hsaco
    fa_full.hsaco
    fa_sw128.hsaco
    ogs_w13.hsaco
    ogs_w2.hsaco
)

set(LLAMA_GPTOSS_AOT_OUTPUTS)
foreach(architecture IN LISTS LLAMA_GPTOSS_AOT_ARCHITECTURES)
    set(gptoss_aot_output_dir "${GPTOSS_AOT_ROOT}/${architecture}")
    set(gptoss_aot_outputs)

    foreach(kernel IN LISTS LLAMA_GPTOSS_AOT_KERNELS)
        list(APPEND gptoss_aot_outputs "${gptoss_aot_output_dir}/${kernel}")
    endforeach()

    add_custom_command(
        OUTPUT ${gptoss_aot_outputs}
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-aot.py"
            --arch "${architecture}"
            --output-dir "${gptoss_aot_output_dir}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-aot.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernel/matmul_q8_0_w8a16.py"
        COMMENT "Generating GPT-OSS AOT kernels for ${architecture}"
        VERBATIM
    )

    list(APPEND LLAMA_GPTOSS_AOT_OUTPUTS ${gptoss_aot_outputs})
endforeach()

add_custom_target(llama-gptoss-aot
    DEPENDS ${LLAMA_GPTOSS_AOT_OUTPUTS}
)
add_dependencies(llama llama-gptoss-aot)
