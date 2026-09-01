target_sources(llama PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-aot-loader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-aot-loader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-aql.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-aql.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-workspace.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama-execution-extension.cpp"
)

find_package(hsa-runtime64 REQUIRED)

target_link_libraries(llama PRIVATE hip::host hsa-runtime64::hsa-runtime64)

add_subdirectory(gptoss)
