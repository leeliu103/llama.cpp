target_sources(llama PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-aot-loader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-aot-loader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-workspace.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama-execution-extension.cpp"
)

target_link_libraries(llama PRIVATE hip::host)
