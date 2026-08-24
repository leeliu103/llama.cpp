set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-repack-hip.cpp"
    TARGET_DIRECTORY llama
    PROPERTIES LANGUAGE HIP
)

target_sources(llama PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/gptoss-repack-hip.cpp"
)
