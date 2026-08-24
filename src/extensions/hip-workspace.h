#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

class llama_hip_workspace {
public:
    llama_hip_workspace() = default;

    ~llama_hip_workspace() {
        (void) free();
    }

    llama_hip_workspace(const llama_hip_workspace &) = delete;
    llama_hip_workspace & operator=(const llama_hip_workspace &) = delete;

    void * data() {
        return data_;
    }

    // The caller must select the allocation device and synchronize users before storage is freed.
    hipError_t reserve(size_t size, size_t preferred_headroom = 0) {
        if (size <= capacity_) {
            return hipSuccess;
        }

        size_t allocation_size = size;
        if (preferred_headroom <= std::numeric_limits<size_t>::max() - size) {
            allocation_size += preferred_headroom;
        }

        hipError_t error = free();
        if (error != hipSuccess) {
            return error;
        }

        void * data = nullptr;
        error       = hipMalloc(&data, allocation_size);
        if (error == hipErrorOutOfMemory && allocation_size != size) {
            (void) hipGetLastError();
            data            = nullptr;
            allocation_size = size;
            error           = hipMalloc(&data, allocation_size);
        }

        if (error == hipSuccess) {
            data_     = data;
            capacity_ = allocation_size;
        }
        return error;
    }

    hipError_t free() {
        if (data_ == nullptr) {
            return hipSuccess;
        }

        hipError_t error = hipFree(data_);
        if (error == hipSuccess) {
            data_     = nullptr;
            capacity_ = 0;
        }
        return error;
    }

private:
    void * data_     = nullptr;
    size_t capacity_ = 0;
};

class llama_hip_workspace_cursor {
public:
    llama_hip_workspace_cursor() = default;

    llama_hip_workspace_cursor(void * data, size_t capacity) :
        data_(static_cast<uint8_t *>(data)), capacity_(capacity),
        valid_(capacity == 0 || (data != nullptr && reinterpret_cast<uintptr_t>(data) % alignment == 0)) {}

    template <typename T>
    T * take(size_t count) {
        if (!valid_ || count == 0) {
            return nullptr;
        }
        static_assert(alignof(T) <= alignment);

        constexpr size_t padding = alignment - 1;
        if (offset_ > std::numeric_limits<size_t>::max() - padding) {
            valid_ = false;
            return nullptr;
        }

        const size_t begin = (offset_ + padding) & ~padding;
        if (count > (std::numeric_limits<size_t>::max() - begin) / sizeof(T)) {
            valid_ = false;
            return nullptr;
        }

        const size_t end = begin + count * sizeof(T);
        if (end > capacity_) {
            valid_ = false;
            return nullptr;
        }

        T * result = data_ == nullptr ? nullptr : reinterpret_cast<T *>(data_ + begin);
        offset_    = end;
        return result;
    }

    size_t size() const {
        return offset_;
    }

    bool valid() const {
        return valid_;
    }

private:
    static constexpr size_t alignment = 256;

    uint8_t * data_     = nullptr;
    size_t    capacity_ = std::numeric_limits<size_t>::max();
    size_t    offset_   = 0;
    bool      valid_    = true;
};
