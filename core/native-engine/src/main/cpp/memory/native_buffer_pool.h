#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace nox::native {

class NativeBufferPool final {
public:
    struct Handle {
        std::uint32_t index{0};
        std::uint64_t generation{0};
        bool valid{false};
    };

    NativeBufferPool(std::uint32_t capacity, std::size_t block_size);
    NativeBufferPool(const NativeBufferPool&) = delete;
    NativeBufferPool& operator=(const NativeBufferPool&) = delete;

    std::int32_t acquire() noexcept;
    Handle acquire_handle() noexcept;
    void release(std::uint32_t index) noexcept;
    bool release(Handle handle) noexcept;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t active_blocks() const noexcept;
    [[nodiscard]] std::uint32_t free_blocks() const noexcept;
    [[nodiscard]] std::size_t block_size() const noexcept;
    [[nodiscard]] std::size_t resident_bytes() const noexcept;
    [[nodiscard]] std::uint64_t invalid_release_count() const noexcept;
    [[nodiscard]] std::uint64_t stale_release_count() const noexcept;

private:
    struct Slot {
        std::unique_ptr<std::byte[]> bytes;
        bool active{false};
        std::uint64_t generation{0};
    };

    mutable std::mutex mutex_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_indices_;
    std::size_t block_size_{0};
    std::uint32_t active_count_{0};
    std::uint64_t invalid_release_count_{0};
    std::uint64_t stale_release_count_{0};
};

} // namespace nox::native
