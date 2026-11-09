#include "memory/native_buffer_pool.h"

#include <algorithm>
#include <limits>

namespace nox::native {

NativeBufferPool::NativeBufferPool(
    std::uint32_t capacity,
    std::size_t block_size)
    : block_size_(std::max<std::size_t>(1u, block_size)) {
    const auto safe_capacity =
        std::max<std::uint32_t>(1u, capacity);

    slots_.reserve(safe_capacity);
    free_indices_.reserve(safe_capacity);

    for (std::uint32_t i = 0; i < safe_capacity; ++i) {
        slots_.push_back(
            Slot{
                std::make_unique<std::byte[]>(block_size_),
                false,
            });
        free_indices_.push_back(
            safe_capacity - 1u - i);
    }
}

std::int32_t NativeBufferPool::acquire() noexcept {
    const auto handle = acquire_handle();
    if(!handle.valid) {
        return -1;
    }
    return static_cast<std::int32_t>(handle.index);
}

NativeBufferPool::Handle NativeBufferPool::acquire_handle() noexcept {
    std::scoped_lock lock(mutex_);

    if (free_indices_.empty()) {
        return {};
    }

    const auto index = free_indices_.back();
    free_indices_.pop_back();

    if (
        index >= slots_.size() ||
        slots_[index].active
    ) {
        if (
            index < slots_.size() &&
            !slots_[index].active
        ) {
            free_indices_.push_back(index);
        }
        return {};
    }

    auto& slot = slots_[index];
    ++slot.generation;
    if(slot.generation==0){
        slot.generation=1;
    }
    slot.active=true;
    ++active_count_;

    return Handle{
        index,
        slot.generation,
        true,
    };
}

void NativeBufferPool::release(
    std::uint32_t index) noexcept {
    std::scoped_lock lock(mutex_);

    if (
        index >= slots_.size() ||
        !slots_[index].active
    ) {
        ++invalid_release_count_;
        return;
    }

    slots_[index].active = false;
    if (active_count_ > 0u) {
        --active_count_;
    }
    free_indices_.push_back(index);
}

bool NativeBufferPool::release(
    Handle handle) noexcept {
    std::scoped_lock lock(mutex_);

    if (
        !handle.valid ||
        handle.index >= slots_.size()
    ) {
        ++invalid_release_count_;
        return false;
    }

    auto& slot = slots_[handle.index];

    if(!slot.active){
        ++invalid_release_count_;
        return false;
    }

    if(slot.generation != handle.generation){
        ++stale_release_count_;
        return false;
    }

    slot.active=false;
    if(active_count_>0u){
        --active_count_;
    }
    free_indices_.push_back(handle.index);
    return true;
}

std::uint32_t NativeBufferPool::capacity() const noexcept {
    std::scoped_lock lock(mutex_);
    return static_cast<std::uint32_t>(
        std::min<std::size_t>(
            slots_.size(),
            std::numeric_limits<std::uint32_t>::max()));
}

std::uint32_t NativeBufferPool::active_blocks() const noexcept {
    std::scoped_lock lock(mutex_);
    return active_count_;
}

std::uint32_t NativeBufferPool::free_blocks() const noexcept {
    std::scoped_lock lock(mutex_);
    return static_cast<std::uint32_t>(
        std::min<std::size_t>(
            free_indices_.size(),
            std::numeric_limits<std::uint32_t>::max()));
}

std::size_t NativeBufferPool::block_size() const noexcept {
    std::scoped_lock lock(mutex_);
    return block_size_;
}

std::size_t NativeBufferPool::resident_bytes() const noexcept {
    std::scoped_lock lock(mutex_);
    if (
        block_size_ != 0u &&
        slots_.size() >
            std::numeric_limits<std::size_t>::max() /
                block_size_
    ) {
        /*
         * This is a telemetry value and a safety input. Saturating at SIZE_MAX is preferable
         * to wrapping into a small value that could make a memory admission layer believe the
         * pool is nearly empty.
         */
        return std::numeric_limits<std::size_t>::max();
    }
    return slots_.size() * block_size_;
}

std::uint64_t
NativeBufferPool::invalid_release_count() const noexcept {
    std::scoped_lock lock(mutex_);
    return invalid_release_count_;
}

std::uint64_t
NativeBufferPool::stale_release_count() const noexcept {
    std::scoped_lock lock(mutex_);
    return stale_release_count_;
}

} // namespace nox::native
