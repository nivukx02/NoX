#include "runtime/hardware_buffer_lease_registry.h"

#include <algorithm>
#include <limits>

namespace nox::native::runtime {

HardwareBufferLeaseRegistry::HardwareBufferLeaseRegistry(
    std::uint32_t maximum_active_leases,
    std::size_t maximum_resident_bytes) noexcept
    : maximum_active_leases_(
          std::max<std::uint32_t>(
              maximum_active_leases,
              1u)),
      maximum_resident_bytes_(
          std::max<std::size_t>(
              maximum_resident_bytes,
              1u)) {}

namespace {

std::size_t approximateResidentBytes(
    const AHardwareBuffer_Desc& desc) noexcept {
    const auto maxValue =
        std::numeric_limits<std::size_t>::max();

    const std::size_t width =
        static_cast<std::size_t>(desc.width);
    const std::size_t height =
        static_cast<std::size_t>(desc.height);
    const std::size_t layers =
        static_cast<std::size_t>(
            std::max<std::uint32_t>(
                desc.layers,
                1u));

    if (width == 0u || height == 0u) {
        return 0u;
    }

    const std::size_t rowStridePixels =
        std::max<std::size_t>(
            width,
            static_cast<std::size_t>(desc.stride));

    /*
     * The NDK descriptor stride includes allocator padding for the luma/linear row in pixels.
     * YUV formats are accounted for using their aggregate plane footprint instead of pretending
     * they are one tightly packed RGB plane. Unknown/vendor formats intentionally use an
     * over-estimate so memory admission fails conservatively rather than under-counting residency.
     */
    std::size_t bytesPerPixelNumerator = 8u;
    std::size_t bytesPerPixelDenominator = 1u;

    switch (desc.format) {
        case AHARDWAREBUFFER_FORMAT_R8_UNORM:
        case AHARDWAREBUFFER_FORMAT_S8_UINT:
            bytesPerPixelNumerator = 1u;
            break;

        case AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM:
        case AHARDWAREBUFFER_FORMAT_D16_UNORM:
        case AHARDWAREBUFFER_FORMAT_R16_UINT:
            bytesPerPixelNumerator = 2u;
            break;

        case AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM:
        case AHARDWAREBUFFER_FORMAT_D24_UNORM:
            bytesPerPixelNumerator = 3u;
            break;

        case AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM:
        case AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM:
        case AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM:
        case AHARDWAREBUFFER_FORMAT_D32_FLOAT:
        case AHARDWAREBUFFER_FORMAT_D24_UNORM_S8_UINT:
        case AHARDWAREBUFFER_FORMAT_R16G16_UINT:
            bytesPerPixelNumerator = 4u;
            break;

        case AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT:
        case AHARDWAREBUFFER_FORMAT_R10G10B10A10_UNORM:
            bytesPerPixelNumerator = 8u;
            break;

        case AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420:
            bytesPerPixelNumerator = 3u;
            bytesPerPixelDenominator = 2u;
            break;

        case AHARDWAREBUFFER_FORMAT_YCbCr_P010:
            bytesPerPixelNumerator = 3u;
            break;

        case AHARDWAREBUFFER_FORMAT_YCbCr_P210:
            bytesPerPixelNumerator = 4u;
            break;

        default:
            /*
             * Vendor, RAW, implementation-defined and opaque formats can have padding,
             * hidden planes, or implementation-specific allocation. Eight bytes/pixel is
             * deliberately conservative because this is a safety budget, not a storage report.
             */
            bytesPerPixelNumerator = 8u;
            bytesPerPixelDenominator = 1u;
            break;
    }

    if (
        rowStridePixels >
        maxValue /
            bytesPerPixelNumerator
    ) {
        return maxValue;
    }

    std::size_t rowBytes =
        rowStridePixels *
        bytesPerPixelNumerator;

    if (
        bytesPerPixelDenominator > 1u
    ) {
        if (
            rowBytes >
            maxValue /
                height
        ) {
            return maxValue;
        }

        rowBytes *= height;

        if (
            rowBytes >
            maxValue /
                layers
        ) {
            return maxValue;
        }

        rowBytes *= layers;
        rowBytes /=
            bytesPerPixelDenominator;
        return rowBytes;
    }

    if (
        rowBytes >
        maxValue /
            height
    ) {
        return maxValue;
    }

    rowBytes *= height;

    if (
        rowBytes >
        maxValue /
            layers
    ) {
        return maxValue;
    }

    return rowBytes * layers;
}

}  // namespace

HardwareBufferLeaseRegistry::~HardwareBufferLeaseRegistry() {
    clear();
}

std::uint64_t HardwareBufferLeaseRegistry::acquire(
    AHardwareBuffer* buffer,
    HardwareBufferLeaseInfo* info) noexcept {
    if (buffer == nullptr) {
        std::scoped_lock lock(mutex_);
        ++acquire_failures_;
        return 0u;
    }

    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(buffer, &desc);

    Lease lease{};
    lease.buffer = buffer;
    lease.info.width = desc.width;
    lease.info.height = desc.height;
    lease.info.layers = desc.layers;
    lease.info.format = desc.format;
    lease.info.usage = desc.usage;
    lease.info.resident_bytes =
        approximateResidentBytes(desc);

    std::scoped_lock lock(mutex_);

    if (
        leases_.size() >=
            static_cast<std::size_t>(
                maximum_active_leases_) ||
        lease.info.resident_bytes >
            maximum_resident_bytes_ -
                std::min(
                    resident_bytes_,
                    maximum_resident_bytes_)
    ) {
        ++acquire_failures_;
        return 0u;
    }

    if (next_lease_id_ == 0u) {
        next_lease_id_ = 1u;
    }

    const auto lease_id = next_lease_id_++;
    lease.info.lease_id = lease_id;

    AHardwareBuffer_acquire(buffer);

    try {
        const auto [iterator, inserted] =
            leases_.emplace(lease_id, lease);

        if (!inserted) {
            AHardwareBuffer_release(buffer);
            ++acquire_failures_;
            return 0u;
        }

        if (
            lease.info.resident_bytes >
                maximum_resident_bytes_ -
                    std::min(
                        resident_bytes_,
                        maximum_resident_bytes_)
        ) {
            leases_.erase(lease_id);
            AHardwareBuffer_release(buffer);
            ++acquire_failures_;
            return 0u;
        }

        resident_bytes_ +=
            lease.info.resident_bytes;
        if (info != nullptr) {
            *info = iterator->second.info;
        }
        return lease_id;
    } catch (...) {
        AHardwareBuffer_release(buffer);
        ++acquire_failures_;
        return 0u;
    }
}

bool HardwareBufferLeaseRegistry::release(
    std::uint64_t lease_id) noexcept {
    if (lease_id == 0u) {
        std::scoped_lock lock(mutex_);
        ++invalid_release_count_;
        return false;
    }

    AHardwareBuffer* buffer = nullptr;
    std::size_t resident_bytes = 0u;

    {
        std::scoped_lock lock(mutex_);
        const auto iterator = leases_.find(lease_id);
        if (iterator == leases_.end()) {
            ++invalid_release_count_;
            return false;
        }

        buffer = iterator->second.buffer;
        resident_bytes = iterator->second.info.resident_bytes;
        leases_.erase(iterator);
        resident_bytes_ =
            resident_bytes_ > resident_bytes
                ? resident_bytes_ - resident_bytes
                : 0u;
    }

    if (buffer != nullptr) {
        AHardwareBuffer_release(buffer);
    }
    return true;
}

void HardwareBufferLeaseRegistry::clear() noexcept {
    std::unordered_map<std::uint64_t, Lease> leases;

    {
        std::scoped_lock lock(mutex_);
        leases.swap(leases_);
        resident_bytes_ = 0u;
    }

    for (auto& entry : leases) {
        if (entry.second.buffer != nullptr) {
            AHardwareBuffer_release(
                entry.second.buffer);
        }
    }
}

std::uint32_t HardwareBufferLeaseRegistry::active_count() const noexcept {
    std::scoped_lock lock(mutex_);
    return static_cast<std::uint32_t>(
        std::min<std::size_t>(
            leases_.size(),
            std::numeric_limits<std::uint32_t>::max()));
}

std::size_t HardwareBufferLeaseRegistry::resident_bytes() const noexcept {
    std::scoped_lock lock(mutex_);
    return resident_bytes_;
}

std::uint64_t HardwareBufferLeaseRegistry::acquire_failures() const noexcept {
    std::scoped_lock lock(mutex_);
    return acquire_failures_;
}

std::uint64_t HardwareBufferLeaseRegistry::invalid_release_count() const noexcept {
    std::scoped_lock lock(mutex_);
    return invalid_release_count_;
}

std::uint32_t HardwareBufferLeaseRegistry::maximum_active_leases() const noexcept {
    std::scoped_lock lock(mutex_);
    return maximum_active_leases_;
}

std::size_t HardwareBufferLeaseRegistry::maximum_resident_bytes() const noexcept {
    std::scoped_lock lock(mutex_);
    return maximum_resident_bytes_;
}

}  // namespace nox::native::runtime
