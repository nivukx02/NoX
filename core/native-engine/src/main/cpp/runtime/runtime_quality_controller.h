#pragma once

namespace nox::native::runtime {

enum class WorkClass {
    Critical,
    Normal,
    Optional,
};

struct RuntimeQualityInput {
    double thermalHeadroom{1.0};
    double memoryHeadroom{1.0};
    double gpuHeadroom{1.0};
    double npuHeadroom{1.0};
    bool batterySaver{false};
    double queueUtilization{0.0};
    double frameBudgetUtilization{0.0};
    double droppedFrameRatio{0.0};
};

struct RuntimeQualityConfig {
    double minimumQualityScale{0.18};
    double thermalFloor{0.18};
    double memoryFloor{0.15};
    double gpuFloor{0.12};
    double npuFloor{0.12};
    double queueShedThreshold{0.90};
    double frameBudgetShedThreshold{1.15};
    double droppedFrameShedThreshold{0.08};
};

struct RuntimeQualityDecision {
    double qualityScale{1.0};
    double pressure{0.0};
    double queueCapacityScale{1.0};
    bool allowGpu{true};
    bool shedOptionalWork{false};
    bool thermalConstrained{false};
    bool memoryConstrained{false};
    bool acceptCriticalWork{true};
};

class RuntimeQualityController final {
public:
    explicit RuntimeQualityController(RuntimeQualityConfig config = {}) noexcept;

    [[nodiscard]] RuntimeQualityDecision evaluate(
        const RuntimeQualityInput& input,
        WorkClass workClass = WorkClass::Normal) const noexcept;

private:
    RuntimeQualityConfig config_{};

    static double sanitize(double value) noexcept;
    static double clamp01(double value) noexcept;
};

} // namespace nox::native::runtime
