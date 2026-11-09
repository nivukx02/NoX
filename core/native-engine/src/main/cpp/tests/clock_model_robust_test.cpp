#include "synchronization/clock_model.h"

int main() {
    using namespace nox::native::sync;
    std::vector<std::uint64_t> remote{
        1000000,2000000,3000000,4000000,5000000,6000000};
    std::vector<std::uint64_t> local{
        1010000,2010005,3010010,9000000,5010020,6010025};
    const auto fit=ClockModel{}.fit(remote,local);
    return (
        fit.valid &&
        fit.confidence>0.5 &&
        fit.jitter_rms_nanos<1e6 &&
        fit.residual_p95_nanos<1e6 &&
        fit.support_ratio>0.70 &&
        fit.outlier_ratio<0.30 &&
        fit.iterations>0
    ) ? 0 : 1;
}
