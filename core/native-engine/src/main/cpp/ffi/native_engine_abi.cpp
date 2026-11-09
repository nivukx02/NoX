#include "ffi/native_engine_abi.h"

#include "runtime/native_engine_runtime.h"

extern "C" std::uint32_t native_engine_abi_version() noexcept {
    return nox::native::kNativeEngineAbiVersion;
}

extern "C" nox::native::NativeRuntimeHandle native_runtime_create(
    const nox::native::NativeRuntimeConfig* config) noexcept {
    try {
        const auto normalized = nox::native::normalize_runtime_config(config);
        return static_cast<nox::native::NativeRuntimeHandle>(
            new nox::native::NativeEngineRuntime(normalized));
    } catch (...) {
        return nullptr;
    }
}

extern "C" bool native_runtime_start(nox::native::NativeRuntimeHandle runtime) noexcept {
    if (runtime == nullptr) return false;
    return static_cast<nox::native::NativeEngineRuntime*>(runtime)->start();
}

extern "C" nox::native::NativeIngestionResult native_runtime_ingest(
    nox::native::NativeRuntimeHandle runtime,
    const nox::native::NativeCapturePacket* packet) noexcept {
    if (runtime == nullptr || packet == nullptr) return {};
    return static_cast<nox::native::NativeEngineRuntime*>(runtime)->ingest(*packet);
}

extern "C" void native_runtime_stop(nox::native::NativeRuntimeHandle runtime) noexcept {
    if (runtime == nullptr) return;
    static_cast<nox::native::NativeEngineRuntime*>(runtime)->stop();
}

extern "C" void native_runtime_destroy(nox::native::NativeRuntimeHandle runtime) noexcept {
    delete static_cast<nox::native::NativeEngineRuntime*>(runtime);
}

extern "C" nox::native::NativeDiagnostics native_runtime_diagnostics(
    nox::native::NativeRuntimeHandle runtime) noexcept {
    if (runtime == nullptr) return {};
    return static_cast<nox::native::NativeEngineRuntime*>(runtime)->diagnostics();
}
