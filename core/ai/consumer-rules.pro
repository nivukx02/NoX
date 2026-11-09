# ONNX Runtime relies on reflective/native bindings on Android.
-keep class ai.onnxruntime.** { *; }

# NoX depth providers are selected dynamically.
-keep class com.nivukx.nox.core.ai.depth.** { *; }
