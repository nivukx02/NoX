import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.security.MessageDigest

plugins { id("com.android.library") }

android {
    namespace = "com.nivukx.nox.core.ai"
    compileSdk = 37
    defaultConfig {
        minSdk = 30
        consumerProguardFiles("consumer-rules.pro")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

val handModelFile = layout.projectDirectory.file("src/main/assets/hand_landmarker.task").asFile
val handModelSha256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
val handModelUrl =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
val handModelMinimumBytes = 1_000_000L

val poseModelFile = layout.projectDirectory.file("src/main/assets/pose_landmarker_full.task").asFile
val poseModelUrl =
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
val poseModelMinimumBytes = 8_000_000L
val poseModelSha256 = "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"

val personSegmentationModelFile =
    layout.projectDirectory
        .file("src/main/assets/selfie_multiclass_256x256.tflite")
        .asFile
val personSegmentationModelUrl =
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite"
val personSegmentationModelMinimumBytes = 14_000_000L
val personSegmentationModelSha256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"

fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            digest.update(buffer, 0, read)
        }
    }
    return digest.digest().joinToString("") { byte ->
        "%02x".format(byte.toInt() and 0xff)
    }
}

fun downloadModel(
    url: String,
    target: File,
    expectedSha256: String,
) {
    target.parentFile.mkdirs()
    val temporary = File(target.parentFile, target.name + ".download")
    var lastFailure: Throwable? = null

    for (attempt in 1..5) {
        try {
            temporary.delete()
            val connection =
                (URI(url).toURL().openConnection() as HttpURLConnection).apply {
                    connectTimeout = 20_000
                    readTimeout = 120_000
                    requestMethod = "GET"
                    instanceFollowRedirects = true
                }
            try {
                val response = connection.responseCode
                check(response in 200..299) {
                    "HTTP $response while downloading $url"
                }
                connection.inputStream.use { input ->
                    temporary.outputStream().use { output ->
                        input.copyTo(output, bufferSize = 64 * 1024)
                    }
                }
            } finally {
                connection.disconnect()
            }

            check(temporary.isFile && temporary.length() > 0L) {
                "Downloaded model is empty: $url"
            }
            val actualSha256 = sha256(temporary)
            check(actualSha256.equals(expectedSha256, ignoreCase = true)) {
                "Model checksum mismatch for $url: expected $expectedSha256, got $actualSha256"
            }
            check(temporary.renameTo(target)) {
                "Unable to atomically install model at ${target.absolutePath}"
            }
            return
        } catch (failure: Throwable) {
            lastFailure = failure
            temporary.delete()
            if (attempt < 5) {
                Thread.sleep(1_000L * attempt)
            }
        }
    }

    throw GradleException(
        "Failed to provision model after 5 attempts: $url",
        lastFailure,
    )
}

fun ensureModel(
    file: File,
    minimumBytes: Long,
    url: String,
    expectedSha256: String,
) {
    if (file.isFile && file.length() >= minimumBytes &&
        sha256(file).equals(expectedSha256, ignoreCase = true)
    ) {
        return
    }
    downloadModel(url, file, expectedSha256)
    check(file.isFile && file.length() >= minimumBytes) {
        "Unable to provision model at ${file.absolutePath}"
    }
    check(sha256(file).equals(expectedSha256, ignoreCase = true)) {
        "Installed model checksum mismatch at ${file.absolutePath}"
    }
}

val ensurePersonSegmentationModel =
    tasks.register("ensurePersonSegmentationModel") {
        outputs.file(personSegmentationModelFile)
        doLast {
            ensureModel(
                file = personSegmentationModelFile,
                minimumBytes = personSegmentationModelMinimumBytes,
                url = personSegmentationModelUrl,
                expectedSha256 = personSegmentationModelSha256,
            )
        }
    }

val ensurePoseLandmarkerModel = tasks.register("ensurePoseLandmarkerModel") {
    outputs.file(poseModelFile)
    doLast {
        ensureModel(
            file = poseModelFile,
            minimumBytes = poseModelMinimumBytes,
            url = poseModelUrl,
            expectedSha256 = poseModelSha256,
        )
    }
}

val ensureHandLandmarkerModel = tasks.register("ensureHandLandmarkerModel") {
    outputs.file(handModelFile)
    doLast {
        ensureModel(
            file = handModelFile,
            minimumBytes = handModelMinimumBytes,
            url = handModelUrl,
            expectedSha256 = handModelSha256,
        )
    }
}

val validateHandModel = tasks.register("validateHandModel") {
    dependsOn(ensureHandLandmarkerModel)
    inputs.file(handModelFile)
    doLast {
        check(handModelFile.isFile && handModelFile.length() > 0L) {
            "Missing MediaPipe hand landmark model: " + handModelFile.absolutePath + ". " +
                "The verified model provisioning task must complete before validation."
        }
        check(sha256(handModelFile).equals(handModelSha256, ignoreCase = true)) {
            "MediaPipe hand landmark model checksum mismatch at ${handModelFile.absolutePath}"
        }
    }
}

tasks.configureEach {
    if (name == "preBuild") {
        dependsOn(validateHandModel)
        dependsOn(ensurePoseLandmarkerModel)
        dependsOn(ensurePersonSegmentationModel)
    }
}

dependencies {
    implementation(project(":core:model"))
    implementation(project(":core:native-engine"))
    implementation("com.google.android.gms:play-services-tasks:18.2.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")
    implementation("com.google.mediapipe:tasks-vision:0.10.29")
    implementation("com.google.mlkit:pose-detection:18.0.0-beta5")
    implementation("com.google.mlkit:pose-detection-accurate:18.0.0-beta5")
    implementation("com.google.mlkit:face-detection:16.1.7")
    implementation("com.google.mlkit:face-mesh-detection:16.0.0-beta3")
    implementation("com.google.mlkit:object-detection:17.0.2")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlin:kotlin-test:2.4.10")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:2.4.10")
}
