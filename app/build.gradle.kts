import java.awt.BasicStroke
import java.awt.Color
import java.awt.GradientPaint
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

val generatedNoxArtworkRes =
    layout.buildDirectory.dir("generated/nox-artwork/res").get().asFile

val syncNoxArtwork =
    tasks.register("syncNoxArtwork") {
        val source = rootProject.file("artwork_16k.png")
        val outputDirectory =
            generatedNoxArtworkRes.resolve("drawable-nodpi")
        val output = outputDirectory.resolve("nox_logo.png")

        inputs.file(source).optional()
        outputs.file(output)

        doLast {
            outputDirectory.mkdirs()
            val targetSize = 2048

            fun createFallbackNoxArtwork(size: Int): BufferedImage {
                val image =
                    BufferedImage(
                        size,
                        size,
                        BufferedImage.TYPE_INT_RGB,
                    )
                val graphics = image.createGraphics()
                try {
                    graphics.setRenderingHint(
                        RenderingHints.KEY_ANTIALIASING,
                        RenderingHints.VALUE_ANTIALIAS_ON,
                    )
                    graphics.setRenderingHint(
                        RenderingHints.KEY_RENDERING,
                        RenderingHints.VALUE_RENDER_QUALITY,
                    )
                    graphics.paint =
                        GradientPaint(
                            0f,
                            0f,
                            Color(7, 12, 28),
                            size.toFloat(),
                            size.toFloat(),
                            Color(28, 14, 53),
                        )
                    graphics.fillRect(0, 0, size, size)

                    graphics.paint =
                        GradientPaint(
                            0f,
                            0f,
                            Color(53, 232, 255),
                            size.toFloat(),
                            size.toFloat(),
                            Color(166, 95, 255),
                        )
                    graphics.stroke =
                        BasicStroke(
                            size * 0.018f,
                            BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND,
                        )
                    graphics.drawRoundRect(
                        (size * 0.15).toInt(),
                        (size * 0.15).toInt(),
                        (size * 0.70).toInt(),
                        (size * 0.70).toInt(),
                        (size * 0.16).toInt(),
                        (size * 0.16).toInt(),
                    )
                    graphics.stroke =
                        BasicStroke(
                            size * 0.065f,
                            BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND,
                        )
                    graphics.drawLine(
                        (size * 0.32).toInt(),
                        (size * 0.67).toInt(),
                        (size * 0.32).toInt(),
                        (size * 0.33).toInt(),
                    )
                    graphics.drawLine(
                        (size * 0.32).toInt(),
                        (size * 0.33).toInt(),
                        (size * 0.68).toInt(),
                        (size * 0.67).toInt(),
                    )
                    graphics.drawLine(
                        (size * 0.68).toInt(),
                        (size * 0.67).toInt(),
                        (size * 0.68).toInt(),
                        (size * 0.33).toInt(),
                    )
                } finally {
                    graphics.dispose()
                }
                return image
            }

            val targetImage =
                if (source.isFile) {
                    val sourceImage =
                        ImageIO.read(source)
                            ?: error("Unable to decode artwork_16k.png")
                    try {
                        BufferedImage(
                            targetSize,
                            targetSize,
                            BufferedImage.TYPE_INT_RGB,
                        ).also { scaledImage ->
                            val graphics = scaledImage.createGraphics()
                            try {
                                graphics.setRenderingHint(
                                    RenderingHints.KEY_INTERPOLATION,
                                    RenderingHints.VALUE_INTERPOLATION_BICUBIC,
                                )
                                graphics.setRenderingHint(
                                    RenderingHints.KEY_RENDERING,
                                    RenderingHints.VALUE_RENDER_QUALITY,
                                )
                                graphics.setRenderingHint(
                                    RenderingHints.KEY_ANTIALIASING,
                                    RenderingHints.VALUE_ANTIALIAS_ON,
                                )
                                graphics.drawImage(
                                    sourceImage,
                                    0,
                                    0,
                                    targetSize,
                                    targetSize,
                                    null,
                                )
                            } finally {
                                graphics.dispose()
                            }
                        }
                    } finally {
                        sourceImage.flush()
                    }
                } else {
                    logger.warn(
                        "artwork_16k.png is absent; generating the deterministic NoX fallback mark. " +
                            "The original artwork remains preferred whenever it is present.",
                    )
                    createFallbackNoxArtwork(targetSize)
                }

            try {
                check(ImageIO.write(targetImage, "png", output)) {
                    "PNG writer is unavailable for generated NoX artwork"
                }
                check(output.isFile && output.length() < 5_000_000L) {
                    "Generated NoX artwork is not install-safe"
                }
            } finally {
                targetImage.flush()
            }
        }
    }

android {
    namespace = "com.nivukx.nox"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.nivukx.nox"
        minSdk = 30
        targetSdk = 37
        versionCode = 12000
        versionName = "1.2.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables.useSupportLibrary = true
    }

    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "armeabi-v7a", "x86_64", "x86")
            isUniversalApk = false
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.0,LGPL2.1}"
    }

    sourceSets {
        getByName("main") {
            res.srcDir(generatedNoxArtworkRes)
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    implementation("com.google.android.gms:play-services-tasks:18.2.0")
    implementation(project(":core:model"))
    implementation(project(":core:sync"))
    implementation(project(":core:network"))
    implementation(project(":core:capture"))
    implementation(project(":core:calibration"))
    implementation(project(":core:reconstruction"))
    implementation(project(":core:ai"))
    implementation(project(":core:native-engine"))

    val composeBom = platform("androidx.compose:compose-bom:2026.09.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material3:material3-adaptive-navigation-suite")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.0")
    implementation("androidx.lifecycle:lifecycle-process:2.11.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlin:kotlin-test:2.4.10")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:2.4.10")
}


tasks.named("preBuild").configure {
    dependsOn(syncNoxArtwork)
}
