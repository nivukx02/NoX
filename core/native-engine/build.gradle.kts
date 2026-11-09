plugins { id("com.android.library") }

android {
    namespace = "com.nivukx.nox.core.nativeengine"
    compileSdk = 37
    defaultConfig {
        minSdk = 30
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++20", "-fvisibility=hidden")
            }
        }
    }
    buildFeatures { prefab = false }
    externalNativeBuild {
        cmake { path = file("CMakeLists.txt") }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    testOptions { unitTests.isReturnDefaultValues = true }
}

dependencies {
    implementation(project(":core:model"))
    testImplementation("junit:junit:4.13.2")
}
