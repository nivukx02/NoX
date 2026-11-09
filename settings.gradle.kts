pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "NoX"
include(":app")
include(":core:model")
include(":core:sync")
include(":core:network")
include(":core:capture")
include(":core:calibration")
include(":core:reconstruction")
include(":core:ai")
include(":core:native-engine")
