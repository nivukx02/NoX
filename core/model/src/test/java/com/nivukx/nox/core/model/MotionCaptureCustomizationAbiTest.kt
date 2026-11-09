package com.nivukx.nox.core.model

import org.junit.Assert.assertTrue
import org.junit.Test

class MotionCaptureCustomizationAbiTest {

    @Test
    fun constructorSignaturesStayWithinJvmParameterSlotLimit() {
        val instance = MotionCaptureCustomization()
        for (constructor in instance.javaClass.declaredConstructors) {
            var slots = 0
            for (parameterType in constructor.parameterTypes) {
                slots +=
                    if (
                        parameterType ==
                            Double::class.javaPrimitiveType ||
                        parameterType ==
                            Long::class.javaPrimitiveType
                    ) {
                        2
                    } else {
                        1
                    }
            }
            assertTrue(
                "Constructor Object exceeds the JVM 255-slot parameter limit: $slots",
                slots <= 255,
            )
        }
    }
}
