package com.nivukx.nox.core.nativeengine.face
import org.junit.Assert.assertEquals
import org.junit.Test
data class NativeFaceState(val confidence: Double,val metric: Boolean,val reprojectionRms: Double)
class NativeFaceStateTest {@Test fun contractAcceptsMeasuredMetricState(){val s=NativeFaceState(.8,true,.5);assertEquals(true,s.metric)}}
