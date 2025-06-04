package com.example.arteka_crohn.segmentation

import com.example.arteka_crohn.Output0

data class ApiSegmentationResult(
    val box: Output0,
    val mask: Array<FloatArray>,
    val conf: Float,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ApiSegmentationResult

        return mask.contentDeepEquals(other.mask)
    }

    override fun hashCode(): Int {
        return mask.contentDeepHashCode()
    }
}
