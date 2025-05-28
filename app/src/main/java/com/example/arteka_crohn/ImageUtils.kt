package com.example.arteka_crohn

import kotlin.math.exp

object ImageUtils {

    fun Array<FloatArray>.fill(value: Float) {
        for (row in this) {
            row.fill(value)
        }
    }
/// <summary>
/// Clones a list of 2D arrays.
/// </summary>
/// <returns>A new list of cloned 2D arrays.</returns>
    fun List<Array<FloatArray>>.clone(): List<Array<FloatArray>> {
        return this.map { array -> array.map { it.clone() }.toTypedArray() }
    }
/// <summary>
/// Scales a 2D array (mask) to the target width and height.
/// </summary>
fun Array<FloatArray>.scaleMask(newHeight: Int, newWidth: Int): Array<FloatArray> {
    val originalHeight = this.size
    if (originalHeight == 0) return Array(newHeight) { FloatArray(newWidth) }
    val originalWidth = this[0].size
    if (originalWidth == 0) return Array(newHeight) { FloatArray(newWidth) }

    val scaled = Array(newHeight) { FloatArray(newWidth) }

    // Implémentation simpliste (plus proche voisin), tu auras besoin d'une meilleure interpolation
    for (y_new in 0 until newHeight) {
        for (x_new in 0 until newWidth) {
            val y_old = (y_new * originalHeight / newHeight.toFloat()).toInt().coerceIn(0, originalHeight - 1)
            val x_old = (x_new * originalWidth / newWidth.toFloat()).toInt().coerceIn(0, originalWidth - 1)
            scaled[y_new][x_new] = this[y_old][x_old]
        }
    }
    return scaled
}


    fun Array<FloatArray>.toMask(): Array<IntArray> =
        map { row -> row.map { if (it > 0) 1 else 0 }.toIntArray() }.toTypedArray()


    fun Array<IntArray>.smooth(kernel: Int) : Array<IntArray> {
        // Using Array because it is faster then List
        val maskFloat = Array(this.size) { i ->
            FloatArray(this[i].size) { j ->
                if (this[i][j] > 0) 1F else 0F
            }
        }
        val gaussianKernel = createGaussianKernel(kernel)
        val blurredImage = applyGaussianBlur(maskFloat, gaussianKernel)
        return thresholdImage(blurredImage)
    }

    private fun createGaussianKernel(size: Int): Array<FloatArray> {
        val sigma = 2F
        val kernel = Array(size) { FloatArray(size) }
        val mean = size / 2
        var sum = 0F

        for (x in 0 until size) {
            for (y in 0 until size) {
                kernel[x][y] = (1F / (2F * Math.PI.toFloat() * sigma * sigma)) * exp(
                    -((x - mean) * (x - mean) + (y - mean) * (y - mean)) / (2F * sigma * sigma)
                )
                sum += kernel[x][y]
            }
        }

        for (x in 0 until size) {
            for (y in 0 until size) {
                kernel[x][y] /= sum
            }
        }

        return kernel
    }

/// <summary>
/// Applies Gaussian blur to the image using the provided kernel.
/// </summary>
/// <param name="image">The input image as a 2D array of floats.</param>
/// <param name="kernel">The Gaussian kernel as a 2D array of floats.</param>
/// <returns>The blurred image as a 2D array of floats.</returns>
    private fun applyGaussianBlur(image: Array<FloatArray>, kernel: Array<FloatArray>): Array<FloatArray> {
        val height = image.size
        val width = image[0].size
        val kernelSize = kernel.size
        val offset = kernelSize / 2
        val blurredImage = Array(height) { FloatArray(width) }

        for (i in image.indices) {
            for (j in image[i].indices) {
                if (i < offset || j < offset || i >= height - offset || j >= width - offset) {
                    blurredImage[i][j] = image[i][j]
                    continue
                }

                var sum = 0F
                for (ki in kernel.indices) {
                    for (kj in kernel[ki].indices) {
                        sum += image[i - offset + ki][j - offset + kj] * kernel[ki][kj]
                    }
                }
                blurredImage[i][j] = sum
            }
        }

        return blurredImage
    }

    private fun thresholdImage(image: Array<FloatArray>): Array<IntArray> {
        val height = image.size
        val width = image[0].size
        return Array(height) { i ->
            IntArray(width) { j ->
                if (image[i][j] > 0.9F) 1 else 0
            }
        }
    }
}