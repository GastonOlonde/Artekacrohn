package com.example.arteka_crohn.segmentation.postprocessing

import android.util.Log
import com.example.arteka_crohn.Output0
import com.example.arteka_crohn.ImageUtils.clone
import com.example.arteka_crohn.ImageUtils.scaleMask
import com.example.arteka_crohn.segmentation.ApiSegmentationResult
import com.example.arteka_crohn.segmentation.model.TensorFlowLiteModel

/**
 * Interface pour le post-traitement des résultats de segmentation
 * Respecte le principe de substitution de Liskov (L dans SOLID)
 */
interface SegmentationPostprocessor {
    fun processDetections(coordinatesArray: FloatArray, maskProtoArray: FloatArray): List<ApiSegmentationResult>?
}

/**
 * Implémentation concrète du post-traitement pour la segmentation d'instance
 */
class InstanceSegmentationPostprocessor(
    private val model: TensorFlowLiteModel,
    private val confidenceThreshold: Float = 0.45F,
    private val iouThreshold: Float = 0.5F
) : SegmentationPostprocessor {

    override fun processDetections(coordinatesArray: FloatArray, maskProtoArray: FloatArray): List<ApiSegmentationResult>? {
        // Traitement des boîtes détectées
        val bestBoxes = findBestBoxes(coordinatesArray) ?: return null
        
        // Traitement des masques
        val maskProto = reshapeMaskOutput(maskProtoArray)
        
        // Création des résultats de segmentation
        return bestBoxes.map { box ->
            ApiSegmentationResult(
                box = box,
                mask = getFinalMask(box, maskProto),
                conf = box.cnf
            )
        }
    }

    private fun findBestBoxes(array: FloatArray): List<Output0>? {
        val output0List = ArrayList<Output0>(model.numElements)
        val labels = model.getLabels()

        for (c in 0 until model.numElements) {
            var maxConf = confidenceThreshold
            var maxIdx = -1
            var currentInd = 4

            // Optimisation: calcul d'index simplifié
            while (currentInd < (model.numChannel - model.masksNum)) {
                val conf = array[c + model.numElements * currentInd]
                if (conf > maxConf) {
                    maxConf = conf
                    maxIdx = currentInd - 4
                }
                currentInd++
            }

            if (maxConf > confidenceThreshold && maxIdx in labels.indices) {
                val cx = array[c]
                val cy = array[c + model.numElements]
                val w = array[c + model.numElements * 2]
                val h = array[c + model.numElements * 3]
                val x1 = cx - w/2f
                val y1 = cy - h/2f
                val x2 = cx + w/2f
                val y2 = cy + h/2f

                if (x1 in 0f..1f && y1 in 0f..1f && x2 in 0f..1f && y2 in 0f..1f) {
                    val maskWeights = FloatArray(model.masksNum) { i ->
                        array[c + model.numElements * (currentInd + i)]
                    }

                    output0List.add(
                        Output0(
                            x1, y1, x2, y2, cx, cy, w, h,
                            maxConf, maxIdx, labels[maxIdx],
                            maskWeights.toList()
                        )
                    )
                }
            }
        }

        return output0List.takeIf { it.isNotEmpty() }?.let { applyNMS(it) }
    }

    private fun applyNMS(output0List: List<Output0>): MutableList<Output0> {
        val sortedBoxes = output0List.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<Output0>()
        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.removeAt(0)
            selectedBoxes.add(first)
            sortedBoxes.removeAll { calculateIoU(first, it) >= iouThreshold }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: Output0, box2: Output0): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    private fun reshapeMaskOutput(floatArray: FloatArray): List<Array<FloatArray>> {
        return List(model.masksNum) { mask ->
            Array(model.xPoints) { r ->
                FloatArray(model.yPoints) { c ->
                    floatArray[model.masksNum * model.yPoints * r + model.masksNum * c + mask]
                }
            }
        }
    }

    private fun getFinalMask(output0: Output0, output1: List<Array<FloatArray>>): Array<FloatArray> {
        return try {
            val output1Copy = output1.clone()
            val relX1 = (output0.x1 * model.xPoints).toInt()
            val relY1 = (output0.y1 * model.yPoints).toInt()
            val relX2 = (output0.x2 * model.xPoints).toInt()
            val relY2 = (output0.y2 * model.yPoints).toInt()

            val zero = Array(model.yPoints) { FloatArray(model.xPoints) }

            // Optimisation: réduire les boucles imbriquées
            for (index in output1Copy.indices) {
                val proto = output1Copy[index]
                val weight = output0.maskWeight[index]

                for (y in maxOf(relY1, 0) until minOf(relY2, model.yPoints)) {
                    for (x in maxOf(relX1, 0) until minOf(relX2, model.xPoints)) {
                        zero[y][x] += proto[y][x] * weight
                    }
                }
            }
            zero.scaleMask(450, 600)
        } catch (e: Exception) {
            Log.e("MaskError", "Error in mask generation", e)
            Array(model.yPoints) { FloatArray(model.xPoints) { 0f } }
        }
    }
}
