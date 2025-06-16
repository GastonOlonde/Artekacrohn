package com.example.arteka_crohn.detection.model

import android.content.Context
import android.util.Log
import com.example.arteka_crohn.Output0
import com.example.arteka_crohn.detection.config.DetectionConfig
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * Implémentation spécifique pour les modèles RT-DETR (Real-Time Detection Transformer)
 * Format d'entrée: [1, 640, 640, 3]
 * Format de sortie: [1, 300, 5] où 5 = [x1, y1, x2, y2, score]
 */
class RTDETRModelDetector(
    private val onMessageCallback: (String) -> Unit
) : BaseModelDetector() {
    
    private val TAG = "RTDETRModelDetector"
    
    override fun initialize(context: Context, modelPath: String, labelPath: String?) {
        onMessage = onMessageCallback
        super.initialize(context, modelPath, labelPath)
        
        // RT-DETR a une entrée fixe de 640x640
        _inputWidth = 640
        _inputHeight = 640
        
        // Forcer le type de modèle à RT_DETR
        _modelType = ModelType.RT_DETR
        
        Log.d(TAG, "Initialized RT-DETR model with input dimensions: ${_inputWidth}x${_inputHeight}")
    }
    
    override fun runInference(inputBuffers: Array<ByteBuffer>): Any {
        val interp = interpreter ?: throw IllegalStateException("Interpreter not initialized")
        
        // Pour RT-DETR, on a un seul buffer de sortie avec les détections
        val outputShape = outputShapes[0]
        val outputBuffer = ByteBuffer.allocateDirect(
            outputShape.reduce { acc, i -> acc * i } * 4 // 4 bytes par float
        ).apply {
            order(ByteOrder.nativeOrder())
        }
        
        val outputs = mapOf<Int, Any>(
            0 to outputBuffer.rewind()
        )
        
        // Exécuter l'inférence sans delegates (selon les exigences)
        interp.runForMultipleInputsOutputs(inputBuffers, outputs)
        
        // Convertir le buffer en tableau pour le post-traitement
        outputBuffer.rewind()
        val outputSize = outputBuffer.capacity() / 4
        val outputArray = FloatArray(outputSize)
        
        for (i in 0 until outputSize) {
            outputArray[i] = outputBuffer.getFloat()
        }
        
        return outputArray
    }
    
    override fun processOutput(rawOutput: Any, confidenceThreshold: Float): List<Output0> {
        if (rawOutput !is FloatArray) {
            throw IllegalArgumentException("Raw output must be FloatArray for RT-DETR")
        }
        
        val detections = processRTDETROutput(rawOutput, confidenceThreshold)
        return applyNMS(detections, DetectionConfig.IOU_THRESHOLD)
    }
    
    /**
     * Traite les sorties du modèle RT-DETR
     * Format: [1, 300, 5] où 5 = [x1, y1, x2, y2, score]
     */
    private fun processRTDETROutput(outputArray: FloatArray, confidenceThreshold: Float): List<Output0> {
        val results = mutableListOf<Output0>()
        
        // RT-DETR a une sortie de format [1, 300, 5]
        val numDetections = 300
        val valuesPerDetection = 5
        
        for (i in 0 until numDetections) {
            val offset = i * valuesPerDetection
            
            // Les coordonnées sont normalisées (0-1)
            val x1 = outputArray[offset]
            val y1 = outputArray[offset + 1]
            val x2 = outputArray[offset + 2]
            val y2 = outputArray[offset + 3]
            val score = outputArray[offset + 4]
            
            // Filtrer par seuil de confiance
            if (score < confidenceThreshold) continue
            
            // Calculer le centre et les dimensions de la boîte
            val width = x2 - x1
            val height = y2 - y1
            val centerX = x1 + width / 2
            val centerY = y1 + height / 2
            
            // Déterminer la classe (pour RT-DETR nous n'avons qu'une classe dans ce format)
            val detectedClass = 0 // Par défaut, classe unique
            val className = if (detectedClass < labels.size) labels[detectedClass] else "Object"
            
            results.add(
                Output0(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    cx = centerX,
                    cy = centerY,
                    w = width,
                    h = height,
                    cnf = score,
                    cls = detectedClass,
                    clsName = className,
                    maskWeight = emptyList() // Pas de masque pour la détection simple
                )
            )
        }
        
        Log.d(TAG, "RT-DETR detected ${results.size} objects above threshold $confidenceThreshold")
        return results
    }
    
    /**
     * Applique Non-Maximum Suppression pour éliminer les boîtes redondantes
     */
    private fun applyNMS(detections: List<Output0>, iouThreshold: Float): List<Output0> {
        if (detections.isEmpty()) return emptyList()
        
        // Trier par score décroissant
        val sortedDetections = detections.sortedByDescending { it.cnf }
        val selectedDetections = mutableListOf<Output0>()
        val visited = BooleanArray(sortedDetections.size) { false }
        
        for (i in sortedDetections.indices) {
            if (visited[i]) continue
            
            selectedDetections.add(sortedDetections[i])
            visited[i] = true
            
            val boxA = floatArrayOf(sortedDetections[i].y1, sortedDetections[i].x1, 
                                   sortedDetections[i].y2, sortedDetections[i].x2)
            
            for (j in i + 1 until sortedDetections.size) {
                if (visited[j]) continue
                
                val boxB = floatArrayOf(sortedDetections[j].y1, sortedDetections[j].x1, 
                                       sortedDetections[j].y2, sortedDetections[j].x2)
                
                if (calculateIoU(boxA, boxB) > iouThreshold) {
                    visited[j] = true
                }
            }
        }
        
        return selectedDetections
    }
    
    /**
     * Calcule l'IoU (Intersection over Union) entre deux boîtes
     * Format de la boîte: [ymin, xmin, ymax, xmax]
     */
    private fun calculateIoU(boxA: FloatArray, boxB: FloatArray): Float {
        // Coordonnées d'intersection
        val yminA = boxA[0]
        val xminA = boxA[1]
        val ymaxA = boxA[2]
        val xmaxA = boxA[3]
        
        val yminB = boxB[0]
        val xminB = boxB[1]
        val ymaxB = boxB[2]
        val xmaxB = boxB[3]
        
        // Points d'intersection
        val xminIntersection = max(xminA, xminB)
        val yminIntersection = max(yminA, yminB)
        val xmaxIntersection = min(xmaxA, xmaxB)
        val ymaxIntersection = min(ymaxA, ymaxB)
        
        // Si pas d'intersection
        if (xmaxIntersection < xminIntersection || ymaxIntersection < yminIntersection) {
            return 0f
        }
        
        // Calcul des aires
        val intersectionArea = (xmaxIntersection - xminIntersection) * (ymaxIntersection - yminIntersection)
        val boxAArea = (xmaxA - xminA) * (ymaxA - yminA)
        val boxBArea = (xmaxB - xminB) * (ymaxB - yminB)
        
        // IoU = intersection / union
        val unionArea = boxAArea + boxBArea - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
}
