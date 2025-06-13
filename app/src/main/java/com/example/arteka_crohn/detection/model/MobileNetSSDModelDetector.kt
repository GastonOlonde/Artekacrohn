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
 * Implémentation spécifique pour les modèles MobileNet SSD
 * Supporte les deux formats:
 * - Single-output: Un seul tenseur contenant toutes les détections
 * - Multi-output: Plusieurs tenseurs (locations, classes, scores, count)
 */
class MobileNetSSDModelDetector(
    private val onMessageCallback: (String) -> Unit
) : BaseModelDetector() {
    
    private val TAG = "MobileNetSSDDetector"
    
    // Format de sortie du modèle (déterminé automatiquement)
    private var isMultiOutputFormat = false
    
    override fun initialize(context: Context, modelPath: String, labelPath: String?) {
        onMessage = onMessageCallback
        super.initialize(context, modelPath, labelPath)
        
        // Déterminer le format de sortie basé sur le nombre de tenseurs
        isMultiOutputFormat = outputShapes.size == 4
        
        Log.d(TAG, "Initialized MobileNet SSD model in ${if(isMultiOutputFormat) "multi" else "single"}-output format")
    }
    
    override fun runInference(inputBuffers: Array<ByteBuffer>): Any {
        val interp = interpreter ?: throw IllegalStateException("Interpreter not initialized")
        
        if (isMultiOutputFormat) {
            // Format multi-output (4 tenseurs: locations, classes, scores, count)
            // Créer les buffers de sortie pour chaque tenseur
            val outputBuffers = outputShapes.mapIndexed { index, shape ->
                ByteBuffer.allocateDirect(shape.reduce { acc, i -> acc * i } * 4).apply {
                    order(ByteOrder.nativeOrder())
                }.also { it.rewind() }
            }
            
            // Mapper les buffers aux indices
            val outputs = mutableMapOf<Int, Any>()
            outputBuffers.forEachIndexed { index, buffer ->
                outputs[index] = buffer
            }
            
            // Exécuter l'inférence
            interp.runForMultipleInputsOutputs(inputBuffers, outputs)
            
            // Créer une structure de données pour contenir tous les résultats
            val results = MultiOutputResult(
                locations = convertBufferToFloatArray(outputBuffers[0]),
                classes = convertBufferToFloatArray(outputBuffers[1]),
                scores = convertBufferToFloatArray(outputBuffers[2]),
                numDetections = convertBufferToFloatArray(outputBuffers[3])[0].toInt()
            )
            
            return results
        } else {
            // Format single-output (un seul tenseur)
            val outputShape = outputShapes[0]
            val outputBuffer = ByteBuffer.allocateDirect(
                outputShape.reduce { acc, i -> acc * i } * 4 // 4 bytes par float
            ).apply {
                order(ByteOrder.nativeOrder())
            }
            
            val outputs = mapOf<Int, Any>(
                0 to outputBuffer.rewind()
            )
            
            // Exécuter l'inférence
            interp.runForMultipleInputsOutputs(inputBuffers, outputs)
            
            // Convertir le buffer en tableau
            outputBuffer.rewind()
            val outputArray = FloatArray(outputBuffer.capacity() / 4)
            
            for (i in outputArray.indices) {
                outputArray[i] = outputBuffer.getFloat()
            }
            
            return outputArray
        }
    }
    
    override fun processOutput(rawOutput: Any, confidenceThreshold: Float): List<Output0> {
        return if (isMultiOutputFormat) {
            if (rawOutput !is MultiOutputResult) {
                throw IllegalArgumentException("Expected MultiOutputResult for multi-output format")
            }
            processMultiOutputFormat(rawOutput, confidenceThreshold)
        } else {
            if (rawOutput !is FloatArray) {
                throw IllegalArgumentException("Expected FloatArray for single-output format")
            }
            processSingleOutputFormat(rawOutput, confidenceThreshold)
        }
    }
    
    /**
     * Traite les sorties au format multi-output (locations, classes, scores, count)
     */
    private fun processMultiOutputFormat(result: MultiOutputResult, confidenceThreshold: Float): List<Output0> {
        val detections = mutableListOf<Output0>()
        val numDetections = result.numDetections
        
        // Extraire les détections valides
        for (i in 0 until numDetections) {
            val score = result.scores[i]
            
            // Ignorer les détections sous le seuil de confiance
            if (score < confidenceThreshold) continue
            
            // Class ID (0-based)
            val classId = result.classes[i].toInt()
            
            // Coordonnées normalisées [ymin, xmin, ymax, xmax] entre 0 et 1
            val ymin = max(0f, min(1f, result.locations[i * 4]))
            val xmin = max(0f, min(1f, result.locations[i * 4 + 1]))
            val ymax = max(0f, min(1f, result.locations[i * 4 + 2]))
            val xmax = max(0f, min(1f, result.locations[i * 4 + 3]))
            
            // Calcul du centre et des dimensions
            val cx = (xmin + xmax) / 2
            val cy = (ymin + ymax) / 2
            val w = xmax - xmin
            val h = ymax - ymin
            
            // Nom de la classe
            val className = if (classId >= 0 && classId < labels.size) labels[classId] else "Unknown"
            
            detections.add(
                Output0(
                    x1 = xmin,
                    y1 = ymin,
                    x2 = xmax,
                    y2 = ymax,
                    cx = cx,
                    cy = cy,
                    w = w,
                    h = h,
                    cnf = score,
                    cls = classId,
                    clsName = className,
                    maskWeight = emptyList()
                )
            )
        }
        
        // Appliquer NMS (Non-Maximum Suppression) pour éliminer les boîtes redondantes
        return applyNMS(detections, DetectionConfig.IOU_THRESHOLD)
    }
    
    /**
     * Traite les sorties au format single-output (un seul tenseur)
     * Format typique: [1, num_detections, num_values_per_detection]
     * où num_values_per_detection = 4 (box) + 1 (score) + 1 (class) ou plus
     */
    private fun processSingleOutputFormat(outputArray: FloatArray, confidenceThreshold: Float): List<Output0> {
        // Détermine la structure du tenseur de sortie
        val outputShape = outputShapes[0]
        
        if (outputShape.size < 3) {
            Log.e(TAG, "Invalid output shape for single-output format: ${outputShape.contentToString()}")
            return emptyList()
        }
        
        val numDetections = outputShape[1]
        val valuesPerDetection = outputShape[2]
        
        val detections = mutableListOf<Output0>()
        
        // Pour chaque détection
        for (i in 0 until numDetections) {
            val offset = i * valuesPerDetection
            
            // Déterminer l'emplacement des composants dans la sortie
            // Format standard: [y_min, x_min, y_max, x_max, score, class_id]
            val ymin = outputArray[offset]
            val xmin = outputArray[offset + 1]
            val ymax = outputArray[offset + 2]
            val xmax = outputArray[offset + 3]
            val score = outputArray[offset + 4]
            val classId = outputArray[offset + 5].toInt()
            
            // Ignorer les détections sous le seuil de confiance
            if (score < confidenceThreshold) continue
            
            // Vérifier que les coordonnées sont valides
            if (ymin < 0f || xmin < 0f || ymax > 1f || xmax > 1f) continue
            
            // Calcul du centre et des dimensions
            val cx = (xmin + xmax) / 2
            val cy = (ymin + ymax) / 2
            val w = xmax - xmin
            val h = ymax - ymin
            
            // Nom de la classe
            val className = if (classId >= 0 && classId < labels.size) labels[classId] else "Unknown"
            
            detections.add(
                Output0(
                    x1 = xmin,
                    y1 = ymin,
                    x2 = xmax,
                    y2 = ymax,
                    cx = cx,
                    cy = cy,
                    w = w,
                    h = h,
                    cnf = score,
                    cls = classId,
                    clsName = className,
                    maskWeight = emptyList()
                )
            )
        }
        
        // Appliquer NMS (Non-Maximum Suppression) pour éliminer les boîtes redondantes
        return applyNMS(detections, DetectionConfig.IOU_THRESHOLD)
    }
    
    /**
     * Applique Non-Maximum Suppression pour éliminer les boîtes redondantes
     */
    private fun applyNMS(detections: List<Output0>, iouThreshold: Float): List<Output0> {
        if (detections.isEmpty()) return detections
        
        // Trier les détections par score (confiance) décroissant
        val sortedDetections = detections.sortedByDescending { it.cnf }
        val visited = BooleanArray(sortedDetections.size) { false }
        val selectedDetections = mutableListOf<Output0>()
        
        // Pour chaque détection par ordre de confiance décroissant
        for (i in sortedDetections.indices) {
            if (visited[i]) continue
            
            // Ajouter cette détection à la sélection
            selectedDetections.add(sortedDetections[i])
            
            // Marquer comme visitée et comparer avec toutes les détections suivantes
            visited[i] = true
            
            // Pour le calcul de l'IoU, on extrait les coordonnées de la boîte
            val boxA = floatArrayOf(sortedDetections[i].y1, sortedDetections[i].x1, sortedDetections[i].y2, sortedDetections[i].x2)
            
            // Comparer avec toutes les détections suivantes
            for (j in i + 1 until sortedDetections.size) {
                if (visited[j]) continue
                
                val boxB = floatArrayOf(sortedDetections[j].y1, sortedDetections[j].x1, sortedDetections[j].y2, sortedDetections[j].x2)
                
                // Si les deux boîtes se chevauchent suffisamment et sont de la même classe,
                // supprimer la boîte avec le score le plus faible
                if (calculateIoU(boxA, boxB) > iouThreshold && 
                    sortedDetections[i].clsName == sortedDetections[j].clsName) {
                    visited[j] = true
                }
            }
        }
        
        return selectedDetections
    }
    
    /**
     * Convertit un ByteBuffer en FloatArray
     */
    private fun convertBufferToFloatArray(buffer: ByteBuffer): FloatArray {
        buffer.rewind()
        val array = FloatArray(buffer.capacity() / 4)
        
        for (i in array.indices) {
            array[i] = buffer.getFloat()
        }
        
        return array
    }
    
    /**
     * Calcule l'IoU (Intersection over Union) entre deux boîtes
     * Format de la boîte: [ymin, xmin, ymax, xmax]
     */
    private fun calculateIoU(boxA: FloatArray, boxB: FloatArray): Float {
        // Calculer les coordonnées d'intersection
        val yminA = boxA[0]
        val xminA = boxA[1]
        val ymaxA = boxA[2]
        val xmaxA = boxA[3]
        
        val yminB = boxB[0]
        val xminB = boxB[1]
        val ymaxB = boxB[2]
        val xmaxB = boxB[3]
        
        // Calculer les points de l'intersection
        val xminIntersection = max(xminA, xminB)
        val yminIntersection = max(yminA, yminB)
        val xmaxIntersection = min(xmaxA, xmaxB)
        val ymaxIntersection = min(ymaxA, ymaxB)
        
        // Si pas d'intersection, IoU = 0
        if (xmaxIntersection < xminIntersection || ymaxIntersection < yminIntersection) {
            return 0f
        }
        
        // Calculer les aires
        val intersectionArea = (xmaxIntersection - xminIntersection) * (ymaxIntersection - yminIntersection)
        val boxAArea = (xmaxA - xminA) * (ymaxA - yminA)
        val boxBArea = (xmaxB - xminB) * (ymaxB - yminB)
        
        // Calculer l'union et l'IoU
        val unionArea = boxAArea + boxBArea - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    /**
     * Classe pour encapsuler les résultats du format multi-output
     */
    data class MultiOutputResult(
        val locations: FloatArray, // [ymin, xmin, ymax, xmax] pour chaque détection
        val classes: FloatArray,   // ID de classe pour chaque détection
        val scores: FloatArray,    // Score de confiance pour chaque détection
        val numDetections: Int     // Nombre de détections valides
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            
            other as MultiOutputResult
            
            if (!locations.contentEquals(other.locations)) return false
            if (!classes.contentEquals(other.classes)) return false
            if (!scores.contentEquals(other.scores)) return false
            if (numDetections != other.numDetections) return false
            
            return true
        }
        
        override fun hashCode(): Int {
            var result = locations.contentHashCode()
            result = 31 * result + classes.contentHashCode()
            result = 31 * result + scores.contentHashCode()
            result = 31 * result + numDetections
            return result
        }
    }
}
