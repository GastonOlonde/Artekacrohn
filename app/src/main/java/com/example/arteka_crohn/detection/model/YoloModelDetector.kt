package com.example.arteka_crohn.detection.model

import android.content.Context
import android.util.Log
import com.example.arteka_crohn.Output0
import com.example.arteka_crohn.detection.config.DetectionConfig
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Implémentation spécifique pour les modèles YOLO V8
 */
class YoloModelDetector(
    private val onMessageCallback: (String) -> Unit
) : BaseModelDetector() {
    
    private val TAG = "YoloModelDetector"
    
    // Configuration spécifique YOLO
    private var numClasses = 80 // Par défaut (COCO)
    
    override fun initialize(context: Context, modelPath: String, labelPath: String?) {
        onMessage = onMessageCallback
        super.initialize(context, modelPath, labelPath)
        
        // Déterminer le nombre de classes basé sur les dimensions de sortie
        if (outputShapes.isNotEmpty()) {
            // Pour YOLOv8, la forme de sortie est [1, 84, 8400] pour 80 classes (COCO)
            // où 84 = 4 (box coords) + 80 (classes)
            val outputTensor = outputShapes[0]
            if (outputTensor.size >= 2) {
                numClasses = outputTensor[1] - 4
                Log.d(TAG, "Detected $numClasses classes for YOLO model")
            }
        }
    }
    
    override fun runInference(inputBuffers: Array<ByteBuffer>): Any {
        // Pour YOLOv8, on a un seul buffer de sortie avec les détections
        val outputBuffer = ByteBuffer.allocateDirect(
            (outputShapes[0].reduce { acc, i -> acc * i } * 4) // 4 bytes par float
        ).apply {
            order(ByteOrder.nativeOrder())
        }
        
        val outputs = mapOf<Int, Any>(
            0 to outputBuffer.rewind()
        )
        
        // Exécution de l'inférence
        interpreter?.runForMultipleInputsOutputs(inputBuffers, outputs)
            ?: throw IllegalStateException("Interpreter not initialized")
            
        // Conversion du ByteBuffer en FloatArray pour le post-traitement
        outputBuffer.rewind()
        val outputArray = FloatArray(outputBuffer.capacity() / 4)
        
        for (i in outputArray.indices) {
            outputArray[i] = outputBuffer.getFloat()
        }
        
        return outputArray
    }
    
    override fun processOutput(rawOutput: Any, confidenceThreshold: Float): List<Output0> {
        if (rawOutput !is FloatArray) {
            throw IllegalArgumentException("Raw output must be FloatArray")
        }
        
        // Formatage des sorties YOLOv8
        return processYoloV8Output(rawOutput, confidenceThreshold)
    }
    
    /**
     * Traitement spécifique de la sortie YOLOv8
     * Format YOLOv8: [1, num_classes + 4, num_anchors]
     * Les 4 premiers éléments sont [x, y, width, height] et les suivants sont les scores de classes
     */
    private fun processYoloV8Output(outputArray: FloatArray, confidenceThreshold: Float): List<Output0> {
        // Récupération des dimensions de sortie
        if (outputShapes.isEmpty()) return emptyList()
        val outputShape = outputShapes[0]
        
        // Détecter le format du modèle YOLOv8
        return if (outputShape.size >= 3 && outputShape[1] == 5) {
            // Format YOLOv8-640 spécifique [1, 5, 8400]
            processYoloV8ChannelFormat(outputArray, confidenceThreshold)
        } else {
            // Format YOLOv8 standard [1, 84, 8400]
            processYoloV8StandardFormat(outputArray, confidenceThreshold)
        }
    }

    /**
     * Traite la sortie YOLOv8 au format standard [1, 84, 8400]
     * où chaque colonne contient [x, y, w, h, class1, class2, ...]
     */
    private fun processYoloV8StandardFormat(outputArray: FloatArray, confidenceThreshold: Float): List<Output0> {
        if (outputShapes.isEmpty()) return emptyList()
        val outputShape = outputShapes[0]
        
        // Pour YOLOv8 typique: [1, 84, 8400]
        val rows = outputShape[1] // 84 = 4 (coords) + 80 (classes)
        val cols = outputShape[2] // 8400 = nombre d'anchors
        
        // Liste des détections finales
        val detections = mutableListOf<Output0>()
        
        // Pour chaque détection (colonne)
        for (i in 0 until cols) {
            // Extraire les coordonnées (centre x, centre y, largeur, hauteur)
            val cx = outputArray[0 + i * rows]
            val cy = outputArray[1 + i * rows]
            val w = outputArray[2 + i * rows]
            val h = outputArray[3 + i * rows]
            
            // Calculer les coordonnées des coins du rectangle
            val xmin = max(0f, min(1f, cx - w / 2))
            val ymin = max(0f, min(1f, cy - h / 2))
            val xmax = max(0f, min(1f, cx + w / 2))
            val ymax = max(0f, min(1f, cy + h / 2))
            
            // Trouver la classe avec le score maximum
            var bestScore = 0f
            var bestClass = 0
            var bestClassName = "unknown"
            
            // Parcourir les scores des classes (indices 4 à rows-1)
            for (j in 4 until rows) {
                val score = outputArray[j + i * rows]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = j - 4 // L'indice de classe commence à 0
                    bestClassName = if (bestClass < labels.size) labels[bestClass] else "unknown"
                }
            }
            
            // Filtrer par seuil de confiance
            if (bestScore < confidenceThreshold) continue
            
            // Créer l'objet de détection (Output0)
            val detection = Output0(
                x1 = xmin,
                y1 = ymin,
                x2 = xmax,
                y2 = ymax,
                cx = cx,
                cy = cy,
                w = w,
                h = h,
                cnf = bestScore,
                cls = bestClass,
                clsName = bestClassName,
                maskWeight = emptyList()
            )
            
            detections.add(detection)
        }
        
        // Appliquer la suppression des non-maximums
        return applyNMS(detections, DetectionConfig.IOU_THRESHOLD)
    }

    /**
     * Traite la sortie YOLOv8 au format par canal [1, 5, 8400]
     * où les données sont organisées par canal (x, y, w, h, conf)
     * Format spécifique pour YOLOv8-640
     */
    private fun processYoloV8ChannelFormat(outputArray: FloatArray, confidenceThreshold: Float): List<Output0> {
        if (outputShapes.isEmpty()) return emptyList()
        val outputShape = outputShapes[0]
        
        // Pour YOLOv8-640: [1, 5, 8400]
        val numChannels = outputShape[1] // 5 (x, y, w, h, conf)
        val numBoxes = outputShape[2] // 8400 = nombre de boîtes
        
        // Liste des détections finales
        val detections = mutableListOf<Output0>()
        
        // Format [1, 5, 8400] où les données sont organisées par canal
        for (i in 0 until numBoxes) {
            // Pour chaque boîte i, accéder aux indices par canal:
            // x = i
            // y = i + numBoxes
            // w = i + 2*numBoxes
            // h = i + 3*numBoxes
            // conf = i + 4*numBoxes
            val x = outputArray[i]
            val y = outputArray[i + numBoxes]
            val w = outputArray[i + 2 * numBoxes]
            val h = outputArray[i + 3 * numBoxes]
            val confidence = min(outputArray[i + 4 * numBoxes], 1.0f)  // Limiter à 1.0
            
            // Ignorer les détections avec confiance trop faible
            if (confidence < confidenceThreshold) continue
            
            // Convertir les coordonnées centrées en coordonnées de boîte
            val x1 = max(0f, min(1f, x - w / 2))
            val y1 = max(0f, min(1f, y - h / 2))
            val x2 = max(0f, min(1f, x + w / 2))
            val y2 = max(0f, min(1f, y + h / 2))
            
            // Pour ce format, on a une seule classe par défaut (anomalie)
            val detectedClass = 0
            val className = labels.getOrElse(detectedClass) { "Anomaly" }
            
            // Ajouter le résultat à la liste
            detections.add(
                Output0(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    cx = x,
                    cy = y,
                    w = w,
                    h = h,
                    cnf = confidence,
                    cls = detectedClass,
                    clsName = className,
                    maskWeight = emptyList()
                )
            )
        }
        
        // Appliquer la suppression des non-maximums
        return applyNMS(detections, DetectionConfig.IOU_THRESHOLD)
    }

    /**
     * Applique l'algorithme de suppression des non-maximums (NMS)
     * pour éliminer les boîtes redondantes
     */
    private fun applyNMS(boxes: List<Output0>, iouThreshold: Float): List<Output0> {
        // Si pas de détections, retourner une liste vide
        if (boxes.isEmpty()) return emptyList()
        
        // Trier les détections par score décroissant
        val sortedDetections = boxes.sortedByDescending { it.cnf }
        val selectedDetections = mutableListOf<Output0>()
        val visited = BooleanArray(sortedDetections.size) { false }
        
        // Pour chaque détection non visitée
        for (i in sortedDetections.indices) {
            // Si cette détection a déjà été "supprimée", continuer à la suivante
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
        
        Log.d(TAG, "NMS reduced detections from ${boxes.size} to ${selectedDetections.size}")
        return selectedDetections
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
}
