package com.example.arteka_crohn.detection.postprocessing

import android.util.Log
import com.example.arteka_crohn.Output0
import com.example.arteka_crohn.detection.config.DetectionConfig
import com.example.arteka_crohn.detection.model.TensorFlowLiteDetectionModel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Implémentation de l'interface DetectionPostprocessor pour les modèles de détection d'objets
 * Supporte différents formats de sortie courants (SSD, YOLO, etc.)
 */
class ObjectDetectionPostprocessor(private val model: TensorFlowLiteDetectionModel) : DetectionPostprocessor {

    /**
     * Traite les résultats bruts de détection pour produire une liste d'objets détectés
     * @param rawOutputs Les sorties brutes du modèle de détection
     * @return Une liste d'objets détectés
     */
    override fun processDetections(rawOutputs: FloatArray): List<Output0> {
        val outputShape = model.getOutputShape()
        
        // Format spécifique YOLOv8-640 : [1, 5, 8400]
        // où 5 = [x, y, w, h, confidence] et 8400 = nombre de boîtes
        if (outputShape.size == 3 && outputShape[1] == 5 && outputShape[2] == 8400) {
            return processYOLOv8_640Output(rawOutputs)
        }
        
        // Autres formats (code existant)
        return when {
            // Format typique SSD/MobileNet: [1, numDetections, 4 + 1 + numClasses]
            outputShape.size == 3 && outputShape[2] > 5 -> processSSDOutput(rawOutputs, outputShape)
            
            // Format YOLO: [1, gridH, gridW, (5 + numClasses) * numAnchors]
            // ou [1, numBoxes, 5 + numClasses] pour YOLOv5/v8
            outputShape.size == 4 || (outputShape.size == 3 && outputShape[1] > 0) -> 
                processYOLOOutput(rawOutputs, outputShape)
            
            // Format non reconnu
            else -> {
                Log.w("DetectionPostprocessor", "Format de sortie non reconnu: ${outputShape.joinToString()}")
                processSSDOutput(rawOutputs, outputShape) // Essayer le format SSD par défaut
            }
        }
    }
    
    /**
     * Traite les sorties au format SSD/MobileNet
     */
    private fun processSSDOutput(outputs: FloatArray, shape: IntArray): List<Output0> {
        val results = mutableListOf<Output0>()
        val numDetections = shape[1]
        val numValues = shape[2]
        val numClasses = numValues - 5 // Soustraire les 4 coordonnées de boîte et 1 score
        
        val labels = model.getLabels()
        
        for (i in 0 until numDetections) {
            val offset = i * numValues
            
            // Extraire le score de confiance
            val confidence = outputs[offset + 4]
            
            // Ignorer les détections avec une confiance inférieure au seuil
            if (confidence < DetectionConfig.CONFIDENCE_THRESHOLD) continue
            
            // Trouver la classe avec le score maximum
            var maxClassScore = 0f
            var detectedClass = 0
            
            for (c in 0 until numClasses) {
                val score = outputs[offset + 5 + c]
                if (score > maxClassScore) {
                    maxClassScore = score
                    detectedClass = c
                }
            }
            
            // Extraire les coordonnées de la boîte (format [y1, x1, y2, x2] ou [x1, y1, x2, y2])
            // Nous supposons ici le format [x1, y1, x2, y2]
            val x1 = outputs[offset]
            val y1 = outputs[offset + 1]
            val x2 = outputs[offset + 2]
            val y2 = outputs[offset + 3]
            
            // Calculer le centre et les dimensions
            val cx = (x1 + x2) / 2
            val cy = (y1 + y2) / 2
            val w = x2 - x1
            val h = y2 - y1
            
            // Créer l'objet Output0 (sans masque car c'est un modèle de détection simple)
            val className = if (detectedClass < labels.size) labels[detectedClass] else "Unknown"
            
            results.add(
                Output0(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    cx = cx,
                    cy = cy,
                    w = w,
                    h = h,
                    cnf = confidence,
                    cls = detectedClass,
                    clsName = className,
                    maskWeight = emptyList() // Pas de masque pour la détection simple
                )
            )
        }
        
        // Appliquer la suppression des non-maximums (NMS)
        return applyNMS(results, DetectionConfig.IOU_THRESHOLD)
    }
    
    /**
     * Traite les sorties au format YOLO
     * Supporte les formats YOLOv3, YOLOv4, YOLOv5 et YOLOv8
     */
    private fun processYOLOOutput(outputs: FloatArray, shape: IntArray): List<Output0> {
        val results = mutableListOf<Output0>()
        val labels = model.getLabels()
        val numClasses = labels.size
        
        // Déterminer la version de YOLO en fonction de la forme de sortie
        when {
            // YOLOv8 format: [1, numBoxes, 4 + 1 + numClasses] 
            // où numBoxes est le nombre total de boîtes prédites
            shape.size == 3 -> {
                processYOLOv8Output(outputs, shape, numClasses, labels, results)
            }
            
            // YOLOv3/v4/v5 format: [1, gridH, gridW, (5 + numClasses) * numAnchors]
            shape.size == 4 -> {
                processYOLOv3v4v5Output(outputs, shape, numClasses, labels, results)
            }
            
            else -> {
                Log.w("DetectionPostprocessor", "Format YOLO non reconnu: ${shape.joinToString()}")
            }
        }
        
        // Appliquer la suppression des non-maximums (NMS)
        return applyNMS(results, DetectionConfig.IOU_THRESHOLD)
    }
    
    /**
     * Traite les sorties au format YOLOv8
     * Format: [1, numBoxes, 4 + 1 + numClasses]
     */
    private fun processYOLOv8Output(
        outputs: FloatArray,
        shape: IntArray,
        numClasses: Int,
        labels: List<String>,
        results: MutableList<Output0>
    ) {
        val numBoxes = shape[1]
        val boxSize = shape[2]
        
        for (i in 0 until numBoxes) {
            val offset = i * boxSize
            
            // Dans YOLOv8, les 4 premiers éléments sont les coordonnées normalisées [x, y, w, h]
            val x = outputs[offset]
            val y = outputs[offset + 1]
            val w = outputs[offset + 2]
            val h = outputs[offset + 3]
            
            // Calculer les coordonnées des coins de la boîte
            val x1 = x - w / 2
            val y1 = y - h / 2
            val x2 = x + w / 2
            val y2 = y + h / 2
            
            // Trouver la classe avec le score maximum
            var maxClassScore = 0f
            var detectedClass = 0
            
            for (c in 0 until numClasses) {
                val score = outputs[offset + 4 + c]
                if (score > maxClassScore) {
                    maxClassScore = score
                    detectedClass = c
                }
            }
            
            // Vérifier si la confiance est suffisante
            if (maxClassScore < DetectionConfig.CONFIDENCE_THRESHOLD) continue
            
            // Créer l'objet Output0
            val className = if (detectedClass < labels.size) labels[detectedClass] else "Unknown"
            
            results.add(
                Output0(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    cx = x,
                    cy = y,
                    w = w,
                    h = h,
                    cnf = maxClassScore,
                    cls = detectedClass,
                    clsName = className,
                    maskWeight = emptyList()
                )
            )
        }
    }
    
    /**
     * Traite les sorties au format YOLOv3/v4/v5
     * Format: [1, gridH, gridW, (5 + numClasses) * numAnchors]
     */
    private fun processYOLOv3v4v5Output(
        outputs: FloatArray,
        shape: IntArray,
        numClasses: Int,
        labels: List<String>,
        results: MutableList<Output0>
    ) {
        val gridHeight = shape[1]
        val gridWidth = shape[2]
        val channels = shape[3]
        
        // Calculer le nombre d'ancres
        val boxAttrs = 5 + numClasses // 5 = x, y, w, h, confidence
        val numAnchors = channels / boxAttrs
        
        // Récupérer les dimensions d'ancres (à adapter selon votre modèle)
        // Ces valeurs sont généralement spécifiques au modèle et devraient être stockées dans DetectionConfig
        val anchors = getAnchors()
        
        // Parcourir chaque cellule de la grille
        for (anchorIdx in 0 until numAnchors) {
            for (gridY in 0 until gridHeight) {
                for (gridX in 0 until gridWidth) {
                    // Calculer l'indice de base pour cette cellule et cette ancre
                    val baseIdx = (gridY * gridWidth + gridX) * channels + anchorIdx * boxAttrs
                    
                    // Extraire la confiance de l'objet
                    val objectConfidence = sigmoid(outputs[baseIdx + 4])
                    
                    // Ignorer les détections avec une confiance faible
                    if (objectConfidence < DetectionConfig.CONFIDENCE_THRESHOLD) continue
                    
                    // Trouver la classe avec le score maximum
                    var maxClassScore = 0f
                    var detectedClass = 0
                    
                    for (c in 0 until numClasses) {
                        val classScore = sigmoid(outputs[baseIdx + 5 + c])
                        if (classScore > maxClassScore) {
                            maxClassScore = classScore
                            detectedClass = c
                        }
                    }
                    
                    // Calculer le score final
                    val confidence = objectConfidence * maxClassScore
                    
                    // Ignorer les détections avec un score final faible
                    if (confidence < DetectionConfig.CONFIDENCE_THRESHOLD) continue
                    
                    // Extraire les coordonnées prédites
                    val x = (sigmoid(outputs[baseIdx]) + gridX) / gridWidth
                    val y = (sigmoid(outputs[baseIdx + 1]) + gridY) / gridHeight
                    
                    // Appliquer les dimensions d'ancre
                    val anchorW = anchors[anchorIdx * 2]
                    val anchorH = anchors[anchorIdx * 2 + 1]
                    
                    val w = exp(outputs[baseIdx + 2]) * anchorW / DetectionConfig.INPUT_SIZE
                    val h = exp(outputs[baseIdx + 3]) * anchorH / DetectionConfig.INPUT_SIZE
                    
                    // Calculer les coordonnées des coins (normalisées entre 0 et 1)
                    val x1 = max(0f, x - w / 2)
                    val y1 = max(0f, y - h / 2)
                    val x2 = min(1f, x + w / 2)
                    val y2 = min(1f, y + h / 2)
                    
                    // Créer l'objet Output0
                    val className = if (detectedClass < labels.size) labels[detectedClass] else "Unknown"
                    
                    results.add(
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
            }
        }
    }
    
    /**
 * Traite les sorties au format YOLOv8-640 [1, 5, 8400]
 * Format spécifique pour le modèle avec entrée 640x640
 */
private fun processYOLOv8_640Output(outputs: FloatArray): List<Output0> {
    val results = mutableListOf<Output0>()
    val numBoxes = 8400
    val labels = model.getLabels()
    
    // Log des valeurs brutes pour le débogage
    Log.d("DetectionProcessor", "Traitement de ${outputs.size} valeurs pour YOLOv8-640")
    
    // Vérifier les valeurs max pour chaque composante
    var maxX = 0f
    var maxY = 0f
    var maxW = 0f
    var maxH = 0f
    var maxConf = 0f
    
    // Format [1, 5, 8400] où les données sont organisées par canal
    // Réorganiser les données pour accéder facilement à chaque boîte
    for (i in 0 until numBoxes) {
        // Dans le format [1, 5, 8400], les données sont organisées par canal
        // Pour chaque boîte i, nous devons accéder aux indices:
        // x = i
        // y = i + numBoxes
        // w = i + 2*numBoxes
        // h = i + 3*numBoxes
        // conf = i + 4*numBoxes
        val x = outputs[i]
        val y = outputs[i + numBoxes]
        val w = outputs[i + 2 * numBoxes]
        val h = outputs[i + 3 * numBoxes]
        val confidence = min(outputs[i + 4 * numBoxes], 1.0f)  // Limiter à 1.0
        
        maxX = max(maxX, x)
        maxY = max(maxY, y)
        maxW = max(maxW, w)
        maxH = max(maxH, h)
        maxConf = max(maxConf, confidence)
        
        // Log des 5 premières et dernières détections pour vérifier
        if (i < 5 || i > numBoxes - 5) {
            Log.d("DetectionProcessor", "Box $i: x=$x, y=$y, w=$w, h=$h, conf=$confidence")
        }
        
        // Ignorer les détections avec une confiance inférieure au seuil
        if (confidence < DetectionConfig.CONFIDENCE_THRESHOLD) continue
        
        // Convertir les coordonnées centrées en coordonnées de boîte
        val x1 = (x - w / 2)
        val y1 = (y - h / 2)
        val x2 = (x + w / 2)
        val y2 = (y + h / 2)
        
        // Limiter les coordonnées entre 0 et 1
        val boxX1 = max(0f, min(1f, x1))
        val boxY1 = max(0f, min(1f, y1))
        val boxX2 = max(0f, min(1f, x2))
        val boxY2 = max(0f, min(1f, y2))
        
        // Calculer le centre et les dimensions normalisées
        val cx = (boxX1 + boxX2) / 2
        val cy = (boxY1 + boxY2) / 2
        val width = boxX2 - boxX1
        val height = boxY2 - boxY1
        
        // Déterminer la classe (pour l'instant, on utilise une classe par défaut)
        val detectedClass = 0
        val className = labels.getOrElse(detectedClass) { "Anomaly" }
        
        // Ajouter le résultat à la liste
        results.add(
            Output0(
                x1 = boxX1,
                y1 = boxY1,
                x2 = boxX2,
                y2 = boxY2,
                cx = cx,
                cy = cy,
                w = width,
                h = height,
                cnf = confidence,
                cls = detectedClass,
                clsName = className,
                maskWeight = emptyList()
            )
        )
    }
    
    Log.d("DetectionProcessor", "Valeurs max: x=$maxX, y=$maxY, w=$maxW, h=$maxH, conf=$maxConf")
    Log.d("DetectionProcessor", "Seuil de confiance: ${DetectionConfig.CONFIDENCE_THRESHOLD}")
    Log.d("DetectionProcessor", "Détections trouvées: ${results.size}")
    
    // Appliquer la suppression des non-maximums pour éliminer les boîtes redondantes
    return applyNMS(results, DetectionConfig.IOU_THRESHOLD)
}
    
    /**
     * Récupère les dimensions d'ancres pour le modèle YOLO
     * À adapter selon le modèle utilisé
     */
    private fun getAnchors(): FloatArray {
        // Ces valeurs sont généralement spécifiques au modèle
        // Pour YOLOv3/v4 sur COCO, les ancres typiques sont:
        return when (model.getModelName()) {
            "yolov3.tflite" -> floatArrayOf(
                10f, 13f, 16f, 30f, 33f, 23f,  // Petites ancres
                30f, 61f, 62f, 45f, 59f, 119f, // Moyennes ancres
                116f, 90f, 156f, 198f, 373f, 326f // Grandes ancres
            )
            "yolov4.tflite" -> floatArrayOf(
                12f, 16f, 19f, 36f, 40f, 28f,
                36f, 75f, 76f, 55f, 72f, 146f,
                142f, 110f, 192f, 243f, 459f, 401f
            )
            "yolov5s.tflite" -> floatArrayOf(
                10f, 13f, 16f, 30f, 33f, 23f,
                30f, 61f, 62f, 45f, 59f, 119f,
                116f, 90f, 156f, 198f, 373f, 326f
            )
            // Ajouter d'autres modèles selon vos besoins
            else -> {
                // Valeurs par défaut pour YOLOv3 sur COCO
                Log.w("DetectionPostprocessor", "Utilisation d'ancres par défaut pour le modèle: ${model.getModelName()}")
                floatArrayOf(
                    10f, 13f, 16f, 30f, 33f, 23f,
                    30f, 61f, 62f, 45f, 59f, 119f,
                    116f, 90f, 156f, 198f, 373f, 326f
                )
            }
        }
    }
    
    /**
     * Fonction sigmoïde pour les calculs YOLO
     */
    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + exp(-x))
    }
    
    /**
     * Applique la suppression des non-maximums (NMS) pour éliminer les boîtes redondantes
     */
    private fun applyNMS(boxes: List<Output0>, iouThreshold: Float): List<Output0> {
        if (boxes.isEmpty()) return emptyList()
        
        // Trier les boîtes par score de confiance décroissant
        val sortedBoxes = boxes.sortedByDescending { it.cnf }
        val selectedBoxes = mutableListOf<Output0>()
        val indexesToRemove = mutableSetOf<Int>()
        
        for (i in sortedBoxes.indices) {
            if (i in indexesToRemove) continue
            
            selectedBoxes.add(sortedBoxes[i])
            
            for (j in i + 1 until sortedBoxes.size) {
                if (j in indexesToRemove) continue
                
                // Calculer l'IoU entre les boîtes i et j
                val iou = calculateIoU(sortedBoxes[i], sortedBoxes[j])
                
                // Si l'IoU est supérieur au seuil, marquer la boîte j pour suppression
                if (iou > iouThreshold) {
                    indexesToRemove.add(j)
                }
            }
        }
        
        return selectedBoxes
    }
    
    /**
     * Calcule l'Intersection over Union (IoU) entre deux boîtes
     */
    private fun calculateIoU(box1: Output0, box2: Output0): Float {
        // Calculer les coordonnées de l'intersection
        val xLeft = maxOf(box1.x1, box2.x1)
        val yTop = maxOf(box1.y1, box2.y1)
        val xRight = minOf(box1.x2, box2.x2)
        val yBottom = minOf(box1.y2, box2.y2)
        
        // Vérifier s'il y a une intersection
        if (xRight < xLeft || yBottom < yTop) return 0f
        
        // Calculer l'aire de l'intersection
        val intersectionArea = (xRight - xLeft) * (yBottom - yTop)
        
        // Calculer les aires des deux boîtes
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        
        // Calculer l'aire de l'union
        val unionArea = box1Area + box2Area - intersectionArea
        
        // Retourner l'IoU
        return intersectionArea / unionArea
    }
}