package com.example.arteka_crohn.detection.model

import android.content.Context
import android.util.Log
import com.example.arteka_crohn.data.Output0
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
        // Considérer tout modèle avec plus d'un tenseur de sortie comme multi-output
        isMultiOutputFormat = outputShapes.size > 1
        
        Log.d(TAG, "Initialized MobileNet SSD model with ${outputShapes.size} output tensors")
        Log.d(TAG, "Using ${if(isMultiOutputFormat) "multi" else "single"}-output format processing")
        
        // Log des dimensions de chaque tenseur de sortie pour le débogage
        outputShapes.forEachIndexed { index, shape ->
            Log.d(TAG, "Output tensor $index shape: ${shape.contentToString()}")
        }
    }
    
    override fun runInference(inputBuffers: Array<ByteBuffer>): Any {
        val interp = interpreter ?: throw IllegalStateException("Interpreter not initialized")
        
        if (isMultiOutputFormat) {
            // Format multi-output
            if (outputShapes.size == 8) {
                // Format spécifique avec 8 tenseurs de sortie
                return runInferenceForEightOutputs(interp, inputBuffers)
            } else if (outputShapes.size == 12) {
                // Format spécifique avec 12 tenseurs de sortie
                return runInferenceForTwelveOutputs(interp, inputBuffers)
            }
            
            // Format standard multi-output (4 tenseurs: locations, classes, scores, count)
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
    
    /**
     * Méthode spécialisée pour exécuter l'inférence avec un modèle à 8 tenseurs de sortie
     * Format spécifique MobileNet avec les tenseurs suivants:
     * - detection_anchor_indices [1,-1]
     * - detection_boxes [1,-1,-1]
     * - detection_classes [1,-1]
     * - detection_multiclass_scores [1,-1,-1]
     * - detection_scores [1,-1]
     * - num_detections [1]
     * - raw_detection_boxes [1,12804,4]
     * - raw_detection_scores [1,12804,2]
     */
    private fun runInferenceForEightOutputs(interp: org.tensorflow.lite.Interpreter, inputBuffers: Array<ByteBuffer>): ExtendedMultiOutputResult {
        // Log tous les tenseurs de sortie pour débugger
        Log.d(TAG, "Running inference for 8-output MobileNet model")
        outputShapes.forEachIndexed { index, shape ->
            Log.d(TAG, "Output $index: shape=${shape.contentToString()}")
        }
        
        try {
            // Utiliser des objets Java typés plutôt que des ByteBuffers
            // Cette approche est plus fiable car TensorFlow s'occupe des conversions de types

            // Créer une HashMap pour les sorties typées
            val outputs = HashMap<Int, Any>()
            
            // Créer des tableaux pour chaque sortie en fonction de sa forme exacte
            // Nous utilisons les dimensions réelles des tenseurs pour éviter les erreurs de forme
            
            // Pour Output 0: raw_detection_boxes [1, 12804, 4]
            val rawBoxesArray = Array(outputShapes[0][0]) { 
                Array(outputShapes[0][1]) { FloatArray(outputShapes[0][2]) } 
            }
            outputs[0] = rawBoxesArray
            
            // Pour Output 1: [1, 1] -> utiliser la forme exacte
            val output1Shape = outputShapes[1]
            val output1Array = Array(output1Shape[0]) { FloatArray(output1Shape[1]) }
            outputs[1] = output1Array
            
            // Pour Output 2: [1] -> utiliser FloatArray
            val output2Array = FloatArray(outputShapes[2][0])
            outputs[2] = output2Array
            
            // Pour Output 3: raw_detection_scores [1, 12804, 2]
            val rawScoresArray = Array(outputShapes[3][0]) { 
                Array(outputShapes[3][1]) { FloatArray(outputShapes[3][2]) } 
            }
            outputs[3] = rawScoresArray
            
            // Pour Output 4: [1, 1, 1] -> utiliser la forme exacte
            val output4Shape = outputShapes[4]
            val output4Array = Array(output4Shape[0]) { 
                Array(output4Shape[1]) { FloatArray(output4Shape[2]) } 
            }
            outputs[4] = output4Array
            
            // Pour Output 5: [1, 1] -> utiliser la forme exacte
            val output5Shape = outputShapes[5]
            val output5Array = Array(output5Shape[0]) { FloatArray(output5Shape[1]) }
            outputs[5] = output5Array
            
            // Pour Output 6: [1, 1] -> utiliser la forme exacte
            val output6Shape = outputShapes[6]
            val output6Array = Array(output6Shape[0]) { FloatArray(output6Shape[1]) }
            outputs[6] = output6Array
            
            // Pour Output 7: [1, 1, 1] -> utiliser la forme exacte
            val output7Shape = outputShapes[7]
            val output7Array = Array(output7Shape[0]) { 
                Array(output7Shape[1]) { FloatArray(output7Shape[2]) } 
            }
            outputs[7] = output7Array
            
            // Exécuter l'inférence
            interp.runForMultipleInputsOutputs(inputBuffers, outputs)
            
            // Extraire numDetections (généralement dans output2Array)
            val numDetections = output2Array[0].toInt()
            Log.d(TAG, "Nombre de détections: $numDetections")
            
            // Convertir les sorties en format uniforme
            // Créer des FloatArrays à partir des tableaux multidimensionnels
            val rawBoxes = FloatArray(rawBoxesArray[0].size * 4)
            for (i in 0 until rawBoxesArray[0].size) {
                for (j in 0 until 4) {
                    rawBoxes[i * 4 + j] = rawBoxesArray[0][i][j]
                }
            }
            
            val rawScores = FloatArray(rawScoresArray[0].size * 2)
            for (i in 0 until rawScoresArray[0].size) {
                for (j in 0 until 2) {
                    rawScores[i * 2 + j] = rawScoresArray[0][i][j]
                }
            }
            
            // Créer d'autres FloatArrays adaptés
            // Pour ceux-ci, on prend des tableaux vides ou avec des valeurs par défaut
            // car nous utiliserons principalement les rawBoxes et rawScores
            val anchorIndices = FloatArray(output1Array[0].size) { 0f }
            val boxes = FloatArray(numDetections * 4) { 0f }
            val classes = FloatArray(numDetections) { 0f }
            val multiclassScores = FloatArray(numDetections * 2) { 0f }
            val scores = FloatArray(numDetections) { 0f }
            
            return ExtendedMultiOutputResult(
                anchorIndices = anchorIndices,
                boxes = boxes,
                classes = classes,
                multiclassScores = multiclassScores,
                scores = scores,
                numDetections = numDetections,
                rawBoxes = rawBoxes,
                rawScores = rawScores
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error during 8-output inference: ${e.message}")
            e.printStackTrace()
            
            // Log plus de détails pour aider au debug
            Log.e(TAG, "Tensor details:")
            for (i in 0 until outputShapes.size) {
                try {
                    val tensor = interp.getOutputTensor(i)
                    Log.e(TAG, "Output $i: shape=${tensor.shape().contentToString()}, " +
                          "dataType=${tensor.dataType()}, numBytes=${tensor.numBytes()}")
                } catch (e2: Exception) {
                    Log.e(TAG, "Error getting tensor $i details: ${e2.message}")
                }
            }
            
            throw e
        }
    }
    
    /**
     * Méthode spécialisée pour exécuter l'inférence avec un modèle à 12 tenseurs de sortie
     * Cette méthode analyse d'abord les formats des tenseurs pour s'adapter aux différentes implémentations
     */
    private fun runInferenceForTwelveOutputs(interp: org.tensorflow.lite.Interpreter, inputBuffers: Array<ByteBuffer>): TwelveOutputResult {
        // Log tous les tenseurs de sortie pour débugger
        Log.d(TAG, "Running inference for 12-output model")
        outputShapes.forEachIndexed { index, shape ->
            Log.d(TAG, "Output $index: shape=${shape.contentToString()}")
        }
        
        // Valider les formats des tenseurs avant de procéder
        validateTwelveOutputFormat()
        
        try {
            // Utiliser des objets Java typés plutôt que des ByteBuffers
            val outputs = HashMap<Int, Any>()
            
            // Analyser les formes des tenseurs pour déterminer les dimensions appropriées
            // et créer des structures de données adaptées
            
            // Déterminer la taille maximale pour les tableaux de détection
            val maxDetections = outputShapes.filter { it.size >= 2 }.maxOfOrNull { it[1] } ?: 100
            Log.d(TAG, "Determined max detections: $maxDetections")
            
            // Créer des tableaux pour chaque sortie en fonction de sa forme
            for (i in 0 until 12) {
                if (i >= outputShapes.size) {
                    Log.w(TAG, "Output index $i exceeds available shapes (${outputShapes.size})")
                    continue
                }
                
                val shape = outputShapes[i]
                Log.d(TAG, "Creating array for output $i with shape ${shape.contentToString()}")
                
                when (shape.size) {
                    1 -> outputs[i] = ByteArray(shape[0])
                    2 -> outputs[i] = Array(shape[0]) { ByteArray(shape[1]) }
                    3 -> outputs[i] = Array(shape[0]) { Array(shape[1]) { ByteArray(shape[2]) } }
                    4 -> outputs[i] = Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { ByteArray(shape[3]) } } }
                    else -> {
                        Log.w(TAG, "Unsupported tensor shape size: ${shape.size} for output $i")
                        outputs[i] = ByteArray(1)
                    }
                }
            }
            
            // Exécuter l'inférence
            interp.runForMultipleInputsOutputs(inputBuffers, outputs)
            
            // Fonction utilitaire pour convertir un byte non signé en float
            fun byteToFloat(value: Byte): Float {
                return value.toInt().and(0xFF).toFloat()
            }
            
            // Extraire numDetections (généralement dans un des tenseurs scalaires)
            var numDetections = 0
            
            // Chercher le tenseur numDetections parmi les sorties
            for (i in 0 until 12) {
                if (i >= outputs.size) continue
                
                val output = outputs[i]
                // Vérifier si c'est un FloatArray (modèle FLOAT32)
                if (output is FloatArray && output.size == 1) {
                    numDetections = output[0].toInt()
                    Log.d(TAG, "Found numDetections in FloatArray tensor $i: $numDetections")
                    break
                }
                // Vérifier si c'est un ByteArray (modèle UINT8)
                else if (output is ByteArray && output.size == 1) {
                    numDetections = byteToFloat(output[0]).toInt()
                    Log.d(TAG, "Found numDetections in ByteArray tensor $i: $numDetections")
                    break
                }
            }
            
            // Si aucun tenseur numDetections n'est trouvé, utiliser une valeur par défaut
            if (numDetections == 0) {
                // Essayer de déterminer le nombre de détections à partir des dimensions des tenseurs
                for (i in 0 until outputShapes.size) {
                    if (outputShapes[i].size >= 2) {
                        val potentialDetections = outputShapes[i][1]
                        if (potentialDetections > numDetections) {
                            numDetections = potentialDetections
                        }
                    }
                }
                
                if (numDetections == 0) {
                    numDetections = 100
                }
                Log.w(TAG, "No numDetections tensor found, using derived value: $numDetections")
            }
            
            // Initialiser les tableaux pour stocker les résultats
            val detectionBoxes = FloatArray(numDetections * 4) { 0f }
            val detectionClasses = FloatArray(numDetections) { 0f }
            val detectionScores = FloatArray(numDetections) { 0f }
            val rawBoxes = FloatArray(maxDetections * 4) { 0f }
            val rawScores = FloatArray(maxDetections * 2) { 0f }
            val anchorBoxes = FloatArray(maxDetections * 4) { 0f }
            val featureMapShapes = FloatArray(10) { 0f } // Taille arbitraire
            val anchorIndices = FloatArray(numDetections) { 0f }
            val multiclassScores = FloatArray(numDetections * 2) { 0f }
            val additionalAttributes = FloatArray(numDetections) { 0f }
            val additionalParameters = FloatArray(10) { 0f } // Taille arbitraire
            
            // Extraire les données des tenseurs
            try {
                // Identifier et extraire les boîtes de détection
                for (i in 0 until 12) {
                    if (i >= outputs.size) continue
                    
                    val output = outputs[i]
                    Log.d(TAG, "Processing output tensor $i of type ${output?.javaClass?.simpleName}")
                    
                    // Traitement pour les tenseurs FLOAT32
                    if (output is Array<*> && output.size > 0) {
                        // Vérifier si c'est un tenseur de boîtes [1, N, 4]
                        if (output[0] is Array<*>) {
                            val firstDim = output[0] as Array<*>
                            if (firstDim.isNotEmpty() && firstDim[0] is FloatArray && (firstDim[0] as FloatArray).size == 4) {
                                Log.d(TAG, "Found FLOAT32 detection boxes in tensor $i")
                                for (j in 0 until min(numDetections, firstDim.size)) {
                                    val box = firstDim[j] as FloatArray
                                    System.arraycopy(box, 0, detectionBoxes, j * 4, 4)
                                }
                            }
                        }
                        // Vérifier si c'est un tenseur de scores [1, N]
                        else if (output[0] is FloatArray) {
                            val scores = output[0] as FloatArray
                            if (scores.size >= numDetections) {
                                Log.d(TAG, "Found FLOAT32 detection scores in tensor $i")
                                System.arraycopy(scores, 0, detectionScores, 0, min(numDetections, scores.size))
                            }
                        }
                    }
                    
                    // Traitement pour les tenseurs UINT8
                    else if (output is Array<*> && output.size > 0) {
                        // Vérifier si c'est un tenseur 4D [1, H, W, C]
                        if (output[0] is Array<*>) {
                            val firstDim = output[0] as Array<*>
                            if (firstDim.isNotEmpty() && firstDim[0] is Array<*>) {
                                val secondDim = firstDim[0] as Array<*>
                                if (secondDim.isNotEmpty() && secondDim[0] is ByteArray) {
                                    Log.d(TAG, "Found UINT8 4D tensor in tensor $i with shape [${output.size}, ${firstDim.size}, ${secondDim.size}, ${(secondDim[0] as ByteArray).size}]")
                                    
                                    // Si c'est un tenseur de boîtes (généralement avec 4 canaux pour x, y, w, h)
                                    if ((secondDim[0] as ByteArray).size == 4) {
                                        Log.d(TAG, "This might be a detection boxes tensor")
                                        var boxIndex = 0
                                        
                                        // Parcourir le tenseur 4D et extraire les boîtes
                                        for (h in 0 until firstDim.size) {
                                            for (w in 0 until secondDim.size) {
                                                if (boxIndex >= numDetections) break
                                                
                                                val boxData = (secondDim[w] as ByteArray)
                                                for (c in 0 until 4) {
                                                    detectionBoxes[boxIndex * 4 + c] = byteToFloat(boxData[c])
                                                }
                                                boxIndex++
                                            }
                                        }
                                    }
                                    // Si c'est un tenseur de classes/scores
                                    else if ((secondDim[0] as ByteArray).size > 4) {
                                        Log.d(TAG, "This might be a classes/scores tensor")
                                        var detectionIndex = 0
                                        
                                        // Parcourir le tenseur 4D et extraire les scores
                                        for (h in 0 until firstDim.size) {
                                            for (w in 0 until secondDim.size) {
                                                if (detectionIndex >= numDetections) break
                                                
                                                val data = (secondDim[w] as ByteArray)
                                                // Supposons que le score est dans le premier canal
                                                detectionScores[detectionIndex] = byteToFloat(data[0])
                                                // Supposons que la classe est dans le deuxième canal
                                                if (data.size > 1) {
                                                    detectionClasses[detectionIndex] = byteToFloat(data[1])
                                                }
                                                detectionIndex++
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Vérifier si c'est un tenseur 3D [1, N, C]
                        else if (output[0] is Array<*> && (output[0] as Array<*>).isNotEmpty() && (output[0] as Array<*>)[0] is ByteArray) {
                            val boxes = output[0] as Array<*>
                            Log.d(TAG, "Found UINT8 3D tensor in tensor $i with shape [${output.size}, ${boxes.size}, ${(boxes[0] as ByteArray).size}]")
                            
                            // Si c'est un tenseur de boîtes [1, N, 4]
                            if ((boxes[0] as ByteArray).size == 4) {
                                Log.d(TAG, "This is likely a detection boxes tensor")
                                for (j in 0 until min(numDetections, boxes.size)) {
                                    val box = boxes[j] as ByteArray
                                    for (k in 0 until 4) {
                                        detectionBoxes[j * 4 + k] = byteToFloat(box[k])
                                    }
                                }
                            }
                            // Si c'est un tenseur de scores [1, N, 1] ou [1, N, C]
                            else if ((boxes[0] as ByteArray).size >= 1) {
                                Log.d(TAG, "This might be a scores or classes tensor")
                                for (j in 0 until min(numDetections, boxes.size)) {
                                    val data = boxes[j] as ByteArray
                                    // Premier canal pour le score
                                    detectionScores[j] = byteToFloat(data[0])
                                    // Deuxième canal pour la classe (si disponible)
                                    if (data.size > 1) {
                                        detectionClasses[j] = byteToFloat(data[1])
                                    }
                                }
                            }
                        }
                        // Vérifier si c'est un tenseur 2D [1, N]
                        else if (output[0] is ByteArray) {
                            val data = output[0] as ByteArray
                            Log.d(TAG, "Found UINT8 2D tensor in tensor $i with shape [${output.size}, ${data.size}]")
                            
                            // Si la taille correspond au nombre de détections, c'est probablement des scores
                            if (data.size >= numDetections) {
                                Log.d(TAG, "This might be a scores tensor")
                                for (j in 0 until min(numDetections, data.size)) {
                                    detectionScores[j] = byteToFloat(data[j])
                                }
                            }
                        }
                    }
                    // Traitement pour les tenseurs 1D ByteArray
                    else if (output is ByteArray && output.size > 1) {
                        Log.d(TAG, "Found UINT8 1D tensor in tensor $i with size ${output.size}")
                        
                        // Si la taille est un multiple de 4, c'est peut-être des boîtes aplaties
                        if (output.size % 4 == 0) {
                            val numBoxes = output.size / 4
                            Log.d(TAG, "This might be flattened boxes data with $numBoxes boxes")
                            for (j in 0 until min(numDetections, numBoxes)) {
                                for (k in 0 until 4) {
                                    detectionBoxes[j * 4 + k] = byteToFloat(output[j * 4 + k])
                                }
                            }
                        }
                        // Sinon, c'est peut-être des scores
                        else if (output.size >= numDetections) {
                            Log.d(TAG, "This might be a scores tensor")
                            for (j in 0 until min(numDetections, output.size)) {
                                detectionScores[j] = byteToFloat(output[j])
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error extracting tensor data: ${e.message}")
            }
            
            return TwelveOutputResult(
                detectionBoxes = detectionBoxes,
                detectionClasses = detectionClasses,
                detectionScores = detectionScores,
                numDetections = numDetections,
                rawBoxes = rawBoxes,
                rawScores = rawScores,
                anchorBoxes = anchorBoxes,
                featureMapShapes = featureMapShapes,
                anchorIndices = anchorIndices,
                multiclassScores = multiclassScores,
                additionalAttributes = additionalAttributes,
                additionalParameters = additionalParameters
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error during 12-output inference: ${e.message}")
            e.printStackTrace()
            
            // Log plus de détails pour aider au debug
            Log.e(TAG, "Tensor details:")
            for (i in 0 until outputShapes.size) {
                try {
                    val tensor = interp.getOutputTensor(i)
                    Log.e(TAG, "Output $i: shape=${tensor.shape().contentToString()}, " +
                          "dataType=${tensor.dataType()}, numBytes=${tensor.numBytes()}")
                } catch (e2: Exception) {
                    Log.e(TAG, "Error getting tensor $i details: ${e2.message}")
                }
            }
            
            throw e
        }
    }
    
    /**
     * Valide les formats des tenseurs pour un modèle à 12 sorties
     * Vérifie que les formes des tenseurs sont cohérentes et adaptées au traitement
     */
    private fun validateTwelveOutputFormat() {
        if (outputShapes.size != 12) {
            Log.w(TAG, "Expected 12 output tensors, but got ${outputShapes.size}")
            return
        }
        
        try {
            // Log des dimensions pour référence
            Log.d(TAG, "12-output format tensor shapes validation:")
            outputShapes.forEachIndexed { index, shape ->
                Log.d(TAG, "Tensor $index: ${shape.contentToString()}")
            }
            
            // Vérifier la présence d'au moins un tenseur de boîtes (forme [1, N, 4])
            var hasBoxesTensor = false
            var hasScoresTensor = false
            var hasNumDetectionsTensor = false
            
            for (shape in outputShapes) {
                // Vérifier si c'est un tenseur de boîtes
                if (shape.size >= 3 && shape[shape.size - 1] == 4) {
                    hasBoxesTensor = true
                }
                
                // Vérifier si c'est un tenseur de scores
                if (shape.size == 2) {
                    hasScoresTensor = true
                }
                
                // Vérifier si c'est un tenseur numDetections
                if (shape.size == 1 && shape[0] == 1) {
                    hasNumDetectionsTensor = true
                }
            }
            
            if (!hasBoxesTensor) {
                Log.w(TAG, "No boxes tensor found in 12-output format")
            }
            
            if (!hasScoresTensor) {
                Log.w(TAG, "No scores tensor found in 12-output format")
            }
            
            if (!hasNumDetectionsTensor) {
                Log.w(TAG, "No numDetections tensor found in 12-output format")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error validating 12-output format: ${e.message}")
        }
    }
    
    /**
     * Traite les sorties au format multi-output avec 12 tenseurs
     */
    private fun processTwelveOutputFormat(result: TwelveOutputResult, confidenceThreshold: Float): List<Output0> {
        val detections = mutableListOf<Output0>()
        val numDetections = result.numDetections
        
        Log.d(TAG, "Processing 12-output format with $numDetections detections")
        
        try {
            // Utiliser les boîtes et scores déjà traités par le modèle
            val boxes = result.detectionBoxes
            val scores = result.detectionScores
            val classes = result.detectionClasses
            
            // Itérer à travers les détections
            for (i in 0 until min(numDetections, scores.size)) {
                val score = scores[i]
                
                // Ignorer les détections sous le seuil de confiance
                if (score < confidenceThreshold) continue
                
                // Récupérer les coordonnées de la boîte [ymin, xmin, ymax, xmax]
                val boxIndex = i * 4
                if (boxIndex + 3 >= boxes.size) {
                    Log.w(TAG, "Box index out of bounds: $boxIndex for boxes size ${boxes.size}")
                    continue
                }
                
                val ymin = boxes[boxIndex]
                val xmin = boxes[boxIndex + 1]
                val ymax = boxes[boxIndex + 2]
                val xmax = boxes[boxIndex + 3]
                
                // Vérifier que les coordonnées sont valides
                if (ymin < 0f || xmin < 0f || ymax > 1f || xmax > 1f || ymin > ymax || xmin > xmax) {
                    Log.w(TAG, "Invalid box coordinates: [$ymin, $xmin, $ymax, $xmax]")
                    continue
                }
                
                // Récupérer la classe
                val classId = if (i < classes.size) classes[i].toInt() else 0
                
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
        } catch (e: Exception) {
            Log.e(TAG, "Error processing 12-output format: ${e.message}")
            e.printStackTrace()
        }
        
        // Appliquer NMS (Non-Maximum Suppression) pour éliminer les boîtes redondantes
        return applyNMS(detections, DetectionConfig.IOU_THRESHOLD)
    }
    
    override fun processOutput(rawOutput: Any, confidenceThreshold: Float): List<Output0> {
        // Vérifier si c'est un format Azure Custom Vision [1,13,13,30]
        if (outputShapes.size == 1 && 
            outputShapes[0].size == 4 && 
            outputShapes[0][1] == 13 && 
            outputShapes[0][2] == 13 && 
            outputShapes[0][3] == 30) {
            // Format Azure Custom Vision détecté
            Log.d(TAG, "Format Azure Custom Vision détecté [1,13,13,30]")
            // Note: cette fonctionnalité a été désactivée temporairement
            Log.w(TAG, "Le traitement du format Azure Custom Vision est désactivé")
            return emptyList()
        }
        
        // Format avec 8 tenseurs de sortie (MobileNet advanced)
        if (rawOutput is ExtendedMultiOutputResult) {
            Log.d(TAG, "Processing 8-tensor MobileNet output format")
            return processExtendedMultiOutputFormat(rawOutput, confidenceThreshold)
        }
        
        // Format avec 12 tenseurs de sortie
        if (rawOutput is TwelveOutputResult) {
            Log.d(TAG, "Processing 12-tensor output format")
            return processTwelveOutputFormat(rawOutput, confidenceThreshold)
        }
        
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
     * Traite les sorties au format multi-output étendu (8 tenseurs)
     */
    private fun processExtendedMultiOutputFormat(result: ExtendedMultiOutputResult, confidenceThreshold: Float): List<Output0> {
        val detections = mutableListOf<Output0>()
        val numDetections = result.numDetections
        
        Log.d(TAG, "Processing extended multi-output format with $numDetections detections")
        
        try {
            // Utiliser les boîtes et scores déjà traités par le modèle
            val boxes = result.boxes
            val scores = result.scores
            val classes = result.classes
            
            // Déterminer la structure des tenseurs
            // La dimension des boîtes devrait être [1, numDetections, 4]
            var boxOffset = 0
            var batchDim = 1
            
            // Itérer à travers les détections
            for (i in 0 until min(numDetections, scores.size)) {
                val score = scores[i]
                
                // Ignorer les détections sous le seuil de confiance
                if (score < confidenceThreshold) continue
                
                // Récupérer les coordonnées de la boîte [ymin, xmin, ymax, xmax]
                val boxIndex = i * 4
                if (boxIndex + 3 >= boxes.size) {
                    Log.w(TAG, "Box index out of bounds: $boxIndex for boxes size ${boxes.size}")
                    continue
                }
                
                val ymin = boxes[boxIndex]
                val xmin = boxes[boxIndex + 1]
                val ymax = boxes[boxIndex + 2]
                val xmax = boxes[boxIndex + 3]
                
                // Vérifier que les coordonnées sont valides
                if (ymin < 0f || xmin < 0f || ymax > 1f || xmax > 1f || ymin > ymax || xmin > xmax) {
                    Log.w(TAG, "Invalid box coordinates: [$ymin, $xmin, $ymax, $xmax]")
                    continue
                }
                
                // Récupérer la classe
                val classId = if (i < classes.size) classes[i].toInt() else 0
                
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
        } catch (e: Exception) {
            Log.e(TAG, "Error processing extended multi-output: ${e.message}")
            e.printStackTrace()
        }
        
        // Appliquer NMS (Non-Maximum Suppression) pour éliminer les boîtes redondantes
        return applyNMS(detections, DetectionConfig.IOU_THRESHOLD)
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
     * Fonction sigmoid pour normaliser les valeurs entre 0 et 1
     */
    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + Math.exp(-x.toDouble()).toFloat())
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
    
    /**
     * Classe pour encapsuler les résultats du format multi-output avec 8 tenseurs
     */
    data class ExtendedMultiOutputResult(
        val anchorIndices: FloatArray,     // [1,-1]
        val boxes: FloatArray,             // [1,-1,-1]
        val classes: FloatArray,           // [1,-1]
        val multiclassScores: FloatArray,  // [1,-1,-1]
        val scores: FloatArray,            // [1,-1]
        val numDetections: Int,            // [1]
        val rawBoxes: FloatArray,          // [1,12804,4]
        val rawScores: FloatArray          // [1,12804,2]
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            
            other as ExtendedMultiOutputResult
            
            if (!anchorIndices.contentEquals(other.anchorIndices)) return false
            if (!boxes.contentEquals(other.boxes)) return false
            if (!classes.contentEquals(other.classes)) return false
            if (!multiclassScores.contentEquals(other.multiclassScores)) return false
            if (!scores.contentEquals(other.scores)) return false
            if (numDetections != other.numDetections) return false
            if (!rawBoxes.contentEquals(other.rawBoxes)) return false
            if (!rawScores.contentEquals(other.rawScores)) return false
            
            return true
        }
        
        override fun hashCode(): Int {
            var result = anchorIndices.contentHashCode()
            result = 31 * result + boxes.contentHashCode()
            result = 31 * result + classes.contentHashCode()
            result = 31 * result + multiclassScores.contentHashCode()
            result = 31 * result + scores.contentHashCode()
            result = 31 * result + numDetections
            result = 31 * result + rawBoxes.contentHashCode()
            result = 31 * result + rawScores.contentHashCode()
            return result
        }
    }
    
    /**
     * Classe pour encapsuler les résultats du format multi-output avec 12 tenseurs
     */
    data class TwelveOutputResult(
        val detectionBoxes: FloatArray,          // [1, N, 4]
        val detectionClasses: FloatArray,        // [1, N]
        val detectionScores: FloatArray,         // [1, N]
        val numDetections: Int,                  // [1]
        val rawBoxes: FloatArray,                // [1, M, 4]
        val rawScores: FloatArray,               // [1, M, C]
        val anchorBoxes: FloatArray,             // [1, M, 4]
        val featureMapShapes: FloatArray,        // [10]
        val anchorIndices: FloatArray,           // [1, N]
        val multiclassScores: FloatArray,        // [1, N, C]
        val additionalAttributes: FloatArray,    // [1, N]
        val additionalParameters: FloatArray     // [10]
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            
            other as TwelveOutputResult
            
            if (!detectionBoxes.contentEquals(other.detectionBoxes)) return false
            if (!detectionClasses.contentEquals(other.detectionClasses)) return false
            if (!detectionScores.contentEquals(other.detectionScores)) return false
            if (numDetections != other.numDetections) return false
            if (!rawBoxes.contentEquals(other.rawBoxes)) return false
            if (!rawScores.contentEquals(other.rawScores)) return false
            if (!anchorBoxes.contentEquals(other.anchorBoxes)) return false
            if (!featureMapShapes.contentEquals(other.featureMapShapes)) return false
            if (!anchorIndices.contentEquals(other.anchorIndices)) return false
            if (!multiclassScores.contentEquals(other.multiclassScores)) return false
            if (!additionalAttributes.contentEquals(other.additionalAttributes)) return false
            if (!additionalParameters.contentEquals(other.additionalParameters)) return false
            
            return true
        }
        
        override fun hashCode(): Int {
            var result = detectionBoxes.contentHashCode()
            result = 31 * result + detectionClasses.contentHashCode()
            result = 31 * result + detectionScores.contentHashCode()
            result = 31 * result + numDetections
            result = 31 * result + rawBoxes.contentHashCode()
            result = 31 * result + rawScores.contentHashCode()
            result = 31 * result + anchorBoxes.contentHashCode()
            result = 31 * result + featureMapShapes.contentHashCode()
            result = 31 * result + anchorIndices.contentHashCode()
            result = 31 * result + multiclassScores.contentHashCode()
            result = 31 * result + additionalAttributes.contentHashCode()
            result = 31 * result + additionalParameters.contentHashCode()
            return result
        }
    }
}
