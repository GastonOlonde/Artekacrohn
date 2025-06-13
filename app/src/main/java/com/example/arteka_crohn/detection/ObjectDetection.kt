package com.example.arteka_crohn.detection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.example.arteka_crohn.Output0
import com.example.arteka_crohn.detection.config.DetectionConfig
import com.example.arteka_crohn.detection.model.TensorFlowLiteDetectionModel
import com.example.arteka_crohn.detection.preprocessing.DetectionImagePreprocessor
import com.example.arteka_crohn.detection.preprocessing.ImagePreprocessor
import com.example.arteka_crohn.detection.postprocessing.DetectionPostprocessor
import com.example.arteka_crohn.detection.postprocessing.ObjectDetectionPostprocessor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

/**
 * Classe principale de détection d'objets qui coordonne les différents composants
 * Version simplifiée par rapport à la segmentation d'instance
 */
class ObjectDetection(
    context: Context,
    modelPath: String,
    labelPath: String?,
    confidenceThreshold: Float = DetectionConfig.CONFIDENCE_THRESHOLD_DRAW,
    private val objectDetectionListener: ObjectDetectionListener,
    private val message: (String) -> Unit
) {
    private val TAG = "ObjectDetection"

    // Composants de détection suivant les principes SOLID
    private val model: TensorFlowLiteDetectionModel = TensorFlowLiteDetectionModel(context, modelPath, labelPath, message)
    private val preprocessor: ImagePreprocessor = DetectionImagePreprocessor(model)
    private val postprocessor: DetectionPostprocessor = ObjectDetectionPostprocessor(model, confidenceThreshold)

    /**
     * Ferme les ressources utilisées par le modèle
     */
    fun close() {
        model.close()
    }
    
    /**
     * Modifie le seuil de confiance pour les détections
     * @param threshold Nouveau seuil de confiance (entre 0.0 et 1.0)
     */
    fun setConfidenceThreshold(threshold: Float) {
        (postprocessor as ObjectDetectionPostprocessor).setConfidenceThreshold(threshold)
    }

    /**
     * Exécute la détection d'objets sur une image
     * @param frame L'image à traiter
     */
    fun invoke(frame: Bitmap) {
        // Vérification de l'initialisation du modèle
        if (!model.isInitialized()) {
            objectDetectionListener.onError("Interpreter not initialized properly")
            return
        }

        try {
            // Prétraitement
            val t0 = SystemClock.uptimeMillis()
            val imageBuffer = preprocessor.preprocess(frame)
            val t1 = SystemClock.uptimeMillis()
            val preProcessTime = t1 - t0
            
            Log.d(TAG, "Prétraitement terminé en $preProcessTime ms")
            
            // Préparation des buffers de sortie selon le format du modèle de détection
            // Pour un modèle typique SSD/YOLO, on a généralement un seul tensor de sortie
            // contenant les boîtes, les scores et les classes
            val outputBuffer = TensorBuffer.createFixedSize(
                model.getOutputShape(),
                TensorFlowLiteDetectionModel.OUTPUT_IMAGE_TYPE
            )
            
            val outputs = mapOf<Int, Any>(
                0 to outputBuffer.buffer.rewind()
            )
            
            // Inférence
            val t2 = SystemClock.uptimeMillis()
            model.runInference(imageBuffer, outputs)
            val t3 = SystemClock.uptimeMillis()
            val inferenceTime = t3 - t2
            
            Log.d(TAG, "Inférence terminée en $inferenceTime ms")
            
            // Post-traitement
            val t4 = SystemClock.uptimeMillis()
            
            // Log des 10 premières valeurs du buffer de sortie pour débogage
            val outputArray = outputBuffer.floatArray
            Log.d(TAG, "Taille du buffer de sortie: ${outputArray.size}")
            if (outputArray.isNotEmpty()) {
                val sampleSize = minOf(10, outputArray.size)
                val sampleValues = outputArray.take(sampleSize).joinToString(", ")
                Log.d(TAG, "Échantillon des valeurs de sortie: $sampleValues")
            }
            
            val detectionResults = postprocessor.processDetections(outputBuffer.floatArray)
            
            if (detectionResults.isEmpty()) {
                Log.d(TAG, "Aucune détection après post-traitement")
                objectDetectionListener.onEmpty()
                return
            }
            
            Log.d(TAG, "Détections trouvées: ${detectionResults.size}")
            
            val t5 = SystemClock.uptimeMillis()
            val postProcessTime = t5 - t4
            
            // Notification des résultats
            objectDetectionListener.onDetect(
                inferenceTime = inferenceTime,
                results = detectionResults,
                preProcessTime = preProcessTime,
                postProcessTime = postProcessTime
            )
            
            // Libération explicite des ressources
            outputBuffer.buffer.clear()
            
            // Pour les ByteBuffer, on peut les réinitialiser
            for (buffer in imageBuffer) {
                buffer.rewind() // Réinitialise la position du buffer à 0
            }

            // Suggestion au garbage collector de s'exécuter si nécessaire
            if (inferenceTime > 200) { // Si l'inférence prend trop de temps
                System.gc()
            }
        } catch (e: Exception) {
            Log.e("ObjectDetection", "Error during inference: ${e.message}", e)
            objectDetectionListener.onError("Error during inference: ${e.message}")
        }
    }
}
