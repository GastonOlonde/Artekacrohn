package com.example.arteka_crohn.detection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.example.arteka_crohn.Output0
import com.example.arteka_crohn.detection.config.DetectionConfig
import com.example.arteka_crohn.detection.model.ModelDetector
import com.example.arteka_crohn.detection.model.ModelDetectorFactory
import com.example.arteka_crohn.detection.preprocessing.DetectionImagePreprocessor
import com.example.arteka_crohn.detection.preprocessing.ImagePreprocessor
import com.example.arteka_crohn.detection.postprocessing.DetectionPostprocessor
import com.example.arteka_crohn.detection.postprocessing.ObjectDetectionPostprocessor

/**
 * Classe principale de détection d'objets qui coordonne les différents composants
 * Utilise la détection automatique de modèle pour choisir le bon détecteur
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

    // Utilisation de la factory pour créer automatiquement le bon détecteur
    private val detectorFactory = ModelDetectorFactory()
    private val detector: ModelDetector = detectorFactory.createDetector(context, modelPath, labelPath, message)
    
    // Composants de détection suivant les principes SOLID
    private val preprocessor: ImagePreprocessor = DetectionImagePreprocessor(
        inputWidth = detector.getInputWidth(),
        inputHeight = detector.getInputHeight(),
        isQuantized = detector.isQuantized(),
        normalizationValues = detector.getNormalizationValues()
    )
    
    // Seuil de confiance pour les détections
    private var confidenceThreshold: Float = confidenceThreshold

    init {
        // Log du type de modèle détecté
        Log.d(TAG, "Modèle détecté: ${detector.getModelType().name}")
    }

    /**
     * Ferme les ressources utilisées par le modèle
     */
    fun close() {
        detector.close()
    }
    
    /**
     * Modifie le seuil de confiance pour les détections
     * @param threshold Nouveau seuil de confiance (entre 0.0 et 1.0)
     */
    fun setConfidenceThreshold(threshold: Float) {
        this.confidenceThreshold = threshold
    }

    /**
     * Exécute la détection d'objets sur une image
     * @param frame L'image à traiter
     */
    fun invoke(frame: Bitmap) {
        // Vérification de l'initialisation du modèle
        if (!detector.isInitialized()) {
            objectDetectionListener.onError("Detector not initialized properly")
            return
        }

        try {
            // Prétraitement
            val t0 = SystemClock.uptimeMillis()
            val imageBuffers = preprocessor.preprocess(frame)
            val t1 = SystemClock.uptimeMillis()
            val preProcessTime = t1 - t0
            
            Log.d(TAG, "Prétraitement terminé en $preProcessTime ms")
            
            // Inférence avec le détecteur spécifique au modèle
            val t2 = SystemClock.uptimeMillis()
            val rawOutput = detector.runInference(imageBuffers)
            val t3 = SystemClock.uptimeMillis()
            val inferenceTime = t3 - t2
            
            Log.d(TAG, "Inférence terminée en $inferenceTime ms")
            
            // Post-traitement
            val t4 = SystemClock.uptimeMillis()
            
            // Utilisation directe du processeur du détecteur pour la sortie brute
            val detectionResults = detector.processOutput(rawOutput, confidenceThreshold)
            
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
            
            // Pour les ByteBuffer, on peut les réinitialiser
            for (buffer in imageBuffers) {
                buffer.rewind() // Réinitialise la position du buffer à 0
            }

            // Suggestion au garbage collector de s'exécuter si nécessaire
            if (inferenceTime > 200) { // Si l'inférence prend trop de temps
                System.gc()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference: ${e.message}", e)
            objectDetectionListener.onError("Error during inference: ${e.message}")
        }
    }
}
