package com.example.arteka_crohn.segmentation

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.example.arteka_crohn.segmentation.model.TensorFlowLiteModel
import com.example.arteka_crohn.segmentation.preprocessing.ImagePreprocessor
import com.example.arteka_crohn.segmentation.preprocessing.TFLiteImagePreprocessor
import com.example.arteka_crohn.segmentation.postprocessing.InstanceSegmentationPostprocessor
import com.example.arteka_crohn.segmentation.postprocessing.SegmentationPostprocessor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

/**
 * Classe principale de segmentation d'instance qui coordonne les différents composants
 * Respecte les principes de ségrégation d'interface (I) et d'inversion de dépendance (D) dans SOLID
 */
class InstanceSegmentation(
    context: Context,
    modelPath: String,
    labelPath: String?,
    private val instanceSegmentationListener: InstanceSegmentationListener,
    private val message: (String) -> Unit
) {
    // Composants de segmentation suivant les principes SOLID
    private val model: TensorFlowLiteModel = TensorFlowLiteModel(context, modelPath, labelPath, message)
    private val preprocessor: ImagePreprocessor = TFLiteImagePreprocessor(model)
    private val postprocessor: SegmentationPostprocessor = InstanceSegmentationPostprocessor(model)

    /**
     * Ferme les ressources utilisées par le modèle
     */
    fun close() {
        model.close()
    }

    /**
     * Exécute la segmentation d'instance sur une image
     * @param frame L'image à traiter
     */
    fun invoke(frame: Bitmap) {
        // Vérification de l'initialisation du modèle
        if (!model.isInitialized()) {
            instanceSegmentationListener.onError("Interpreter not initialized properly")
            return
        }

        try {
            // Prétraitement
            val t0 = SystemClock.uptimeMillis()
            val imageBuffer = preprocessor.preprocess(frame)
            val t1 = SystemClock.uptimeMillis()
            val preProcessTime = t1 - t0
            
            // Préparation des buffers de sortie
            val coordinatesBuffer = TensorBuffer.createFixedSize(
                intArrayOf(1, model.numChannel, model.numElements),
                TensorFlowLiteModel.OUTPUT_IMAGE_TYPE
            )
            val maskProtoBuffer = TensorBuffer.createFixedSize(
                intArrayOf(1, model.xPoints, model.yPoints, model.masksNum),
                TensorFlowLiteModel.OUTPUT_IMAGE_TYPE
            )
            
            val outputBuffer = mapOf<Int, Any>(
                0 to coordinatesBuffer.buffer.rewind(),
                1 to maskProtoBuffer.buffer.rewind()
            )
            
            // Inférence
            val t2 = SystemClock.uptimeMillis()
            model.runInference(imageBuffer, outputBuffer)
            val t3 = SystemClock.uptimeMillis()
            val inferenceTime = t3 - t2
            
            // Post-traitement
            val t4 = SystemClock.uptimeMillis()
            val segmentationResults = postprocessor.processDetections(
                coordinatesBuffer.floatArray, 
                maskProtoBuffer.floatArray
            )
            
            if (segmentationResults == null) {
                instanceSegmentationListener.onEmpty()
                return
            }
            
            val t5 = SystemClock.uptimeMillis()
            val postProcessTime = t5 - t4
            
            // Notification des résultats
            instanceSegmentationListener.onDetect(
                interfaceTime = inferenceTime,
                results = segmentationResults,
                preProcessTime = preProcessTime,
                postProcessTime = postProcessTime
            )
            
            // Libération explicite des ressources
            coordinatesBuffer.buffer.clear()
            maskProtoBuffer.buffer.clear()
            
            // Pour les ByteBuffer, on peut les réinitialiser
            for (buffer in imageBuffer) {
                buffer.rewind() // Réinitialise la position du buffer à 0
            }

            // Suggestion au garbage collector de s'exécuter si nécessaire
            if (inferenceTime > 200) { // Si l'inférence prend trop de temps
                System.gc()
            }
        } catch (e: Exception) {
            Log.e("InstanceSegmentation", "Error during inference: ${e.message}", e)
            instanceSegmentationListener.onError("Error during inference: ${e.message}")
        }
    }
}