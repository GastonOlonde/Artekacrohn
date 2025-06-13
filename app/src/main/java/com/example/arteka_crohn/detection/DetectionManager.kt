package com.example.arteka_crohn.detection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.example.arteka_crohn.camera.ImageAnalysisCallback
import com.example.arteka_crohn.detection.config.DetectionConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Gestionnaire de détection responsable de l'initialisation du modèle,
 * du traitement des images et de la gestion des résultats de détection.
 */
class DetectionManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner, // Gestionnaire de la vie de cycle de l'activité
    private val modelName: String, // Nom du modèle de détection
    private val confidenceThreshold: Float = DetectionConfig.CONFIDENCE_THRESHOLD_DRAW, // Seuil de confiance pour les détections
    private val detectionListener: ObjectDetectionListener // Listener pour les résultats de détection
) : ImageAnalysisCallback {

    private var objectDetection: ObjectDetection? = null
    private val detectionLock = ReentrantLock()
    private var currentConfidenceThreshold = confidenceThreshold
    
    @Volatile private var isActive = true
    private var lastLoadedModelName: String = modelName

    /**
     * Initialise le modèle de détection
     */
    suspend fun initializeDetection() {
        if (!isActive) return
        
        try {
            Log.d(TAG, "Starting ObjectDetection initialization with $modelName ...")
            
            val detector = ObjectDetection(
                context = context,
                modelPath = "models/$modelName",
                labelPath = "labels/labels.txt",
                confidenceThreshold = currentConfidenceThreshold,
                objectDetectionListener = detectionListener,
                message = { msg ->
                    lifecycleOwner.lifecycleScope.launch(Dispatchers.Main) {
                        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                    }
                }
            )
            
            detectionLock.withLock {
                objectDetection = detector
                lastLoadedModelName = modelName
            }
            
            Log.d(TAG, "ObjectDetection initialized successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ObjectDetection", e)
            withContext(Dispatchers.Main) {
                Toast.makeText(
                    context,
                    "Failed to load model: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    /**
     * Traite l'image reçue du CameraManager
     */
    override fun onImageAnalyzed(bitmap: Bitmap) {
        if (!isActive) return
        
        detectionLock.withLock {
            val currentDetector = objectDetection ?: return
            currentDetector.invoke(bitmap)
        }
    }

    /**
     * Nettoie les ressources de détection
     */
    fun cleanup() {
        isActive = false
        detectionLock.withLock {
            try {
                objectDetection?.close()
            } catch (_: Exception) {}
            objectDetection = null
        }
    }

    /**
     * Change le modèle de détection
     */
    suspend fun changeModel(newModelName: String) {
        if (newModelName == lastLoadedModelName) return
        
        cleanup()
        isActive = true
        initializeDetection()
    }
    
    /**
     * Modifie le seuil de confiance pour les détections
     * @param threshold Nouveau seuil de confiance (entre 0.0 et 1.0)
     */
    fun setConfidenceThreshold(threshold: Float) {
        val validThreshold = threshold.coerceIn(0.1f, 0.95f)
        if (validThreshold != currentConfidenceThreshold) {
            currentConfidenceThreshold = validThreshold
            detectionLock.withLock {
                objectDetection?.setConfidenceThreshold(validThreshold)
            }
            Log.d(TAG, "Seuil de confiance modifié: $validThreshold")
        }
    }

    companion object {
        private const val TAG = "DetectionManager"
    }
}
