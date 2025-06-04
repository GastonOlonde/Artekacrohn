package com.example.arteka_crohn.segmentation

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.example.arteka_crohn.segmentation.ApiSegmentationResult
import com.example.arteka_crohn.segmentation.InstanceSegmentation
import com.example.arteka_crohn.camera.ImageAnalysisCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Gestionnaire de segmentation responsable de l'initialisation du modèle,
 * du traitement des images et de la gestion des résultats de segmentation.
 */
class SegmentationManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val modelName: String,
    private val segmentationListener: InstanceSegmentationListener
) : ImageAnalysisCallback {

    private var instanceSegmentation: InstanceSegmentation? = null
    private val segmentationLock = ReentrantLock()
    
    @Volatile private var isActive = true
    private var lastLoadedModelName: String = modelName

    /**
     * Initialise le modèle de segmentation
     */
    suspend fun initializeSegmentation() {
        if (!isActive) return
        
        try {
            Log.d(TAG, "Starting InstanceSegmentation initialization with $modelName ...")
            
            val segmenter = InstanceSegmentation(
                context = context,
                modelPath = "models/$modelName",
                labelPath = "labels/labels.txt",
                instanceSegmentationListener = segmentationListener,
                message = { msg ->
                    lifecycleOwner.lifecycleScope.launch(Dispatchers.Main) {
                        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                    }
                }
            )
            
            segmentationLock.withLock {
                instanceSegmentation = segmenter
                lastLoadedModelName = modelName
            }
            
            Log.d(TAG, "InstanceSegmentation initialized successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing InstanceSegmentation", e)
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
        
        segmentationLock.withLock {
            val currentSegmenter = instanceSegmentation ?: return
            currentSegmenter.invoke(bitmap)
        }
    }

    /**
     * Nettoie les ressources de segmentation
     */
    fun cleanup() {
        isActive = false
        segmentationLock.withLock {
            try {
                instanceSegmentation?.close()
            } catch (_: Exception) {}
            instanceSegmentation = null
        }
    }

    /**
     * Change le modèle de segmentation
     */
    suspend fun changeModel(newModelName: String) {
        if (newModelName == lastLoadedModelName) return
        
        cleanup()
        isActive = true
        initializeSegmentation()
    }

    companion object {
        private const val TAG = "SegmentationManager"
    }
}
