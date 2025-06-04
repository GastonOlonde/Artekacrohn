package com.example.arteka_crohn.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors

/**
 * Interface de callback pour l'analyse d'image
 */
interface ImageAnalysisCallback {
    fun onImageAnalyzed(bitmap: Bitmap)
}

/**
 * Gestionnaire de caméra responsable de l'initialisation, la configuration
 * et l'analyse des images de la caméra.
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val imageAnalysisCallback: ImageAnalysisCallback
) {
    private var cameraProvider: ProcessCameraProvider? = null
    private val imageAnalysisExecutor = Executors.newSingleThreadExecutor()
    
    @Volatile private var isActive = true

    /**
     * Démarre la caméra et configure l'analyse d'image
     */
    fun startCamera() {
        if (!isActive) return
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                
                // Configuration du preview
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                
                // Configuration de l'analyseur d'image
                val imageAnalyzerUseCase = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also {
                        it.setAnalyzer(imageAnalysisExecutor, ImageAnalyzer())
                    }
                
                // Sélection de la caméra arrière par défaut
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                
                // Liaison des cas d'utilisation au cycle de vie
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzerUseCase
                )
                
                Log.d(TAG, "Camera started successfully")
            } catch (exc: Exception) {
                Log.e(TAG, "Failed to start camera", exc)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Nettoie les ressources de la caméra
     */
    fun cleanup() {
        isActive = false
        try {
            imageAnalysisExecutor.shutdownNow()
        } catch (_: Exception) {}
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {}
        cameraProvider = null
    }

    /**
     * Classe interne pour l'analyse d'image
     */
    private inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            if (!isActive) {
                imageProxy.close()
                return
            }
            
            // Création d'un bitmap à partir de l'image
            val bitmapBuffer = createBitmap(imageProxy.width, imageProxy.height)
            imageProxy.use { proxy ->
                val buffer = proxy.planes[0].buffer
                buffer.rewind()
                bitmapBuffer.copyPixelsFromBuffer(buffer)
            }
            
            // Rotation du bitmap selon l'orientation de l'appareil
            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            }
            
            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer,
                0,
                0,
                bitmapBuffer.width,
                bitmapBuffer.height,
                matrix,
                true
            )
            
            // Envoi du bitmap au callback
            imageAnalysisCallback.onImageAnalyzed(rotatedBitmap)
        }
    }

    companion object {
        private const val TAG = "CameraManager"
    }
}
