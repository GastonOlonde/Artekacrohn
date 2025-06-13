package com.example.arteka_crohn.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.AspectRatio
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
    private val imageAnalysisCallback: ImageAnalysisCallback
) {
    private var cameraProvider: ProcessCameraProvider? = null
    private val imageAnalysisExecutor = Executors.newSingleThreadExecutor()
    
    @Volatile private var isActive = true
    private var isInitialized = false
    
    // Stockage de la dernière image capturée pour l'affichage des détections
    private var lastFrame: Bitmap? = null

    /**
     * Démarre la caméra et configure l'analyse d'image
     */
    fun startCamera(
        lifecycleOwner: LifecycleOwner,
        surfaceProvider: Preview.SurfaceProvider
    ) {
        if (!isActive) return
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                if (!isActive) return@addListener

                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases(lifecycleOwner, surfaceProvider)
                isInitialized = true
                Log.d(TAG, "Camera started successfully")
            } catch (exc: Exception) {
                Log.e(TAG, "Failed to start camera", exc)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Lie les cas d'utilisation de la caméra
     */
    private fun bindCameraUseCases(
        lifecycleOwner: LifecycleOwner,
        surfaceProvider: Preview.SurfaceProvider
    ) {
        if (!isActive) return
        
        val cameraProvider = cameraProvider ?: return
        
        try {
            // Configuration du preview avec ratio 4:3
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(surfaceProvider)
                }
            
            // Configuration de l'analyseur d'image avec ratio 4:3
            val imageAnalyzerUseCase = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(imageAnalysisExecutor, ImageAnalyzer())
                }
            
            // Sélection de la caméra arrière par défaut
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            // Liaison des cas d'utilisation au cycle de vie
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzerUseCase
            )
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    /**
     * Reprend la caméra après une pause
     */
    fun resumeCamera() {
        isActive = true
    }

    /**
     * Nettoie les ressources de la caméra
     */
    fun cleanup() {
        isActive = false
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {}
        
        try {
            if (!imageAnalysisExecutor.isShutdown) {
                imageAnalysisExecutor.shutdownNow()
            }
        } catch (_: Exception) {}
        
        // Libérer la mémoire de la dernière image
        lastFrame?.recycle()
        lastFrame = null
        
        cameraProvider = null
        isInitialized = false
    }
    
    /**
     * Récupère la dernière image capturée par la caméra
     * @return La dernière image capturée ou null si aucune image n'est disponible
     */
    fun getLastFrame(): Bitmap? {
        return lastFrame
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

            // Recycler l'ancienne image si elle existe
            lastFrame?.recycle()
            
            // Stocker la nouvelle image
            lastFrame = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)

            // Envoi du bitmap au callback
            imageAnalysisCallback.onImageAnalyzed(rotatedBitmap)
        }
    }

    companion object {
        private const val TAG = "CameraManager"
        
        // Constantes pour les ratios d'aspect de la caméra
        private const val RATIO_4_3 = AspectRatio.RATIO_4_3
        private const val RATIO_16_9 = AspectRatio.RATIO_16_9
    }
}
