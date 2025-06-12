package com.example.arteka_crohn.detection.model

import android.content.Context
import android.util.Log
import com.example.arteka_crohn.MetaData.TEMP_CLASSES
import com.example.arteka_crohn.MetaData.extractNamesFromLabelFile
import com.example.arteka_crohn.MetaData.extractNamesFromMetadata
import com.example.arteka_crohn.detection.config.DetectionConfig
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer

/**
 * Classe responsable de la gestion du modèle TensorFlow Lite pour la détection d'objets
 * Version simplifiée par rapport à TensorFlowLiteModel pour la segmentation
 */
class TensorFlowLiteDetectionModel(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String?,
    private val onMessage: (String) -> Unit
) {
    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()
    private var modelBuffer: MappedByteBuffer? = null
    
    // Propriétés du modèle
    var inputWidth = DetectionConfig.DEFAULT_INPUT_WIDTH
        private set
    var inputHeight = DetectionConfig.DEFAULT_INPUT_HEIGHT
        private set
    private var outputShape: IntArray = intArrayOf()
    
    // Delegates pour l'accélération matérielle
    private var gpuDelegateInstance: org.tensorflow.lite.gpu.GpuDelegate? = null
    private var nnApiDelegateInstance: org.tensorflow.lite.nnapi.NnApiDelegate? = null
    
    init {
        try {
            initializeInterpreter()
        } catch (e: Exception) {
            Log.e("TFLite", "Error initializing interpreter", e)
            onMessage("Failed to load model: ${e.message}")
            throw e
        }
    }

    private fun initializeInterpreter() {
        val options = Interpreter.Options()
        var delegateAppliedInfo = "No delegate tried or fallback to CPU."

        // Choix d'un seul délégué à la fois pour éviter les conflits
        val useGpu = true // Préférer GPU, changer à false pour utiliser NNAPI

        if (useGpu) {
            try {
                val delegateOptions = org.tensorflow.lite.gpu.GpuDelegateFactory.Options().apply {
                    setPrecisionLossAllowed(true)
                    setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENCL)
                }
                gpuDelegateInstance = org.tensorflow.lite.gpu.GpuDelegate(delegateOptions)

                options.addDelegate(gpuDelegateInstance)
                delegateAppliedInfo = "Using GPU Delegate."
                Log.d("TFLite", "GPU delegate added.")
            } catch (e: Throwable) {
                Log.w("TFLite_GPU", "GPU Delegate creation/configuration failed. Falling back to CPU.", e)
                gpuDelegateInstance?.close()
                gpuDelegateInstance = null
                // XNNPACK pour CPU
                options.setUseXNNPACK(true)
                delegateAppliedInfo = "GPU delegate failed. Using CPU with XNNPACK."
            }
        } else {
            try {
                val nnApiDelegate = org.tensorflow.lite.nnapi.NnApiDelegate()
                options.addDelegate(nnApiDelegate)
                nnApiDelegateInstance = nnApiDelegate

                delegateAppliedInfo = "Using NNAPI Delegate."
                Log.d("TFLite", "NNAPI delegate added.")
            } catch (e: Exception) {
                Log.w("TFLite_NNAPI", "NNAPI Delegate not available or failed: ${e.message}")
                nnApiDelegateInstance?.close()
                nnApiDelegateInstance = null
                // XNNPACK pour CPU
                options.setUseXNNPACK(true)
                delegateAppliedInfo = "NNAPI delegate failed. Using CPU with XNNPACK."
            }
        }

        // Si aucun délégué n'est configuré, utiliser XNNPACK
        if (gpuDelegateInstance == null && nnApiDelegateInstance == null) {
            options.setUseXNNPACK(true)
            delegateAppliedInfo = "Using CPU with XNNPACK."
        }

        Log.d("TFLiteConfig", "Delegate status: $delegateAppliedInfo")

        // Chargement du modèle
        modelBuffer = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(modelBuffer!!, options)

        Log.i("TFLite", "Interpreter initialized with $delegateAppliedInfo")

        // Chargement des labels
        labels.addAll(extractNamesFromMetadata(modelBuffer!!))
        if (labels.isEmpty()) {
            if (labelPath == null) {
                onMessage("Model not contains metadata, provide LABELS_PATH in Constants.kt")
                labels.addAll(TEMP_CLASSES)
            } else {
                labels.addAll(extractNamesFromLabelFile(context, labelPath))
            }
        }

        // Récupération des dimensions du modèle
        extractModelDimensions()
    }

    private fun extractModelDimensions() {
        val interp = interpreter ?: return
        
        // Extraction des dimensions d'entrée
        val inputShape = interp.getInputTensor(0)?.shape()
        if (inputShape != null) {
            // Format typique [1, height, width, channels] ou [1, channels, height, width]
            if (inputShape.size == 4) {
                if (inputShape[1] == 3) {
                    // Format [1, channels, height, width]
                    inputHeight = inputShape[2]
                    inputWidth = inputShape[3]
                } else {
                    // Format [1, height, width, channels]
                    inputHeight = inputShape[1]
                    inputWidth = inputShape[2]
                }
            }
        }

        // Extraction des dimensions de sortie
        val outputTensor = interp.getOutputTensor(0)
        outputShape = outputTensor?.shape() ?: intArrayOf()
        
        Log.d("TFLiteDetection", "Model input dimensions: ${inputWidth}x${inputHeight}")
        Log.d("TFLiteDetection", "Model output shape: ${outputShape.joinToString()}")
    }

    fun runInference(inputs: Array<ByteBuffer>, outputs: Map<Int, Any>) {
        interpreter?.runForMultipleInputsOutputs(inputs, outputs)
            ?: throw IllegalStateException("Interpreter not initialized")
    }

    fun getLabels(): List<String> = labels

    fun getOutputShape(): IntArray = outputShape
    
    /**
     * Retourne le nom du fichier modèle sans le chemin
     */
    fun getModelName(): String {
        return modelPath.substringAfterLast("/")
    }

    fun close() {
        try {
            interpreter?.close()
            interpreter = null
            
            gpuDelegateInstance?.close()
            gpuDelegateInstance = null
            
            nnApiDelegateInstance?.close()
            nnApiDelegateInstance = null
            
            modelBuffer = null
            
            // Suggestion au garbage collector
            System.gc()
        } catch (e: Exception) {
            Log.e("TFLite", "Error closing interpreter", e)
        }
    }

    fun isInitialized(): Boolean {
        return interpreter != null && inputWidth > 0 && inputHeight > 0 && outputShape.isNotEmpty()
    }

    companion object {
        val INPUT_IMAGE_TYPE = DetectionConfig.INPUT_IMAGE_TYPE
        val OUTPUT_IMAGE_TYPE = DetectionConfig.OUTPUT_IMAGE_TYPE
    }
}