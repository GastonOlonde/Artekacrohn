package com.example.arteka_crohn.segmentation.model

import android.content.Context
import android.util.Log
import com.example.arteka_crohn.MetaData.TEMP_CLASSES
import com.example.arteka_crohn.MetaData.extractNamesFromLabelFile
import com.example.arteka_crohn.MetaData.extractNamesFromMetadata
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer

/**
 * Classe responsable de la gestion du modèle TensorFlow Lite
 * Respecte le principe de responsabilité unique (S dans SOLID)
 */
class TensorFlowLiteModel(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String?,
    private val onMessage: (String) -> Unit
) {
    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()
    private var modelBuffer: MappedByteBuffer? = null
    
    // Propriétés du modèle
    var tensorWidth = 0
        private set
    var tensorHeight = 0
        private set
    var numChannel = 0
        private set
    var numElements = 0
        private set
    var xPoints = 0
        private set
    var yPoints = 0
        private set
    var masksNum = 0
        private set
    
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
        
        val inputShape = interp.getInputTensor(0)?.shape()
        val outputShape0 = interp.getOutputTensor(0)?.shape()
        val outputShape1 = interp.getOutputTensor(1)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape0 != null) {
            numChannel = outputShape0[1]
            numElements = outputShape0[2]
        }

        if (outputShape1 != null) {
            if (outputShape1[1] == 32) {
                masksNum = outputShape1[1]
                xPoints = outputShape1[2]
                yPoints = outputShape1[3]
            } else {
                xPoints = outputShape1[1]
                yPoints = outputShape1[2]
                masksNum = outputShape1[3]
            }
        }
    }

    fun runInference(inputs: Array<ByteBuffer>, outputs: Map<Int, Any>) {
        interpreter?.runForMultipleInputsOutputs(inputs, outputs)
            ?: throw IllegalStateException("Interpreter not initialized")
    }

    fun getLabels(): List<String> = labels

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
        return interpreter != null && tensorWidth != 0 && tensorHeight != 0 &&
                numChannel != 0 && numElements != 0 &&
                xPoints != 0 && yPoints != 0 && masksNum != 0
    }

    companion object {
        val INPUT_IMAGE_TYPE = DataType.FLOAT32
        val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
    }
}
