package com.example.arteka_crohn.detection.model

import android.content.Context
import android.util.Log
import com.example.arteka_crohn.MetaData.TEMP_CLASSES
import com.example.arteka_crohn.MetaData.extractNamesFromLabelFile
import com.example.arteka_crohn.MetaData.extractNamesFromMetadata
import com.example.arteka_crohn.Output0
import com.example.arteka_crohn.detection.config.DetectionConfig
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.util.*

/**
 * Classe abstraite implémentant les fonctionnalités communes à tous les détecteurs
 * Fournit les méthodes de base pour l'initialisation de l'interpréteur, la gestion des délégués,
 * le chargement des labels, et d'autres fonctionnalités partagées.
 */
abstract class BaseModelDetector : ModelDetector {
    protected var interpreter: Interpreter? = null
    protected var modelBuffer: MappedByteBuffer? = null
    protected var labels = mutableListOf<String>()
    
    // Chemins des fichiers
    protected lateinit var modelPath: String
    protected var labelPath: String? = null
    
    // Contexte Android pour accéder aux ressources
    protected lateinit var appContext: Context
    
    // Dimensions du modèle
    protected var _inputWidth = DetectionConfig.DEFAULT_INPUT_WIDTH
    protected var _inputHeight = DetectionConfig.DEFAULT_INPUT_HEIGHT
    protected var outputShapes: List<IntArray> = listOf()
    
    // Propriétés du modèle
    protected var isModelQuantized = false
    
    // Délégués pour l'accélération matérielle
    private var gpuDelegateInstance: org.tensorflow.lite.gpu.GpuDelegate? = null
    private var nnApiDelegateInstance: org.tensorflow.lite.nnapi.NnApiDelegate? = null
    
    // Callback pour les messages
    protected lateinit var onMessage: (String) -> Unit
    
    // Type de modèle détecté
    protected var _modelType = ModelType.UNKNOWN
    
    override fun initialize(context: Context, modelPath: String, labelPath: String?) {
        this.modelPath = modelPath
        this.labelPath = labelPath
        this.appContext = context
        
        try {
            // Chargement du fichier modèle
            modelBuffer = FileUtil.loadMappedFile(context, modelPath)
            
            // Initialisation de l'interpréteur avec les délégués appropriés
            initializeInterpreter()
            
            // Extraction des dimensions du modèle
            extractModelDimensions()
            
            // Auto-détection du type de modèle
            autoDetectModelType()
            
            // Chargement des labels
            labels.addAll(loadLabels())
            
            Log.i("BaseModelDetector", "Initialized model: ${getModelName()} as ${_modelType.name}")
        } catch (e: Exception) {
            Log.e("BaseModelDetector", "Error initializing model: ${e.message}", e)
            onMessage("Failed to load model: ${e.message}")
            throw e
        }
    }
    
    /**
     * Initialise l'interpréteur TensorFlow Lite avec les délégués appropriés
     */
    protected fun initializeInterpreter() {
        val options = Interpreter.Options()
        var delegateAppliedInfo = ""

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
                Log.d("BaseModelDetector", "GPU delegate added.")
            } catch (e: Throwable) {
                Log.w("BaseModelDetector", "GPU Delegate creation/configuration failed. Falling back to CPU.", e)
                gpuDelegateInstance?.close()
                gpuDelegateInstance = null
            }
        } else {
            try {
                nnApiDelegateInstance = org.tensorflow.lite.nnapi.NnApiDelegate()
                options.addDelegate(nnApiDelegateInstance)
                delegateAppliedInfo = "Using NNAPI Delegate."
                Log.d("BaseModelDetector", "NNAPI delegate added.")
            } catch (e: Throwable) {
                Log.w("BaseModelDetector", "NNAPI Delegate creation/configuration failed. Falling back to CPU.", e)
                nnApiDelegateInstance?.close()
                nnApiDelegateInstance = null
            }
        }

        // Configuration des threads pour le CPU fallback
        options.setNumThreads(4)

        try {
            val modelBuf = modelBuffer ?: throw IllegalStateException("Model buffer not initialized")
            interpreter = Interpreter(modelBuf, options)
            Log.i("BaseModelDetector", "Interpreter initialized with $delegateAppliedInfo")
        } catch (e: Exception) {
            // Essai de fallback sans délégués en cas d'échec
            try {
                val fallbackOptions = Interpreter.Options().setNumThreads(4)
                interpreter = Interpreter(modelBuffer!!, fallbackOptions)
                Log.w("BaseModelDetector", "Fallback to CPU: ${e.message}")
            } catch (fallbackEx: Exception) {
                Log.e("BaseModelDetector", "Failed to initialize interpreter: ${fallbackEx.message}")
                throw fallbackEx
            }
        }
    }
    
    /**
     * Extrait les dimensions d'entrée et de sortie du modèle
     */
    protected fun extractModelDimensions() {
        val interp = interpreter ?: throw RuntimeException("Interpreter not initialized")
        
        // Extraction des dimensions d'entrée
        val inputShape = interp.getInputTensor(0).shape()
        
        // Détection si le modèle est quantifié
        val inputType = interp.getInputTensor(0).dataType()
        isModelQuantized = inputType == org.tensorflow.lite.DataType.UINT8
        
        Log.d("BaseModelDetector", "Input tensor type: $inputType, isQuantized: $isModelQuantized")
        
        // Format typique [1, height, width, channels] ou [1, channels, height, width]
        if (inputShape.size == 4) {
            if (inputShape[1] == 3) {
                // Format [1, channels, height, width]
                _inputHeight = inputShape[2]
                _inputWidth = inputShape[3]
            } else {
                // Format [1, height, width, channels]
                _inputHeight = inputShape[1]
                _inputWidth = inputShape[2]
            }
        }
        
        // Extraction des dimensions de sortie (tous les tenseurs)
        val outputCount = interp.outputTensorCount
        val shapes = mutableListOf<IntArray>()
        
        for (i in 0 until outputCount) {
            shapes.add(interp.getOutputTensor(i).shape())
        }
        
        outputShapes = shapes
        
        Log.d("BaseModelDetector", "Model input dimensions: ${_inputWidth}x${_inputHeight}")
        Log.d("BaseModelDetector", "Model output shapes: ${outputShapes.joinToString { it.contentToString() }}")
    }
    
    /**
     * Détecte automatiquement le type de modèle
     */
    protected fun autoDetectModelType() {
        // Utiliser le nom de fichier et les shapes pour détecter le type de modèle
        _modelType = ModelType.detectModelType(
            getModelName(),
            interpreter?.getInputTensor(0)?.shape() ?: intArrayOf(),
            outputShapes
        )
    }
    
    /**
     * Charge les labels depuis les métadonnées ou depuis un fichier externe
     */
    override fun loadLabels(): List<String> {
        val labelsList = mutableListOf<String>()
        
        // 1. Essayer d'extraire depuis les métadonnées du modèle
        modelBuffer?.let {
            val metadataLabels = extractNamesFromMetadata(it)
            if (metadataLabels.isNotEmpty()) {
                return metadataLabels
            }
        }
        
        // 2. Essayer de charger depuis le fichier de labels
        labelPath?.let { path ->
            try {
                val fileLabels = extractNamesFromLabelFile(appContext, path)
                if (fileLabels.isNotEmpty()) {
                    labelsList.addAll(fileLabels)
                    return labelsList
                }
            } catch (e: Exception) {
                Log.w("BaseModelDetector", "Failed to load labels from file: ${e.message}")
            }
        }
        
        // 3. Utiliser les classes temporaires par défaut
        if (labelsList.isEmpty()) {
            labelsList.addAll(TEMP_CLASSES)
            onMessage("No labels found. Using default labels.")
        }
        
        return labelsList
    }
    
    /**
     * Retourne le nom du fichier modèle sans le chemin
     */
    protected fun getModelName(): String {
        return modelPath.substringAfterLast("/")
    }
    
    override fun getInputDimensions(): Pair<Int, Int> {
        return Pair(_inputWidth, _inputHeight)
    }
    
    /**
     * Obtient la largeur d'entrée attendue par le modèle
     * @return Largeur d'entrée en pixels
     */
    override fun getInputWidth(): Int {
        return _inputWidth
    }
    
    /**
     * Obtient la hauteur d'entrée attendue par le modèle
     * @return Hauteur d'entrée en pixels
     */
    override fun getInputHeight(): Int {
        return _inputHeight
    }
    
    /**
     * Indique si le modèle utilise des entrées quantifiées (UINT8)
     * @return true si le modèle est quantifié, false sinon (Float32)
     */
    override fun isQuantized(): Boolean {
        return isModelQuantized
    }
    
    override fun requiresNormalization(): Boolean {
        return !isModelQuantized
    }
    
    override fun getNormalizationParams(): Pair<Float, Float> {
        return Pair(DetectionConfig.INPUT_MEAN, DetectionConfig.INPUT_STANDARD_DEVIATION)
    }
    
    /**
     * Obtient les valeurs de normalisation pour le prétraitement des images
     * @return Triple (moyenne, écart-type, scale) pour la normalisation
     */
    override fun getNormalizationValues(): Triple<Float, Float, Float> {
        val scale = if (isModelQuantized) 1.0f else 1.0f
        return Triple(
            DetectionConfig.INPUT_MEAN, 
            DetectionConfig.INPUT_STANDARD_DEVIATION,
            scale
        )
    }
    
    override fun close() {
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
            Log.e("BaseModelDetector", "Error closing interpreter", e)
        }
    }
    
    override fun isInitialized(): Boolean {
        return interpreter != null && _inputWidth > 0 && _inputHeight > 0 && outputShapes.isNotEmpty()
    }
    
    override fun getModelType(): ModelType {
        return _modelType
    }
}
