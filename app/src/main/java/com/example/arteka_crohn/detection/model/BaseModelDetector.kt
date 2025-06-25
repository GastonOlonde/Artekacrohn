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
     * Initialise l'interpréteur TensorFlow Lite avec les options appropriées
     */
    protected fun initializeInterpreter() {
        val options = Interpreter.Options()
        var delegateAppliedInfo = ""

        // Choix d'un seul délégué à la fois pour éviter les conflits
        val useGpu = DetectionConfig.USE_GPU_DELEGATE // Rendre configurable

        if (useGpu) {
            try {
                // S'assurer que tout ancien délégué GPU a été fermé correctement
                closeGpuDelegate()
                
                // Forcer le GC avant de créer un nouveau délégué GPU
                System.gc()
                try { Thread.sleep(100) } catch (ignored: InterruptedException) {}
                
                // Récupérer le nom du modèle pour ajuster les paramètres GPU
                val modelName = getModelName().lowercase()

                val delegateOptions = org.tensorflow.lite.gpu.GpuDelegateFactory.Options().apply {
                    setPrecisionLossAllowed(true)
                    setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENCL)
                    setQuantizedModelsAllowed(true) // Meilleure performance pour modèles quantifiés
                }
                
                try {
                    gpuDelegateInstance = org.tensorflow.lite.gpu.GpuDelegate(delegateOptions)
                    options.addDelegate(gpuDelegateInstance)
                    delegateAppliedInfo = "Using GPU Delegate."
                    Log.d("BaseModelDetector", "GPU delegate added for $modelName")
                } catch (e: Exception) {
                    Log.e("BaseModelDetector", "Failed to initialize GPU delegate for $modelName: ${e.message}. Falling back to CPU.", e)
                    // En cas d'échec, on utilise le CPU
                    closeGpuDelegate() // Nettoyer les ressources partiellement initialisées
                    System.gc() // Forcer la libération de mémoire
                    try { Thread.sleep(100) } catch (ignored: InterruptedException) {}
                    
                    options.setUseNNAPI(false)
                    options.setNumThreads(4) // Utiliser plusieurs threads CPU
                    delegateAppliedInfo = "Using CPU with 4 threads (GPU init failed)."
                }
            } catch (e: Exception) {
                Log.e("BaseModelDetector", "Error configuring GPU delegate: ${e.message}", e)
                // Utiliser le CPU en fallback
                options.setUseNNAPI(false)
                options.setNumThreads(4)
                delegateAppliedInfo = "Using CPU fallback."
            }
        } else {
            // Configurer pour utiliser NNAPI
            try {
                val nnApiOptions = org.tensorflow.lite.nnapi.NnApiDelegate.Options().apply {
                    setUseNnapiCpu(true)
                    setExecutionPreference(org.tensorflow.lite.nnapi.NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
                }
                nnApiDelegateInstance = org.tensorflow.lite.nnapi.NnApiDelegate(nnApiOptions)
                options.addDelegate(nnApiDelegateInstance)
                delegateAppliedInfo = "Using NNAPI Delegate."
            } catch (e: Exception) {
                Log.e("BaseModelDetector", "Failed to initialize NNAPI delegate: ${e.message}", e)
                // En cas d'échec, utiliser le CPU
                closeNnapiDelegate() // Nettoyer les ressources partiellement initialisées
                options.setUseNNAPI(false)
                options.setNumThreads(4) // Utiliser plusieurs threads CPU
                delegateAppliedInfo = "Using CPU with 4 threads."
            }
        }

        // Créer l'interpréteur avec les options configurées
        modelBuffer?.let {
            try {
                // Récupérer le nom du modèle pour le logging
                val modelName = getModelName()
                Log.i("BaseModelDetector", "Creating interpreter for model: $modelName with $delegateAppliedInfo")
                
                // Optimisations pour les grands modèles
                val isLargeModel = modelName.toLowerCase().contains("yolo11")
                if (isLargeModel) {
                    // Pour les grands modèles comme YOLOv11
                    val modelSizeMB = it.array().size / (1024 * 1024)
                    Log.d("BaseModelDetector", "Large model detected ($modelSizeMB MB), applying optimizations")
                    
                    options.setAllowFp16PrecisionForFp32(true) // Réduire l'empreinte mémoire
                    options.setAllowBufferHandleOutput(true) // Améliorer la gestion mémoire
                    options.setCancellable(true) // Permettre l'annulation si nécessaire
                }
                
                // Créer l'interpréteur avec options optimisées
                interpreter = Interpreter(it, options)
                Log.i("BaseModelDetector", "Interpreter initialized successfully. $delegateAppliedInfo")
            } catch (e: Exception) {
                Log.e("BaseModelDetector", "Failed to create interpreter with delegate: ${e.message}", e)
                
                // Nettoyer toutes les ressources
                closeGpuDelegate()
                closeNnapiDelegate()
                
                // Force garbage collection
                System.gc()
                try { Thread.sleep(200) } catch (ignored: InterruptedException) {}
                
                // Réessayer avec CPU uniquement
                val cpuOptions = Interpreter.Options().apply {
                    setNumThreads(4)
                    // Pour les grands modèles, on peut quand même optimiser
                    if (getModelName().toLowerCase().contains("yolo11")) {
                        setAllowFp16PrecisionForFp32(true)
                    }
                }
                
                try {
                    Log.i("BaseModelDetector", "Retrying with CPU-only options")
                    interpreter = Interpreter(it, cpuOptions)
                    Log.i("BaseModelDetector", "Interpreter initialized with CPU fallback.")
                } catch (e2: Exception) {
                    Log.e("BaseModelDetector", "Failed to create interpreter even with CPU fallback: ${e2.message}", e2)
                    throw e2
                }
            }
        } ?: throw IllegalStateException("Model buffer is null")
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
            // Fermer l'interpréteur avant les délégués
            interpreter?.let {
                try {
                    it.close()
                    Log.d("BaseModelDetector", "Interpreter closed successfully")
                } catch (e: Exception) {
                    Log.e("BaseModelDetector", "Error closing interpreter", e)
                } finally {
                    interpreter = null
                }
            }
            
            // Utiliser les méthodes dédiées pour fermer les délégués
            closeGpuDelegate()
            closeNnapiDelegate()
            
            // Libérer le buffer du modèle
            modelBuffer = null
            
            Log.d("BaseModelDetector", "All resources closed successfully")
        } catch (e: Exception) {
            Log.e("BaseModelDetector", "Error during resource cleanup", e)
        }
    }
    
    /**
     * Ferme proprement le délégué GPU
     */
    private fun closeGpuDelegate() {
        gpuDelegateInstance?.let {
            try {
                it.close()
                Log.d("BaseModelDetector", "GPU delegate closed successfully")
            } catch (e: Exception) {
                Log.e("BaseModelDetector", "Error closing GPU delegate", e)
            } finally {
                gpuDelegateInstance = null
            }
        }
    }

    /**
     * Ferme proprement le délégué NNAPI
     */
    private fun closeNnapiDelegate() {
        nnApiDelegateInstance?.let {
            try {
                it.close()
                Log.d("BaseModelDetector", "NNAPI delegate closed successfully")
            } catch (e: Exception) {
                Log.e("BaseModelDetector", "Error closing NNAPI delegate", e)
            } finally {
                nnApiDelegateInstance = null
            }
        }
    }
    
    override fun isInitialized(): Boolean {
        return interpreter != null && _inputWidth > 0 && _inputHeight > 0 && outputShapes.isNotEmpty()
    }
    
    override fun getModelType(): ModelType {
        return _modelType
    }
}
