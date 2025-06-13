package com.example.arteka_crohn.detection.model

import android.content.Context
import android.util.Log
import com.example.arteka_crohn.Output0
import java.nio.ByteBuffer

/**
 * Factory pour créer et initialiser automatiquement le bon détecteur
 * selon le type de modèle détecté
 */
class ModelDetectorFactory {
    private val TAG = "ModelDetectorFactory"
    
    /**
     * Crée et initialise un détecteur adapté au modèle spécifié
     * @param context Contexte Android
     * @param modelPath Chemin vers le fichier du modèle
     * @param labelPath Chemin optionnel vers le fichier de labels
     * @param onMessage Callback pour les messages d'information/erreur
     * @return Détecteur initialisé et adapté au type de modèle
     */
    fun createDetector(
        context: Context,
        modelPath: String,
        labelPath: String?,
        onMessage: (String) -> Unit
    ): ModelDetector {
        Log.d(TAG, "Creating detector for model: $modelPath")
        
        // Premières tentatives de détection basées sur le nom du fichier
        val modelName = modelPath.substringAfterLast("/")
        val initialType = ModelType.detectFromFilename(modelName)
        
        Log.d(TAG, "Initial model type detection from filename: ${initialType.name}")
        
        // Créer le détecteur approprié selon le type détecté
        val detector = when (initialType) {
            ModelType.YOLO_V8 -> {
                Log.d(TAG, "Creating YOLO detector")
                YoloModelDetector(onMessage)
            }
            ModelType.MOBILENET_SSD -> {
                Log.d(TAG, "Creating MobileNet SSD detector")
                MobileNetSSDModelDetector(onMessage)
            }
            ModelType.UNKNOWN -> {
                // Si le type est inconnu, on crée un détecteur temporaire pour extraire les infos
                Log.d(TAG, "Unknown model type, creating temporary detector to extract information")
                createTemporaryDetectorForInspection(onMessage)
            }
        }
        
        // Initialiser le détecteur
        try {
            detector.initialize(context, modelPath, labelPath)
            
            // Vérifier le type après analyse des tenseurs (au cas où l'initial était UNKNOWN)
            val finalType = detector.getModelType()
            
            if (initialType == ModelType.UNKNOWN && finalType != ModelType.UNKNOWN) {
                // Si on a pu détecter le type après inspection des tenseurs,
                // recréer le bon détecteur
                Log.d(TAG, "Model type detected after inspection: ${finalType.name}")
                return createSpecificDetector(context, modelPath, labelPath, finalType, onMessage)
            }
            
            return detector
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing detector: ${e.message}", e)
            onMessage("Failed to initialize model detector: ${e.message}")
            throw e
        }
    }
    
    /**
     * Crée un détecteur temporaire pour analyser le modèle et ses tenseurs
     * afin de déterminer son type exact
     */
    private fun createTemporaryDetectorForInspection(onMessage: (String) -> Unit): ModelDetector {
        // On utilise un détecteur de base pour l'inspection
        return object : BaseModelDetector() {
            init {
                this.onMessage = onMessage
            }
            
            override fun runInference(inputBuffers: Array<ByteBuffer>): Any {
                throw UnsupportedOperationException("Temporary detector is for inspection only")
            }
            
            override fun processOutput(rawOutput: Any, confidenceThreshold: Float): List<Output0> {
                throw UnsupportedOperationException("Temporary detector is for inspection only")
            }
        }
    }
    
    /**
     * Crée un détecteur spécifique selon le type détecté après inspection
     */
    private fun createSpecificDetector(
        context: Context,
        modelPath: String,
        labelPath: String?,
        modelType: ModelType,
        onMessage: (String) -> Unit
    ): ModelDetector {
        val detector = when (modelType) {
            ModelType.YOLO_V8 -> YoloModelDetector(onMessage)
            ModelType.MOBILENET_SSD -> MobileNetSSDModelDetector(onMessage)
            else -> throw IllegalArgumentException("Unsupported model type: ${modelType.name}")
        }
        
        detector.initialize(context, modelPath, labelPath)
        return detector
    }
}
