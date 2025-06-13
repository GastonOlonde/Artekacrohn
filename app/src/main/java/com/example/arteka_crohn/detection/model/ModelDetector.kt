package com.example.arteka_crohn.detection.model

import android.content.Context
import com.example.arteka_crohn.Output0
import java.nio.ByteBuffer

/**
 * Interface définissant le contrat pour tous les détecteurs de modèles
 * Permet d'implémenter différentes stratégies de détection selon le type de modèle
 */
interface ModelDetector {
    /**
     * Initialise le détecteur avec le contexte et les chemins des fichiers nécessaires
     * @param context Contexte Android
     * @param modelPath Chemin vers le fichier du modèle TFLite
     * @param labelPath Chemin optionnel vers le fichier de labels
     */
    fun initialize(context: Context, modelPath: String, labelPath: String?)
    
    /**
     * Charge les labels depuis les métadonnées du modèle ou depuis un fichier externe
     * @return Liste des labels/classes reconnus par le modèle
     */
    fun loadLabels(): List<String>
    
    /**
     * Exécute l'inférence du modèle avec les données d'entrée prétraitées
     * @param inputBuffers Tableaux de ByteBuffer contenant les données d'entrée prétraitées
     * @return Données brutes de sortie du modèle
     */
    fun runInference(inputBuffers: Array<ByteBuffer>): Any
    
    /**
     * Post-traite les données brutes de sortie du modèle en détections utilisables
     * @param rawOutput Données brutes de sortie du modèle
     * @param confidenceThreshold Seuil de confiance minimum pour les détections
     * @return Liste des détections (boîtes englobantes, classes, scores)
     */
    fun processOutput(rawOutput: Any, confidenceThreshold: Float): List<Output0>
    
    /**
     * Obtient les dimensions attendues pour l'entrée du modèle
     * @return Paire (width, height) pour les dimensions d'entrée
     */
    fun getInputDimensions(): Pair<Int, Int>
    
    /**
     * Détermine si le modèle nécessite une normalisation des entrées
     * @return true si le modèle nécessite une normalisation
     */
    fun requiresNormalization(): Boolean
    
    /**
     * Obtient les valeurs de normalisation (moyenne, écart-type)
     * @return Paire (moyenne, écart-type) pour la normalisation
     */
    fun getNormalizationParams(): Pair<Float, Float>
    
    /**
     * Libère les ressources utilisées par le détecteur
     */
    fun close()
    
    /**
     * Vérifie si le détecteur est correctement initialisé
     * @return true si le détecteur est initialisé et prêt à être utilisé
     */
    fun isInitialized(): Boolean
    
    /**
     * Obtient le type de modèle
     * @return Type de modèle (YOLO_V8, MOBILENET_SSD, etc.)
     */
    fun getModelType(): ModelType
    
    /**
     * Obtient la largeur d'entrée attendue par le modèle
     * @return Largeur d'entrée en pixels
     */
    fun getInputWidth(): Int
    
    /**
     * Obtient la hauteur d'entrée attendue par le modèle
     * @return Hauteur d'entrée en pixels
     */
    fun getInputHeight(): Int
    
    /**
     * Indique si le modèle utilise des entrées quantifiées (UINT8)
     * @return true si le modèle est quantifié, false sinon (Float32)
     */
    fun isQuantized(): Boolean
    
    /**
     * Obtient les valeurs de normalisation pour le prétraitement des images
     * @return Triple (moyenne, écart-type, scale) pour la normalisation
     */
    fun getNormalizationValues(): Triple<Float, Float, Float>
}
