package com.example.arteka_crohn.detection.utils

import android.util.Log
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.charset.StandardCharsets

/**
 * Classe utilitaire pour extraire les métadonnées des modèles TensorFlow Lite
 * avec une gestion robuste des erreurs
 */
class ModelMetadataExtractor(private val modelBuffer: MappedByteBuffer) {
    private val TAG = "ModelMetadataExtractor"
    
    // MetadataExtractor de TensorFlow Lite
    private val metadataExtractor: MetadataExtractor?
    
    init {
        // Initialisation du MetadataExtractor avec gestion des erreurs
        metadataExtractor = try {
            MetadataExtractor(modelBuffer)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to extract metadata: ${e.message}")
            null
        }
    }
    
    /**
     * Vérifie si le modèle contient des métadonnées
     * @return true si le modèle contient des métadonnées accessibles
     */
    fun hasMetadata(): Boolean {
        return metadataExtractor != null && metadataExtractor.hasMetadata()
    }
    
    /**
     * Extrait les noms des classes (labels) depuis les métadonnées du modèle
     * @return Liste des noms de classes ou liste vide si non disponible
     */
    fun extractLabels(): List<String> {
        if (!hasMetadata()) return emptyList()
        
        try {
            // Tenter d'obtenir les labels associés au premier tenseur de sortie
            val outputTensorMetadata = metadataExtractor?.getOutputTensorMetadata(0)
            if (outputTensorMetadata != null) {
                // Parcourir les fichiers associés pour trouver les labels
                for (i in 0 until outputTensorMetadata.associatedFilesLength()) {
                    val file = outputTensorMetadata.associatedFiles(i)
                    if (file != null && file.name() != null && file.name().contains("label", ignoreCase = true)) {
                        val labelStream = metadataExtractor?.getAssociatedFile(file.name())
                        if (labelStream != null) {
                            val labelBuffer = inputStreamToByteBuffer(labelStream)
                            return parseLabelsFromBuffer(labelBuffer)
                        }
                    }
                }
            }
            
            // Tenter d'obtenir les labels depuis les métadonnées générales
            val associatedFiles = metadataExtractor?.associatedFileNames
            if (associatedFiles != null) {
                for (fileName in associatedFiles) {
                    if (fileName.contains("label", ignoreCase = true)) {
                        val labelStream = metadataExtractor?.getAssociatedFile(fileName)
                        if (labelStream != null) {
                            val labelBuffer = inputStreamToByteBuffer(labelStream)
                            return parseLabelsFromBuffer(labelBuffer)
                        }
                    }
                }
            }
            
            // Si aucun fichier de labels n'est trouvé, chercher dans les métadonnées du modèle
            val modelMetadata = metadataExtractor?.modelMetadata
            if (modelMetadata != null) {
                val description = modelMetadata.description()
                if (description != null && description.contains(",")) {
                    // Certains modèles stockent les labels directement dans la description, séparés par des virgules
                    return description.split(",").map { it.trim() }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting labels from metadata: ${e.message}", e)
        }
        
        return emptyList()
    }
    
    /**
     * Extrait les ancres (anchors) pour les modèles de détection qui en utilisent
     * @return Liste des ancres ou null si non disponible
     */
    fun extractAnchors(): List<Float>? {
        if (!hasMetadata()) return null
        
        try {
            // Tenter d'obtenir les ancres depuis les métadonnées du modèle
            val associatedFiles = metadataExtractor?.associatedFileNames
            if (associatedFiles != null) {
                for (fileName in associatedFiles) {
                    if (fileName.contains("anchor", ignoreCase = true)) {
                        val anchorStream = metadataExtractor?.getAssociatedFile(fileName)
                        if (anchorStream != null) {
                            val anchorBuffer = inputStreamToByteBuffer(anchorStream)
                            return parseAnchorsFromBuffer(anchorBuffer)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting anchors from metadata: ${e.message}", e)
        }
        
        return null
    }
    
    /**
     * Extrait la description du modèle depuis les métadonnées
     * @return Description du modèle ou null si non disponible
     */
    fun getModelDescription(): String? {
        if (!hasMetadata()) return null
        
        try {
            val modelMetadata = metadataExtractor?.modelMetadata
            if (modelMetadata != null) {
                return modelMetadata.description()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting model description: ${e.message}", e)
        }
        
        return null
    }
    
    /**
     * Convertit un InputStream en ByteBuffer
     * @param inputStream InputStream à convertir
     * @return ByteBuffer contenant les données de l'InputStream
     */
    private fun inputStreamToByteBuffer(inputStream: InputStream): ByteBuffer {
        val bytes = inputStream.readBytes()
        val buffer = ByteBuffer.allocateDirect(bytes.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return buffer
    }
    
    /**
     * Extrait les informations des tenseurs d'entrée du modèle
     * @return Map des noms de tenseurs d'entrée avec leurs descriptions
     */
    fun getInputTensorInfo(): Map<String, String> {
        val result = mutableMapOf<String, String>()
        
        if (!hasMetadata()) return result
        
        try {
            val inputTensorCount = metadataExtractor?.inputTensorCount ?: 0
            for (i in 0 until inputTensorCount) {
                val tensorMetadata = metadataExtractor?.getInputTensorMetadata(i)
                if (tensorMetadata != null) {
                    val name = tensorMetadata.name() ?: "input_$i"
                    val description = tensorMetadata.description() ?: "No description"
                    result[name] = description
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting input tensor info: ${e.message}", e)
        }
        
        return result
    }
    
    /**
     * Extrait les informations des tenseurs de sortie du modèle
     * @return Map des noms de tenseurs de sortie avec leurs descriptions
     */
    fun getOutputTensorInfo(): Map<String, String> {
        val result = mutableMapOf<String, String>()
        
        if (!hasMetadata()) return result
        
        try {
            val outputTensorCount = metadataExtractor?.outputTensorCount ?: 0
            for (i in 0 until outputTensorCount) {
                val tensorMetadata = metadataExtractor?.getOutputTensorMetadata(i)
                if (tensorMetadata != null) {
                    val name = tensorMetadata.name() ?: "output_$i"
                    val description = tensorMetadata.description() ?: "No description"
                    result[name] = description
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting output tensor info: ${e.message}", e)
        }
        
        return result
    }
    
    /**
     * Parse un buffer contenant des labels (un par ligne)
     * @param buffer ByteBuffer contenant les labels
     * @return Liste des labels
     */
    private fun parseLabelsFromBuffer(buffer: ByteBuffer): List<String> {
        val byteArray = ByteArray(buffer.capacity())
        buffer.get(byteArray)
        val content = String(byteArray, StandardCharsets.UTF_8)
        
        return content.lines()
            .filter { it.isNotBlank() }
            .map { it.trim() }
    }
    
    /**
     * Parse un buffer contenant des ancres (valeurs flottantes)
     * @param buffer ByteBuffer contenant les ancres
     * @return Liste des valeurs d'ancres
     */
    private fun parseAnchorsFromBuffer(buffer: ByteBuffer): List<Float> {
        val anchors = mutableListOf<Float>()
        buffer.rewind()
        
        try {
            while (buffer.hasRemaining()) {
                anchors.add(buffer.float)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing anchors: ${e.message}", e)
        }
        
        return anchors
    }
    
    /**
     * Extrait toutes les informations disponibles dans les métadonnées
     * @return Map contenant toutes les informations extraites
     */
    fun extractAllMetadata(): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        
        // Vérifier si le modèle a des métadonnées
        if (!hasMetadata()) {
            result["has_metadata"] = false
            return result
        }
        
        result["has_metadata"] = true
        
        // Description du modèle
        getModelDescription()?.let {
            result["description"] = it
        }
        
        // Labels
        val labels = extractLabels()
        if (labels.isNotEmpty()) {
            result["labels"] = labels
        }
        
        // Anchors
        extractAnchors()?.let {
            result["anchors"] = it
        }
        
        // Informations sur les tenseurs d'entrée
        val inputInfo = getInputTensorInfo()
        if (inputInfo.isNotEmpty()) {
            result["input_tensors"] = inputInfo
        }
        
        // Informations sur les tenseurs de sortie
        val outputInfo = getOutputTensorInfo()
        if (outputInfo.isNotEmpty()) {
            result["output_tensors"] = outputInfo
        }
        
        return result
    }
}
