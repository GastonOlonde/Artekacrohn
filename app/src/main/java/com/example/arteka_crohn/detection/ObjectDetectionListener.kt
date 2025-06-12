package com.example.arteka_crohn.detection

import com.example.arteka_crohn.Output0

/**
 * Interface pour recevoir les callbacks de détection d'objets.
 * Version simplifiée par rapport à l'interface InstanceSegmentationListener
 */
interface ObjectDetectionListener {
    /**
     * Appelé lorsqu'une erreur se produit pendant la détection
     */
    fun onError(error: String)
    
    /**
     * Appelé lorsque des objets sont détectés
     */
    fun onDetect(
        inferenceTime: Long,
        results: List<Output0>,
        preProcessTime: Long,
        postProcessTime: Long
    )
    
    /**
     * Appelé lorsqu'aucun objet n'est détecté
     */
    fun onEmpty()
}
