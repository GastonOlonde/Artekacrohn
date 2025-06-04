package com.example.arteka_crohn.segmentation

/**
 * Interface pour recevoir les callbacks de segmentation d'instance.
 * Cette interface est utilisée comme un adaptateur pour l'interface
 * InstanceSegmentation.InstanceSegmentationListener.
 */
interface InstanceSegmentationListener {
    /**
     * Appelé lorsqu'une erreur se produit pendant la segmentation
     */
    fun onError(error: String)
    
    /**
     * Appelé lorsque des instances sont détectées
     */
    fun onDetect(
        interfaceTime: Long,
        results: List<ApiSegmentationResult>,
        preProcessTime: Long,
        postProcessTime: Long
    )
    
    /**
     * Appelé lorsqu'aucune instance n'est détectée
     */
    fun onEmpty()
}
