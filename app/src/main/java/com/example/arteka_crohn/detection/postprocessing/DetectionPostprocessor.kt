package com.example.arteka_crohn.detection.postprocessing

import com.example.arteka_crohn.Output0

/**
 * Interface pour le post-traitement des résultats de détection
 * Suit le principe d'ouverture/fermeture (O) dans SOLID
 */
interface DetectionPostprocessor {
    /**
     * Traite les résultats bruts de détection pour produire une liste d'objets détectés
     * @param rawOutputs Les sorties brutes du modèle de détection
     * @return Une liste d'objets détectés
     */
    fun processDetections(rawOutputs: FloatArray): List<Output0>
}
