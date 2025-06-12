package com.example.arteka_crohn.segmentation.config

import org.tensorflow.lite.DataType

/**
 * Classe de configuration centralisée pour les paramètres de segmentation
 * Facilite la maintenance et la modification des paramètres
 */
object SegmentationConfig {
    // Paramètres du modèle TensorFlow Lite
    const val INPUT_MEAN = 0f
    const val INPUT_STANDARD_DEVIATION = 255f
    val INPUT_IMAGE_TYPE = DataType.FLOAT32
    val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
    
    // Seuils pour la détection
    const val CONFIDENCE_THRESHOLD = 0.45F
    const val IOU_THRESHOLD = 0.5F
    
    // Dimensions des masques finaux
    const val FINAL_MASK_WIDTH = 1024
    const val FINAL_MASK_HEIGHT = 1024
    
    // Paramètres d'affichage
    const val CONTOUR_THICKNESS = 2
    const val DEFAULT_CONTOUR_THICKNESS_PIXELS = 1
    const val BINARIZATION_THRESHOLD = 0.2f
    const val MIN_CONTOUR_PIXEL_SIZE = 1
    
    // Paramètres de mise à l'échelle
    const val PRESERVE_ASPECT_RATIO = true
    const val DEFAULT_SCALE_FACTOR = 0.5f

    const val RENDER_EVERY_N_FRAMES = 2

}
