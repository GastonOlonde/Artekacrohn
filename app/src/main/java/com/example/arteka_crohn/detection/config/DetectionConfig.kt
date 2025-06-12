package com.example.arteka_crohn.detection.config

import org.tensorflow.lite.DataType

/**
 * Classe de configuration centralisée pour les paramètres de détection d'objets
 * Facilite la maintenance et la modification des paramètres
 */
object DetectionConfig {
    // Paramètres du modèle TensorFlow Lite
    const val INPUT_MEAN = 0f
    const val INPUT_STANDARD_DEVIATION = 255f
    val INPUT_IMAGE_TYPE = DataType.FLOAT32
    val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
    
    // Seuils pour la détection
    /**
     * Seuil de confiance pour les détections
     * Toute détection avec une confiance inférieure à ce seuil sera ignorée
     */
    const val CONFIDENCE_THRESHOLD = 0.5f  // Valeur très basse pour le débogage
    const val IOU_THRESHOLD = 0.5F
    
    // Dimensions d'entrée standard pour les modèles de détection
    // À ajuster selon les modèles utilisés (par exemple 300x300 pour SSD, 416x416 pour YOLO)
    const val DEFAULT_INPUT_WIDTH = 320
    const val DEFAULT_INPUT_HEIGHT = 320
    
    // Paramètres d'affichage
    const val BOX_THICKNESS = 2
    const val TEXT_SIZE = 14f
    const val TEXT_PADDING = 6
    
    // Paramètres de mise à l'échelle
    const val PRESERVE_ASPECT_RATIO = true
    const val DEFAULT_SCALE_FACTOR = 0.5f

    const val RENDER_EVERY_N_FRAMES = 2

    // Paramètres du modèle
    const val INPUT_SIZE = 640 // Taille d'entrée du modèle (640x640 pour YOLOv8-640, 416x416 pour YOLO standard)
    const val BATCH_SIZE = 1 // Taille du lot pour l'inférence
    const val PIXEL_SIZE = 3 // Nombre de canaux de couleur (RGB)
    
    // Seuils de confiance
    const val CONFIDENCE_THRESHOLD_DRAW = 0.5f // Seuil de confiance pour les détections
    const val IOU_THRESHOLD_DRAW = 0.5f // Seuil IoU pour la suppression des non-maximums (NMS)
    
    // Paramètres d'affichage pour les détections
    const val PRESERVE_ASPECT_RATIO_DRAW = true
    const val DEFAULT_SCALE_FACTOR_DRAW = 0.5f
    const val BOX_THICKNESS_DRAW = 2
    const val TEXT_SIZE_DRAW = 40f
    const val TEXT_PADDING_DRAW = 10f
    
    // Paramètres de mise à l'échelle
    const val PRESERVE_ASPECT_RATIO_DRAW_UNIFORM = true // Préserver le ratio d'aspect lors de la mise à l'échelle uniforme
    const val DEFAULT_SCALE_FACTOR_DRAW_UNIFORM = 1.0f // Facteur d'échelle par défaut pour la mise à l'échelle uniforme
    
    // Paramètres de prétraitement
    const val NORMALIZE_IMAGE_DRAW = true // Normaliser les valeurs des pixels (0-1 au lieu de 0-255)
    val MEAN_RGB_DRAW = floatArrayOf(127.5f, 127.5f, 127.5f) // Valeurs moyennes RGB pour la normalisation
    val STD_RGB_DRAW = floatArrayOf(127.5f, 127.5f, 127.5f) // Écarts-types RGB pour la normalisation
    
    // Paramètres d'inférence
    const val NUM_THREADS_DRAW = 4 // Nombre de threads pour l'inférence
    const val USE_GPU_DRAW = true // Utiliser le GPU si disponible
    const val USE_NNAPI_DRAW = false // Utiliser NNAPI si disponible
    
    // Paramètres de post-traitement
    const val MAX_DETECTIONS_DRAW = 10 // Nombre maximum de détections à afficher
    
    // Paramètres spécifiques aux modèles YOLO
    object YOLO {
        // Paramètres généraux YOLO
        const val DEFAULT_INPUT_SIZE = 416 // Taille d'entrée standard pour YOLO
        const val DEFAULT_GRID_SIZE = 13 // Taille de la grille pour YOLOv3/v4 (416/32)
        
        // Paramètres YOLOv3
        object V3 {
            // Dimensions des ancres pour YOLOv3 (COCO)
            val ANCHORS = floatArrayOf(
                10f, 13f, 16f, 30f, 33f, 23f,  // Petites ancres
                30f, 61f, 62f, 45f, 59f, 119f, // Moyennes ancres
                116f, 90f, 156f, 198f, 373f, 326f // Grandes ancres
            )
            const val NUM_ANCHORS = 9 // 3 ancres par échelle, 3 échelles
            const val NUM_CLASSES = 80 // Nombre de classes COCO par défaut
        }
        
        // Paramètres YOLOv4
        object V4 {
            // Dimensions des ancres pour YOLOv4 (COCO)
            val ANCHORS = floatArrayOf(
                12f, 16f, 19f, 36f, 40f, 28f,
                36f, 75f, 76f, 55f, 72f, 146f,
                142f, 110f, 192f, 243f, 459f, 401f
            )
            const val NUM_ANCHORS = 9 // 3 ancres par échelle, 3 échelles
            const val NUM_CLASSES = 80 // Nombre de classes COCO par défaut
        }
        
        // Paramètres YOLOv5
        object V5 {
            // Dimensions des ancres pour YOLOv5 (COCO)
            val ANCHORS = floatArrayOf(
                10f, 13f, 16f, 30f, 33f, 23f,
                30f, 61f, 62f, 45f, 59f, 119f,
                116f, 90f, 156f, 198f, 373f, 326f
            )
            const val NUM_ANCHORS = 9 // 3 ancres par échelle, 3 échelles
            const val NUM_CLASSES = 80 // Nombre de classes COCO par défaut
        }
        
        // Paramètres YOLOv8
        object V8 {
            // YOLOv8 utilise un format de sortie différent sans ancres prédéfinies
            const val NUM_CLASSES = 80 // Nombre de classes COCO par défaut
        }
    }
}
