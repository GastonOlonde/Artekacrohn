package com.example.arteka_crohn.detection.model

/**
 * Énumération des types de modèles de détection d'objets supportés
 * avec méthodes pour la détection automatique
 */
enum class ModelType {
    YOLO_V8,        // YOLOv8 models (n, s, m, l, x)
    MOBILENET_SSD,  // MobileNet SSD models (single or multi-output format)
    UNKNOWN;        // Type inconnu ou non supporté
    
    companion object {
        /**
         * Tente de détecter le type de modèle à partir du nom de fichier
         * @param filename Nom du fichier du modèle
         * @return Type de modèle détecté ou UNKNOWN si non reconnu
         */
        fun detectFromFilename(filename: String): ModelType {
            val lowerFilename = filename.lowercase()
            return when {
                lowerFilename.contains("yolo") && (
                    lowerFilename.contains("v8") || 
                    lowerFilename.contains("-n") || 
                    lowerFilename.contains("-s") || 
                    lowerFilename.contains("-m") || 
                    lowerFilename.contains("-l") || 
                    lowerFilename.contains("-x")
                ) -> YOLO_V8
                lowerFilename.contains("ssd") || 
                lowerFilename.contains("mobilenet") || 
                lowerFilename.contains("coco") -> MOBILENET_SSD
                else -> UNKNOWN
            }
        }
        
        /**
         * Détecte le type de modèle en analysant les dimensions des tenseurs d'entrée/sortie
         * @param inputShape Shape du tenseur d'entrée
         * @param outputShape Shape du tenseur de sortie
         * @return Type de modèle détecté ou UNKNOWN si non reconnu
         */
        fun detectFromShapes(inputShape: IntArray, outputShapes: List<IntArray>): ModelType {
            // YOLOv8 a généralement une seule sortie avec shape [1, 84, 8400] (pour COCO)
            // ou [1, n+5, 8400] où n est le nombre de classes
            if (outputShapes.size == 1) {
                val outputShape = outputShapes[0]
                if (outputShape.size == 3 && outputShape[0] == 1 && outputShape[2] == 8400) {
                    return YOLO_V8
                }
            }
            
            // MobileNet SSD a généralement 4 sorties (multi-output) ou 1 sortie (single-output)
            // Format multi-output: locations [1,N,4], categories [1,N], scores [1,N], num_detections [1]
            // Format single-output: [1,N,4+2] ou [1,N,4+1+numClasses]
            if (outputShapes.size == 4) {
                return MOBILENET_SSD
            } else if (outputShapes.size == 1) {
                val outputShape = outputShapes[0]
                if (outputShape.size == 3 && outputShape[0] == 1 && outputShape[2] >= 6) {
                    return MOBILENET_SSD
                }
            }
            
            return UNKNOWN
        }
        
        /**
         * Méthode combinée pour déterminer le type de modèle en utilisant
         * toutes les informations disponibles
         */
        fun detectModelType(
            filename: String,
            inputShape: IntArray,
            outputShapes: List<IntArray>
        ): ModelType {
            // D'abord essayer par le nom de fichier
            var detectedType = detectFromFilename(filename)
            
            // Si non concluant, analyser les shapes
            if (detectedType == UNKNOWN) {
                detectedType = detectFromShapes(inputShape, outputShapes)
            }
            
            return detectedType
        }
    }
}
