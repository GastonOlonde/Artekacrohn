package com.example.arteka_crohn.detection.preprocessing

import android.graphics.Bitmap
import android.graphics.Matrix
import com.example.arteka_crohn.detection.config.DetectionConfig
import com.example.arteka_crohn.detection.model.TensorFlowLiteDetectionModel
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Implémentation de l'interface ImagePreprocessor pour les modèles de détection d'objets
 */
class DetectionImagePreprocessor(private val model: TensorFlowLiteDetectionModel) : ImagePreprocessor {

    /**
     * Prétraite une image pour la détection d'objets
     * @param bitmap L'image à prétraiter
     * @return Un tableau de ByteBuffer contenant l'image prétraitée
     */
    override fun preprocess(bitmap: Bitmap): Array<ByteBuffer> {
        // Redimensionner l'image aux dimensions attendues par le modèle
        val scaledBitmap = scaleBitmap(bitmap, model.inputWidth, model.inputHeight)
        
        // Allouer un ByteBuffer pour stocker l'image prétraitée
        val imgData = ByteBuffer.allocateDirect(
            model.inputWidth * model.inputHeight * 3 * 4 // 3 canaux (RGB) * 4 bytes par float
        ).apply {
            order(ByteOrder.nativeOrder())
        }
        imgData.rewind()
        
        // Normaliser et convertir l'image en tableau de floats
        val intValues = IntArray(model.inputWidth * model.inputHeight)
        scaledBitmap.getPixels(intValues, 0, model.inputWidth, 0, 0, model.inputWidth, model.inputHeight)
        
        // Convertir les pixels en valeurs normalisées
        for (i in 0 until model.inputWidth * model.inputHeight) {
            val pixel = intValues[i]
            
            // Extraire les valeurs RGB
            val r = (pixel shr 16 and 0xFF)
            val g = (pixel shr 8 and 0xFF)
            val b = (pixel and 0xFF)
            
            // Normaliser les valeurs (généralement entre 0 et 1 ou -1 et 1)
            val normalizedR = (r - DetectionConfig.INPUT_MEAN) / DetectionConfig.INPUT_STANDARD_DEVIATION
            val normalizedG = (g - DetectionConfig.INPUT_MEAN) / DetectionConfig.INPUT_STANDARD_DEVIATION
            val normalizedB = (b - DetectionConfig.INPUT_MEAN) / DetectionConfig.INPUT_STANDARD_DEVIATION
            
            // Ajouter au ByteBuffer dans l'ordre attendu par le modèle (RGB)
            imgData.putFloat(normalizedR)
            imgData.putFloat(normalizedG)
            imgData.putFloat(normalizedB)
        }
        
        // Libérer le bitmap redimensionné
        if (scaledBitmap != bitmap) {
            scaledBitmap.recycle()
        }
        
        return arrayOf(imgData)
    }
    
    /**
     * Redimensionne un bitmap aux dimensions spécifiées
     */
    private fun scaleBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        if (bitmap.width == targetWidth && bitmap.height == targetHeight) {
            return bitmap
        }
        
        val matrix = Matrix()
        
        if (DetectionConfig.PRESERVE_ASPECT_RATIO) {
            // Calculer le ratio pour préserver l'aspect
            val scaleX = targetWidth.toFloat() / bitmap.width
            val scaleY = targetHeight.toFloat() / bitmap.height
            val scale = minOf(scaleX, scaleY)
            
            // Appliquer la mise à l'échelle uniforme
            matrix.postScale(scale, scale)
            
            // Créer un bitmap intermédiaire avec la mise à l'échelle uniforme
            val scaledBitmap = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
            )
            
            // Créer un bitmap final avec remplissage (padding) pour atteindre les dimensions cibles
            val finalBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(finalBitmap)
            
            // Calculer les positions pour centrer l'image
            val left = (targetWidth - scaledBitmap.width) / 2f
            val top = (targetHeight - scaledBitmap.height) / 2f
            
            // Dessiner l'image mise à l'échelle sur le bitmap final
            canvas.drawBitmap(scaledBitmap, left, top, null)
            
            // Libérer le bitmap intermédiaire
            if (scaledBitmap != bitmap) {
                scaledBitmap.recycle()
            }
            
            return finalBitmap
        } else {
            // Mise à l'échelle directe sans préserver le ratio d'aspect
            return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        }
    }
}
