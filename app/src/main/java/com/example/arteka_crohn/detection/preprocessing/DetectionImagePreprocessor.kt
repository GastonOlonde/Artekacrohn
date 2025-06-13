package com.example.arteka_crohn.detection.preprocessing

import android.graphics.Bitmap
import android.graphics.Matrix
import com.example.arteka_crohn.detection.config.DetectionConfig
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Implémentation de l'interface ImagePreprocessor pour les modèles de détection d'objets
 * Compatible avec tous les types de détecteurs de l'architecture modulaire
 */
class DetectionImagePreprocessor(
    private val inputWidth: Int,
    private val inputHeight: Int,
    private val isQuantized: Boolean = false,
    private val normalizationValues: Triple<Float, Float, Float>? = null
) : ImagePreprocessor {

    /**
     * Prétraite une image pour la détection d'objets
     * @param bitmap L'image à prétraiter
     * @return Un tableau de ByteBuffer contenant l'image prétraitée
     */
    override fun preprocess(bitmap: Bitmap): Array<ByteBuffer> {
        // Redimensionner l'image aux dimensions attendues par le modèle
        val scaledBitmap = scaleBitmap(bitmap, inputWidth, inputHeight)
        
        // Allouer un ByteBuffer pour stocker l'image prétraitée
        val bytesPerChannel = if (isQuantized) 1 else 4 // 1 byte pour uint8, 4 bytes pour float32
        val imgData = ByteBuffer.allocateDirect(
            inputWidth * inputHeight * 3 * bytesPerChannel // 3 canaux (RGB)
        ).apply {
            order(ByteOrder.nativeOrder())
        }
        imgData.rewind()
        
        // Normaliser et convertir l'image en tableau de floats/bytes
        val intValues = IntArray(inputWidth * inputHeight)
        scaledBitmap.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        
        // Déterminer les valeurs de normalisation à utiliser
        val mean = normalizationValues?.first ?: DetectionConfig.INPUT_MEAN
        val std = normalizationValues?.second ?: DetectionConfig.INPUT_STANDARD_DEVIATION
        val scale = normalizationValues?.third ?: 1.0f
        
        // Convertir les pixels en valeurs normalisées
        for (i in 0 until inputWidth * inputHeight) {
            val pixel = intValues[i]
            
            // Extraire les valeurs RGB
            val r = (pixel shr 16 and 0xFF)
            val g = (pixel shr 8 and 0xFF)
            val b = (pixel and 0xFF)
            
            if (isQuantized) {
                // Pour les modèles quantifiés (UINT8)
                imgData.put(r.toByte())
                imgData.put(g.toByte())
                imgData.put(b.toByte())
            } else {
                // Pour les modèles en float32
                // Normaliser les valeurs (généralement entre 0 et 1 ou -1 et 1)
                val normalizedR = (r - mean) / std * scale
                val normalizedG = (g - mean) / std * scale
                val normalizedB = (b - mean) / std * scale
                
                // Ajouter au ByteBuffer dans l'ordre attendu par le modèle (RGB)
                imgData.putFloat(normalizedR)
                imgData.putFloat(normalizedG)
                imgData.putFloat(normalizedB)
            }
        }
        
        // Libérer le bitmap redimensionné
        if (scaledBitmap != bitmap) {
            scaledBitmap.recycle()
        }
        
        // Réinitialiser la position du buffer pour la lecture
        imgData.rewind()
        
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
