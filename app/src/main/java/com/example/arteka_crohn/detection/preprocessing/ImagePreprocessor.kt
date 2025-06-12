package com.example.arteka_crohn.detection.preprocessing

import android.graphics.Bitmap
import java.nio.ByteBuffer

/**
 * Interface pour le prétraitement d'images avant l'inférence
 * Suit le principe d'ouverture/fermeture (O) dans SOLID
 */
interface ImagePreprocessor {
    /**
     * Prétraite une image pour la détection d'objets
     * @param bitmap L'image à prétraiter
     * @return Un tableau de ByteBuffer contenant l'image prétraitée
     */
    fun preprocess(bitmap: Bitmap): Array<ByteBuffer>
}
