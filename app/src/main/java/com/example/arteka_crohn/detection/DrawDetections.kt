package com.example.arteka_crohn.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import com.example.arteka_crohn.Output0
import com.example.arteka_crohn.R
import com.example.arteka_crohn.detection.config.DetectionConfig

/**
 * Classe responsable du dessin des boîtes englobantes et des étiquettes pour les objets détectés
 */
class DrawDetections(private val context: Context) {

    // Couleurs pour différentes classes
    private val boxColors = listOf(
        R.color.overlay_red,
        R.color.overlay_green,
        R.color.overlay_blue,
        R.color.overlay_yellow,
        R.color.overlay_purple,
        R.color.overlay_orange
    )
    
    // Peintures pour le dessin
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = DetectionConfig.BOX_THICKNESS_DRAW.toFloat()
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
        textSize = DetectionConfig.TEXT_SIZE_DRAW
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        color = Color.WHITE
    }
    
    private val textBackgroundPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    /**
     * Dessine les boîtes englobantes et les étiquettes pour les objets détectés
     * @param detections La liste des objets détectés
     * @param screenWidth Largeur de l'écran pour l'affichage
     * @param screenHeight Hauteur de l'écran pour l'affichage
     * @param scaleFactor Facteur d'échelle pour contrôler la qualité
     * @return L'image avec les boîtes englobantes et les étiquettes
     */
    fun invoke(
        detections: List<Output0>,
        screenWidth: Int = 0,
        screenHeight: Int = 0,
        scaleFactor: Float = DetectionConfig.DEFAULT_SCALE_FACTOR_DRAW
    ): Bitmap {
        if (detections.isEmpty()) {
            Log.w("DrawDetections", "No detections to draw.")
            return createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }
        
        // Déterminer les dimensions de sortie
        val outputWidth = if (screenWidth > 0) screenWidth else 640
        val outputHeight = if (screenHeight > 0) screenHeight else 640
        
        // Créer un bitmap vide pour le résultat
        val outputBitmap = createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)
        
        // Dessiner chaque détection
        for (detection in detections) {
            // Sélectionner une couleur basée sur la classe
            val colorIndex = detection.cls % boxColors.size
            val color = ContextCompat.getColor(context, boxColors[colorIndex])
            
            // Configurer les peintures avec la couleur sélectionnée
            boxPaint.color = color
            textBackgroundPaint.color = color
            
            // Convertir les coordonnées normalisées en coordonnées de pixels
            val left = (detection.x1 * outputWidth).toInt()
            val top = (detection.y1 * outputHeight).toInt()
            val right = (detection.x2 * outputWidth).toInt()
            val bottom = (detection.y2 * outputHeight).toInt()
            
            // Vérifier que les coordonnées sont valides
            if (left < 0 || top < 0 || right > outputWidth || bottom > outputHeight) {
                Log.w("DrawDetections", "Coordonnées hors limites: ($left, $top) - ($right, $bottom)")
                continue
            }
            
            // Dessiner la boîte englobante
            canvas.drawRect(left.toFloat(), top.toFloat(), right.toFloat(), bottom.toFloat(), boxPaint)
            
            // Préparer le texte à afficher
            val displayText = "${detection.clsName} ${(detection.cnf * 100).toInt()}%"
            val textBounds = Rect()
            textPaint.getTextBounds(displayText, 0, displayText.length, textBounds)
            
            val textWidth = textBounds.width()
            val textHeight = textBounds.height()
            val textPadding = DetectionConfig.TEXT_PADDING_DRAW.toInt()
            
            // Calculer la position du texte
            var textX = left.toFloat()
            var textYBase = top - textPadding.toFloat()
            
            // Ajuster si le texte sort en haut
            val bgHeight = textHeight + textPadding * 2
            if (textYBase - bgHeight < 0) {
                textYBase = top + textHeight + textPadding * 2.toFloat()
            }
            
            // Dessiner le fond du texte
            val textBackgroundRect = Rect(
                textX.toInt(),
                (textYBase - textHeight - textPadding).toInt(),
                (textX + textWidth + textPadding * 2).toInt(),
                textYBase.toInt()
            )
            canvas.drawRect(textBackgroundRect, textBackgroundPaint)
            
            // Dessiner le texte
            canvas.drawText(
                displayText,
                textX + textPadding,
                textYBase - textPadding,
                textPaint
            )
        }
        
        return outputBitmap
    }
    
    /**
     * Version simplifiée pour dessiner directement sur un bitmap existant
     * @param bitmap L'image sur laquelle dessiner
     * @param detections La liste des objets détectés
     * @return L'image avec les boîtes englobantes et les étiquettes
     */
    fun drawDetectionsOnBitmap(bitmap: Bitmap, detections: List<Output0>): Bitmap {
        return invoke(detections, bitmap.width, bitmap.height)
    }
    
    /**
     * Version avec mise à l'échelle pour dessiner sur un bitmap avec des dimensions cibles
     * @param bitmap L'image sur laquelle dessiner (non utilisée directement)
     * @param detections La liste des objets détectés
     * @param targetWidth Largeur cible pour l'affichage
     * @param targetHeight Hauteur cible pour l'affichage
     * @return L'image avec les boîtes englobantes et les étiquettes, mise à l'échelle
     */
    fun drawDetectionsOnBitmapScaled(
        bitmap: Bitmap,
        detections: List<Output0>,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {
        return invoke(detections, targetWidth, targetHeight)
    }
}