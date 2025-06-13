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
        
        // Calculer les facteurs d'échelle en supposant une image carrée pour YOLOv8
        // La plupart des modèles de détection utilisent des dimensions carrées (ex: 640x640)
        val inputWidth = 640f  // Dimensions typiques des modèles YOLO (carré)
        val inputHeight = 640f // Dimensions typiques des modèles YOLO (carré)
        
        val scaleX = outputWidth.toFloat() / inputWidth
        val scaleY = outputHeight.toFloat() / inputHeight
        
        // Utiliser une mise à l'échelle uniforme pour préserver le ratio d'aspect
        val uniformScale = minOf(scaleX, scaleY)
        
        // Calculer les décalages pour centrer l'image
        val offsetX = (outputWidth - inputWidth * uniformScale) / 2f
        val offsetY = (outputHeight - inputHeight * uniformScale) / 2f
        
        // Dessiner chaque détection
        for (detection in detections) {
            // Sélectionner une couleur basée sur la classe
            val colorIndex = detection.cls % boxColors.size
            val color = ContextCompat.getColor(context, boxColors[colorIndex])
            
            // Configurer les peintures avec la couleur sélectionnée
            boxPaint.color = color
            textBackgroundPaint.color = color
            
            // CORRECTION POUR BBOX DÉCALÉES:
            // 1. Calculer la mise à l'échelle originale utilisée lors du prétraitement
            val modelInputSize = 640f // Taille d'entrée du modèle (typiquement 640×640 pour YOLOv8)
            val originalWidth = 480f  // Largeur originale de l'image (mode portrait)
            val originalHeight = 640f // Hauteur originale de l'image (mode portrait)
            
            val scaleFactorPreprocessingX = modelInputSize / originalWidth
            val scaleFactorPreprocessingY = modelInputSize / originalHeight
            val scalePreprocessing = minOf(scaleFactorPreprocessingX, scaleFactorPreprocessingY)
            
            // 2. Calculer les offsets utilisés lors du prétraitement
            val offsetXPreprocessing = (modelInputSize - (originalWidth * scalePreprocessing)) / 2f / modelInputSize
            val offsetYPreprocessing = (modelInputSize - (originalHeight * scalePreprocessing)) / 2f / modelInputSize
            
            // 3. Dénormaliser les coordonnées en tenant compte du prétraitement
            val x1Corrected = (detection.x1 - offsetXPreprocessing) / (1f - 2f * offsetXPreprocessing)
            val x2Corrected = (detection.x2 - offsetXPreprocessing) / (1f - 2f * offsetXPreprocessing)
            val y1Corrected = detection.y1 // Pas de correction verticale car le ratio 4:3 ne crée pas d'offset vertical
            val y2Corrected = detection.y2
            
            // 4. Convertir les coordonnées normalisées corrigées en pixels
            val left = offsetX + (x1Corrected * inputWidth * uniformScale)
            val top = offsetY + (y1Corrected * inputHeight * uniformScale)
            val right = offsetX + (x2Corrected * inputWidth * uniformScale)
            val bottom = offsetY + (y2Corrected * inputHeight * uniformScale)
            
            // Vérifier que les coordonnées sont valides
            if (left < 0 || top < 0 || right > outputWidth || bottom > outputHeight) {
                Log.w("DrawDetections", "Coordonnées hors limites: ($left, $top) - ($right, $bottom)")
                continue
            }
            
            // Dessiner la boîte englobante
            canvas.drawRect(left, top, right, bottom, boxPaint)
            
            // Préparer le texte à afficher
            val displayText = "${detection.clsName} ${(detection.cnf * 100).toInt()}%"
            val textBounds = Rect()
            textPaint.getTextBounds(displayText, 0, displayText.length, textBounds)
            
            val textWidth = textBounds.width()
            val textHeight = textBounds.height()
            val textPadding = DetectionConfig.TEXT_PADDING_DRAW.toInt()
            
            // Calculer la position du texte
            var textX = left
            var textYBase = top - textPadding
            
            // Ajuster si le texte sort en haut
            val bgHeight = textHeight + textPadding * 2
            if (textYBase - bgHeight < 0) {
                textYBase = top + textHeight + textPadding * 2
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
        if (detections.isEmpty()) {
            Log.w("DrawDetections", "No detections to draw.")
            return createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        }
        
        // Créer un bitmap vide pour le résultat
        val outputBitmap = createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)
        
        // Calculer les facteurs d'échelle
        val scaleX = targetWidth.toFloat() / bitmap.width
        val scaleY = targetHeight.toFloat() / bitmap.height
        
        // Utiliser une mise à l'échelle uniforme pour préserver le ratio d'aspect
        val uniformScale = minOf(scaleX, scaleY)
        
        // Calculer les décalages pour centrer l'image
        val offsetX = (targetWidth - bitmap.width * uniformScale) / 2f
        val offsetY = (targetHeight - bitmap.height * uniformScale) / 2f
        
        // Dessiner chaque détection
        for (detection in detections) {
            // Sélectionner une couleur basée sur la classe
            val colorIndex = detection.cls % boxColors.size
            val color = ContextCompat.getColor(context, boxColors[colorIndex])
            
            // Configurer les peintures avec la couleur sélectionnée
            boxPaint.color = color
            textBackgroundPaint.color = color
            
            // CORRECTION POUR BBOX DÉCALÉES:
            // 1. Calculer la mise à l'échelle originale utilisée lors du prétraitement
            val modelInputSize = 640f // Taille d'entrée du modèle (typiquement 640×640 pour YOLOv8)
            val originalWidth = 480f  // Largeur originale de l'image (mode portrait)
            val originalHeight = 640f // Hauteur originale de l'image (mode portrait)
            
            val scaleFactorPreprocessingX = modelInputSize / originalWidth
            val scaleFactorPreprocessingY = modelInputSize / originalHeight
            val scalePreprocessing = minOf(scaleFactorPreprocessingX, scaleFactorPreprocessingY)
            
            // 2. Calculer les offsets utilisés lors du prétraitement
            val offsetXPreprocessing = (modelInputSize - (originalWidth * scalePreprocessing)) / 2f / modelInputSize
            val offsetYPreprocessing = (modelInputSize - (originalHeight * scalePreprocessing)) / 2f / modelInputSize
            
            // 3. Dénormaliser les coordonnées en tenant compte du prétraitement
            val x1Corrected = (detection.x1 - offsetXPreprocessing) / (1f - 2f * offsetXPreprocessing)
            val x2Corrected = (detection.x2 - offsetXPreprocessing) / (1f - 2f * offsetXPreprocessing)
            val y1Corrected = detection.y1 // Pas de correction verticale car le ratio 4:3 ne crée pas d'offset vertical
            val y2Corrected = detection.y2
            
            // 4. Convertir les coordonnées normalisées corrigées en pixels
            val left = offsetX + (x1Corrected * bitmap.width * uniformScale)
            val top = offsetY + (y1Corrected * bitmap.height * uniformScale)
            val right = offsetX + (x2Corrected * bitmap.width * uniformScale)
            val bottom = offsetY + (y2Corrected * bitmap.height * uniformScale)
            
            // Vérifier que les coordonnées sont valides
            if (left < 0 || top < 0 || right > targetWidth || bottom > targetHeight) {
                Log.w("DrawDetections", "Coordonnées hors limites: ($left, $top) - ($right, $bottom)")
                continue
            }
            
            // Dessiner la boîte englobante
            canvas.drawRect(left, top, right, bottom, boxPaint)
            
            // Préparer le texte à afficher
            val displayText = "${detection.clsName} ${(detection.cnf * 100).toInt()}%"
            val textBounds = Rect()
            textPaint.getTextBounds(displayText, 0, displayText.length, textBounds)
            
            val textWidth = textBounds.width()
            val textHeight = textBounds.height()
            val textPadding = DetectionConfig.TEXT_PADDING_DRAW.toInt()
            
            // Calculer la position du texte
            var textX = left
            var textYBase = top - textPadding
            
            // Ajuster si le texte sort en haut
            val bgHeight = textHeight + textPadding * 2
            if (textYBase - bgHeight < 0) {
                textYBase = top + textHeight + textPadding * 2
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
}