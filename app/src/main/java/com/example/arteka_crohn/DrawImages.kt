package com.example.arteka_crohn

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.core.graphics.set

// IMPORTANT : Assure-toi que les types et noms de champs utilisés ci-dessous
// (comme segmentationResult.mask, segmentationResult.box.x1, segmentationResult.box.clsName, etc.)
// correspondent EXACTEMENT à la structure de TA classe SegmentationResult existante.

// Si ta classe SegmentationResult n'a pas de champ 'box' mais des champs individuels pour
// les coordonnées, le nom de la classe, etc., tu devras adapter les accès.
// Par exemple, si tu as:
// data class SegmentationResult(
//     val mask: Array<FloatArray>,
//     val x1_coord: Float, // exemple de nom de champ différent
//     val y1_coord: Float,
//     val categoryName: String, // exemple de nom de champ différent
//     val score: Float,         // exemple de nom de champ différent
//     val categoryId: Int
// )
// alors tu devras remplacer:
// segmentationResult.box.x1       -> segmentationResult.x1_coord
// segmentationResult.box.y1       -> segmentationResult.y1_coord
// segmentationResult.box.clsName  -> segmentationResult.categoryName
// segmentationResult.box.conf     -> segmentationResult.score
// segmentationResult.box.cls      -> segmentationResult.categoryId


class DrawImages(private val context: Context) {

    // Couleurs plus vives pour une meilleure visibilité
    private val boxColors = listOf(
        R.color.overlay_red,
        R.color.overlay_green,
        R.color.overlay_blue,
        R.color.overlay_yellow,
        R.color.overlay_purple,
        R.color.overlay_orange
    )

    companion object {
        private const val DEFAULT_CONTOUR_THICKNESS_PIXELS = 2 // Augmentation de l'épaisseur par défaut
        private const val BINARIZATION_THRESHOLD = 0.5f
        private const val MIN_CONTOUR_PIXEL_SIZE = 2 // Taille minimale des pixels du contour
    }

    fun invoke(
        results: List<SegmentationResult>,
        contourThickness: Int = DEFAULT_CONTOUR_THICKNESS_PIXELS,
        screenWidth: Int = 0,
        screenHeight: Int = 0,
        scaleFactor: Float = 0.5f // Nouveau paramètre pour contrôler la qualité
    ): Bitmap {
        if (results.isEmpty()) {
            Log.w("DrawImages", "No results to draw.")
            return createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }

        val firstValidResult = results.firstOrNull { res ->
            res.mask.isNotEmpty() && res.mask[0].isNotEmpty()
        } ?: return createBitmap(1, 1, Bitmap.Config.ARGB_8888)

        val firstMask = firstValidResult.mask
        val maskHeight = firstMask.size
        val maskWidth = firstMask[0].size
        
        // Calcul des dimensions avec facteur d'échelle
        val scaledWidth = (if (screenWidth > 0) screenWidth else maskWidth) * scaleFactor
        val scaledHeight = (if (screenHeight > 0) screenHeight else maskHeight) * scaleFactor
        
        val outputBitmap = createBitmap(scaledWidth.toInt(), scaledHeight.toInt(), Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)

        results.forEach { result ->
            val currentMask = result.mask
            if (currentMask.isEmpty() || currentMask[0].isEmpty() ||
                currentMask.size != maskHeight || currentMask[0].size != maskWidth
            ) {
                Log.e("DrawImages", "Skipping result due to inconsistent/empty mask dimensions.")
                return@forEach
            }

            val colorIndex = result.box.cls % boxColors.size
            val colorResId = boxColors[colorIndex]
            
            // Utilisation d'un bitmap intermédiaire plus petit
            val tempBitmap = createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888)
            val tempCanvas = Canvas(tempBitmap)
            
            drawMorphologicalContourAndText(
                tempBitmap,
                tempCanvas,
                result,
                colorResId,
                contourThickness,
                maskWidth,
                maskHeight,
                maskWidth,
                maskHeight
            )
            
            // Mise à l'échelle du bitmap temporaire
            val scaledBitmap = Bitmap.createScaledBitmap(
                tempBitmap,
                scaledWidth.toInt(),
                scaledHeight.toInt(),
                true
            )
            
            // Dessin du bitmap mis à l'échelle
            canvas.drawBitmap(scaledBitmap, 0f, 0f, null)
            
            // Nettoyage
            tempBitmap.recycle()
            scaledBitmap.recycle()
        }

        return outputBitmap
    }

    private fun binarizeMask(inputMask: Array<FloatArray>, width: Int, height: Int, threshold: Float): Array<IntArray> {
        return Array(height) { y ->
            IntArray(width) { x ->
                if (inputMask[y][x] > threshold) 1 else 0
            }
        }
    }

    private fun dilateBinaryMask(inputBinaryMask: Array<IntArray>, width: Int, height: Int): Array<IntArray> {
        val dilatedMask = Array(height) { IntArray(width) { 0 } }
        for (y in 0 until height) {
            for (x in 0 until width) {
                var activate = false
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        val ny = y + dy
                        val nx = x + dx
                        if (ny in 0 until height && nx in 0 until width && inputBinaryMask[ny][nx] > 0) {
                            activate = true
                            break
                        }
                    }
                    if (activate) break
                }
                if (activate) {
                    dilatedMask[y][x] = 1
                }
            }
        }
        return dilatedMask
    }

    private fun dilateBinaryMaskByRadius(inputBinaryMask: Array<IntArray>, dilationRadius: Int, width: Int, height: Int): Array<IntArray> {
        if (dilationRadius < 0) {
            Log.w("DrawImages", "dilateBinaryMaskByRadius called with negative radius, returning original.")
            return inputBinaryMask.map { it.clone() }.toTypedArray()
        }
        if (dilationRadius == 0) {
            return inputBinaryMask.map { it.clone() }.toTypedArray()
        }

        val finalDilatedMask = Array(height) { IntArray(width) { 0 } }
        for (r in 0 until height) {
            for (c in 0 until width) {
                if (inputBinaryMask[r][c] > 0) {
                    for (dr in -dilationRadius..dilationRadius) {
                        for (dc in -dilationRadius..dilationRadius) {
                            val nr = r + dr
                            val nc = c + dc
                            if (nr in 0 until height && nc in 0 until width) {
                                finalDilatedMask[nr][nc] = 1
                            }
                        }
                    }
                }
            }
        }
        return finalDilatedMask
    }

    private fun drawMorphologicalContourAndText(
        bitmap: Bitmap,
        canvas: Canvas,
        segmentationResult: SegmentationResult, // Utilise TA classe SegmentationResult
        contourColorResId: Int,
        targetContourThickness: Int,
        maskWidth: Int,    // Dimensions du masque d'origine
        maskHeight: Int,   // Dimensions du masque d'origine
        displayWidth: Int, // Dimensions finales d'affichage
        displayHeight: Int // Dimensions finales d'affichage
    ) {
        // Adapte l'accès à 'segmentationResult.mask' si le nom du champ est différent
        val probabilityMask = segmentationResult.mask
        val contourColor = ContextCompat.getColor(context, contourColorResId)

        val binaryOriginalMask = binarizeMask(probabilityMask, maskWidth, maskHeight, BINARIZATION_THRESHOLD)
        val dilated1pxMask = dilateBinaryMask(binaryOriginalMask, maskWidth, maskHeight)

        // --- Appliquer un flou gaussien pour adoucir le contour ---
        val thinOuterContour = Array(maskHeight) { y ->
            IntArray(maskWidth) { x ->
                if (dilated1pxMask[y][x] > 0 && binaryOriginalMask[y][x] == 0) 1 else 0
            }
        }

        val dilationRadiusForThickness = if (targetContourThickness > 0) (targetContourThickness - 1) / 2 else 0

        // --- Appliquer la dilatation pour obtenir l'épaisseur de contour souhaitée ---
        val finalContourMask = if (targetContourThickness <= 1 && targetContourThickness > 0) {
            thinOuterContour
        } else if (targetContourThickness > 1) {
            dilateBinaryMaskByRadius(thinOuterContour, dilationRadiusForThickness, maskWidth, maskHeight)
        } else {
            Array(maskHeight) { IntArray(maskWidth) { 0 } }
        }

        // --- Dessiner le contour sur le bitmap avec une meilleure visibilité ---
        // Calcul des facteurs d'échelle pour adapter le masque à la taille d'affichage
        val scaleX = displayWidth.toFloat() / maskWidth.toFloat()
        val scaleY = displayHeight.toFloat() / maskHeight.toFloat()
        
        // Calculer une taille de contour adaptée à l'écran
        // Plus l'écran est grand, plus le contour doit être épais pour rester visible
        val contourPixelSize = (MIN_CONTOUR_PIXEL_SIZE * scaleX).toInt().coerceAtLeast(MIN_CONTOUR_PIXEL_SIZE)
        
        // Rendre le contour plus visible en dessinant plusieurs pixels
        for (y in 0 until maskHeight) {
            for (x in 0 until maskWidth) {
                if (finalContourMask[y][x] > 0) {
                    // Calculer les coordonnées dans l'espace d'affichage
                    val displayX = (x * scaleX).toInt()
                    val displayY = (y * scaleY).toInt()
                    
                    // Dessiner un "gros pixel" pour chaque point du contour pour une meilleure visibilité
                    for (dy in -contourPixelSize until contourPixelSize) {
                        for (dx in -contourPixelSize until contourPixelSize) {
                            val nx = displayX + dx
                            val ny = displayY + dy
                            // S'assurer que les coordonnées sont dans les limites
                            if (nx in 0 until displayWidth && ny in 0 until displayHeight) {
                                bitmap[nx, ny] = contourColor
                            }
                        }
                    }
                }
            }
        }

        // --- Dessiner le texte de la classe ---
        // Adapte les accès aux champs 'box.clsName', 'box.conf', 'box.x1', 'box.y1'
        // selon la structure de TA classe SegmentationResult.
        // Par exemple, si tu as segmentationResult.className, segmentationResult.confidenceScore, etc.
        val className = segmentationResult.box.clsName // Exemple: peut être segmentationResult.className
        val confidence = segmentationResult.conf   // Exemple: peut être segmentationResult.confidenceScore
        val textToShow = "${className} ${String.format("%.2f", confidence)}"

        val textPaint = Paint().apply {
            this.color = contourColor
            this.style = Paint.Style.FILL
            this.textSize = 15f
            this.isAntiAlias = true
            this.typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }

        val textBounds = android.graphics.Rect()
        textPaint.getTextBounds(textToShow, 0, textToShow.length, textBounds)

        val textHeight = textBounds.height()
        val textWidth = textBounds.width()
        val textPadding = 8

        // Adapte les accès à 'box.x1' et 'box.y1'
        val boxX1 = segmentationResult.box.x1 // Exemple: peut être segmentationResult.normX1
        val boxY1 = segmentationResult.box.y1 // Exemple: peut être segmentationResult.normY1

        // Ajuster les coordonnées en fonction de la taille d'affichage
        var textX = (boxX1 * displayWidth)
        var textYBase = (boxY1 * displayHeight) - textPadding // Position au-dessus de la box (baseline)

        // Ajuster si le texte sort en haut
        val bgHeight = textHeight + textPadding // Hauteur approximative du texte
        if (textYBase - bgHeight < 0) { // Si le texte sort du haut
            textYBase = (boxY1 * displayHeight) + textPadding + textHeight + textPadding // Position en dessous
        }
        if (textX < 0) textX = 0f

        // Affichage du texte sans background
        canvas.drawText(textToShow, textX + textPadding, textYBase, textPaint)

    }
}