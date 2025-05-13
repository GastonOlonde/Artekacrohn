package com.example.arteka_crohn

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.core.graphics.set

class DrawImages(private val context: Context) {

    private val boxColors = listOf(
        R.color.overlay_red,
        R.color.overlay_green,
    )

    fun invoke(results: List<SegmentationResult>) : Bitmap {
        val width = results.first().mask[0].size
        val height = results.first().mask.size
        val combined = createBitmap(width, height)

        results.forEach { result ->
            val colorResId = boxColors[result.box.cls % 10]
            applyTransparentOverlay(context, combined, result, colorResId)
        }
        return combined
    }

    private fun applyTransparentOverlay(context: Context, overlay: Bitmap, segmentationResult: SegmentationResult, overlayColorResId: Int) {
        val width = overlay.width
        val height = overlay.height

        val overlayColor = ContextCompat.getColor(context, overlayColorResId)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val maskValue = segmentationResult.mask[y][x]
                if (maskValue > 0) {
                    overlay[x, y] = applyTransparentOverlayColor(overlayColor)
                }
            }
        }

        val canvas = Canvas(overlay)

        val boxPaint = Paint().apply {
            color = ContextCompat.getColor(context, overlayColorResId)
            strokeWidth = 2F
            style = Paint.Style.STROKE
        }

        val box = segmentationResult.box

        val left = (box.x1 * width).toInt()
        val top = (box.y1 * height).toInt()
        val right = (box.x2 * width).toInt()
        val bottom = (box.y2 * height).toInt()

//        canvas.drawRect(left.toFloat(), top.toFloat(), right.toFloat(), bottom.toFloat(), boxPaint)

        val textBackgroundPaint = Paint().apply {
            this.color = ContextCompat.getColor(context, overlayColorResId)
            this.style = Paint.Style.FILL
        }

        val textPaint = Paint().apply {
            this.color = Color.WHITE
            this.style = Paint.Style.FILL
            this.textSize = 16f
        }

        val bounds = android.graphics.Rect()
        textPaint.getTextBounds(box.clsName, 0, box.clsName.length, bounds)

        val textWidth = bounds.width()
        val textHeight = bounds.height()
        val padding = 2

        canvas.drawRect(
            left.toFloat(),
            top.toFloat() - textHeight - 2 * padding,
            left + textWidth + 2 * padding.toFloat(),
            top.toFloat(),
            textBackgroundPaint
        )
        canvas.drawText(box.clsName, left.toFloat() + padding, top.toFloat() - padding.toFloat(), textPaint)
    }

    private fun applyTransparentOverlayColor(color: Int): Int {
        val alpha = Color.alpha(color) / 2 // Adjust alpha for transparency
        val red = Color.red(color)
        val green = Color.green(color)
        val blue = Color.blue(color)

        return Color.argb(alpha, red, green, blue)
    }
}