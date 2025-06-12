package com.example.arteka_crohn.segmentation.preprocessing

import android.annotation.SuppressLint
import android.graphics.Bitmap
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer
import com.example.arteka_crohn.segmentation.model.TensorFlowLiteModel
import android.util.Log

/**
 * Interface pour le prétraitement des images
 * Respecte le principe d'ouverture/fermeture (O dans SOLID)
 */
interface ImagePreprocessor {
    fun preprocess(frame: Bitmap): Array<ByteBuffer>
}

/**
 * Implémentation concrète du préprocesseur d'images pour TensorFlow Lite
 */
class TFLiteImagePreprocessor(private val model: TensorFlowLiteModel) : ImagePreprocessor {
    
    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(TensorFlowLiteModel.INPUT_IMAGE_TYPE))
        .build()
    
    @SuppressLint("UseKtx")
    override fun preprocess(frame: Bitmap): Array<ByteBuffer> {
        // Utiliser Bitmap.createScaledBitmap qui est plus efficace
        val resizedBitmap = Bitmap.createScaledBitmap(
            frame,
            model.tensorWidth,
            model.tensorHeight,
            true // Filtre bilinéaire pour de meilleurs résultats
        )

        val tensorImage = TensorImage(TensorFlowLiteModel.INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        return arrayOf(imageProcessor.process(tensorImage).buffer)
    }
    
    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
    }
}
