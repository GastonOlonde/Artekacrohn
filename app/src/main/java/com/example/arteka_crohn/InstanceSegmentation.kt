package com.example.arteka_crohn

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.example.arteka_crohn.ImageUtils.clone
import com.example.arteka_crohn.ImageUtils.scaleMask
import com.example.arteka_crohn.MetaData.extractNamesFromLabelFile
import com.example.arteka_crohn.MetaData.extractNamesFromMetadata
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer


class InstanceSegmentation(
    context: Context,
    modelPath: String,
    labelPath: String?,
    private val instanceSegmentationListener: InstanceSegmentationListener,
    private val message: (String) -> Unit
) {
    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private var xPoints = 0
    private var yPoints = 0
    private var masksNum = 0
    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    private var gpuDelegateInstance: org.tensorflow.lite.gpu.GpuDelegate? = null
    private var nnApiDelegateInstance: org.tensorflow.lite.nnapi.NnApiDelegate? = null

    init {
        try {
            val options = Interpreter.Options()
            var delegateAppliedInfo = "No delegate tried or fallback to CPU."

            // Tentative 1: GPU Delegate avec options
            try {
                val delegateOptions = org.tensorflow.lite.gpu.GpuDelegateFactory.Options().apply {
                    setPrecisionLossAllowed(true) // Important si ton modèle peut tourner en FP16 sur GPU

                    setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENCL)
                }
                gpuDelegateInstance = org.tensorflow.lite.gpu.GpuDelegate(delegateOptions)

                options.addDelegate(gpuDelegateInstance)
                delegateAppliedInfo = "Attempting to use GPU Delegate."
                // Le vrai succès sera loggué par TFLite lui-même lors de l'init de l'Interpreter
            } catch (e: Throwable) {
                Log.w("TFLite_GPU", "GPU Delegate creation/configuration failed. Falling back.", e)
                gpuDelegateInstance = null // Assure-toi qu'il est null s'il échoue
                // Ne pas s'arrêter ici, on essaiera NNAPI ou CPU
            }

            // Tentative 2: NNAPI Delegate (si GPU n'est pas explicitement voulu ou a échoué)
            if (gpuDelegateInstance == null) {
                try {
                    val nnApiDelegate = org.tensorflow.lite.nnapi.NnApiDelegate()
                    options.addDelegate(nnApiDelegate)

                    delegateAppliedInfo = "NNAPI Delegate configured. GPU delegate failed or was not attempted."
                    Log.d("TFLite", "NNAPI delegate added.");
                } catch (e: Exception) {
                    Log.w("TFLite_NNAPI", "NNAPI Delegate not available or failed: ${e.message}")
                    // Tombera sur CPU + XNNPACK
                    delegateAppliedInfo = "GPU and NNAPI delegates failed or not available. Using CPU."
                }
            }


            // XNNPACK pour CPU (toujours une bonne idée en fallback ou si aucun delegate n'est utilisé)
            options.setUseXNNPACK(true)
            // options.setNumThreads(4) // Teste avec plus de threads si tu es souvent sur CPU

            Log.d("TFLiteConfig", "Delegate status before interpreter init: $delegateAppliedInfo")

            val model = FileUtil.loadMappedFile(context, modelPath)
            interpreter = Interpreter(model, options) // C'est ICI que le delegate est VRAIMENT appliqué

            Log.i("TFLite", "Interpreter initialized. Check Logcat for TFLite's own delegate messages.")

            labels.addAll(extractNamesFromMetadata(model))
            if (labels.isEmpty()) {
                if (labelPath == null) {
                    message("Model not contains metadata, provide LABELS_PATH in Constants.kt")
                    labels.addAll(MetaData.TEMP_CLASSES)
                } else {
                    labels.addAll(extractNamesFromLabelFile(context, labelPath))
                }
            }

            val inputShape = interpreter.getInputTensor(0)?.shape()
            val outputShape0 = interpreter.getOutputTensor(0)?.shape()
            val outputShape1 = interpreter.getOutputTensor(1)?.shape()

            if (inputShape != null) {
                tensorWidth = inputShape[1]
                tensorHeight = inputShape[2]
                if (inputShape[1] == 3) {
                    tensorWidth = inputShape[2]
                    tensorHeight = inputShape[3]
                }
            }

            if (outputShape0 != null) {
                numChannel = outputShape0[1]
                numElements = outputShape0[2]
            }

            if (outputShape1 != null) {
                if (outputShape1[1] == 32) {
                    masksNum = outputShape1[1]
                    xPoints = outputShape1[2]
                    yPoints = outputShape1[3]
                } else {
                    xPoints = outputShape1[1]
                    yPoints = outputShape1[2]
                    masksNum = outputShape1[3]
                }
            }

        } catch (e: Exception) {
            Log.e("TFLite", "Error initializing interpreter", e)
            message("Failed to load model: ${e.message}")
            throw e
        }
    }


    fun close() {
        interpreter.close()
        gpuDelegateInstance?.close()
        nnApiDelegateInstance?.close()
    }

    fun invoke(frame: Bitmap) {
        if (tensorWidth == 0 || tensorHeight == 0 ||
            numChannel == 0 || numElements == 0 ||
            xPoints == 0 || yPoints == 0 || masksNum == 0
        ) {
            instanceSegmentationListener.onError("Interpreter not initialized properly")
            return
        }

        val t0 = SystemClock.uptimeMillis()
        val imageBuffer = preProcess(frame)
        val t1 = SystemClock.uptimeMillis()
        val preProcessTime = t1 - t0
        //Log.d("SystemClockProfiler", "Preprocess time: $preProcessTime ms")

        val coordinatesBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, numChannel, numElements),
            OUTPUT_IMAGE_TYPE
        )
        val maskProtoBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, xPoints, yPoints, masksNum),
            OUTPUT_IMAGE_TYPE
        )

        val outputBuffer = mapOf<Int, Any>(
            0 to coordinatesBuffer.buffer.rewind(),
            1 to maskProtoBuffer.buffer.rewind()
        )

        val t2 = SystemClock.uptimeMillis()
        interpreter.runForMultipleInputsOutputs(imageBuffer, outputBuffer)
        val t3 = SystemClock.uptimeMillis()
        val interfaceTime = t3 - t2
        //Log.d("SystemClockProfiler", "Inference time: $interfaceTime ms")

        val t4 = SystemClock.uptimeMillis()

        // Traitement des boîtes détectées
        val bestBoxes = bestBox(coordinatesBuffer.floatArray) ?: run {
            instanceSegmentationListener.onEmpty()
            return
        }
        
        // Traitement des masques
        val maskProto = reshapeMaskOutput(maskProtoBuffer.floatArray)
        
        // Création des résultats de segmentation
        val segmentationResults = bestBoxes.map {
            ApiSegmentationResult(
                box = it,
                mask = getFinalMask(it, maskProto),
                conf = it.cnf,
            )
        }
        val t4f = SystemClock.uptimeMillis()
        // Log.d("SystemClockProfiler", "Final mask generation time: ${t4f - t4e} ms")

        val postProcessTime = t4f - t4
        // Log.d("SystemClockProfiler", "Postprocess total time: $postProcessTime ms")

        instanceSegmentationListener.onDetect(
            interfaceTime = interfaceTime,
            results = segmentationResults,
            preProcessTime = preProcessTime,
            postProcessTime = postProcessTime
        )
    }

    private fun getFinalMask(output0: Output0, output1: List<Array<FloatArray>>): Array<FloatArray> {

        return try {
            val output1Copy = output1.clone()
            val relX1 = (output0.x1 * xPoints).toInt()
            val relY1 = (output0.y1 * yPoints).toInt()
            val relX2 = (output0.x2 * xPoints).toInt()
            val relY2 = (output0.y2 * yPoints).toInt()

            val zero = Array(yPoints) { FloatArray(xPoints) }

            // Optimisation: réduire les boucles imbriquées
            for (index in output1Copy.indices) {
                val proto = output1Copy[index]
                val weight = output0.maskWeight[index]

                for (y in maxOf(relY1, 0) until minOf(relY2, yPoints)) {
                    for (x in maxOf(relX1, 0) until minOf(relX2, xPoints)) {
                        zero[y][x] += proto[y][x] * weight
                    }
                }
            }
            zero.scaleMask(450, 600)
        } catch (e: Exception) {
            Log.e("MaskError", "Error in mask generation", e)
            Array(yPoints) { FloatArray(xPoints) { 0f } }
        }
    }

    private fun reshapeMaskOutput(floatArray: FloatArray): List<Array<FloatArray>> {
        return List(masksNum) { mask ->
            Array(xPoints) { r ->
                FloatArray(yPoints) { c ->
                    floatArray[masksNum * yPoints * r + masksNum * c + mask]
                }
            }
        }
    }

    private fun bestBox(array: FloatArray): List<Output0>? {
        val output0List = ArrayList<Output0>(numElements) // Pré-allocation

        for (c in 0 until numElements) {
            var maxConf = CONFIDENCE_THRESHOLD
            var maxIdx = -1
            var currentInd = 4

            // Optimisation: calcul d'index simplifié
            while (currentInd < (numChannel - masksNum)) {
                val conf = array[c + numElements * currentInd]
                if (conf > maxConf) {
                    maxConf = conf
                    maxIdx = currentInd - 4
                }
                currentInd++
            }

            if (maxConf > CONFIDENCE_THRESHOLD && maxIdx in labels.indices) {
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - w/2f
                val y1 = cy - h/2f
                val x2 = cx + w/2f
                val y2 = cy + h/2f

                if (x1 in 0f..1f && y1 in 0f..1f && x2 in 0f..1f && y2 in 0f..1f) {
                    val maskWeights = FloatArray(masksNum) { i ->
                        array[c + numElements * (currentInd + i)]
                    }

                    output0List.add(
                        Output0(
                            x1, y1, x2, y2, cx, cy, w, h,
                            maxConf, maxIdx, labels[maxIdx],
                            maskWeights.toList()
                        )
                    )
                }
            }
        }

        return output0List.takeIf { it.isNotEmpty() }?.let { applyNMS(it) }
    }

    private fun applyNMS(output0List: List<Output0>): MutableList<Output0> {
        val sortedBoxes = output0List.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<Output0>()
        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.removeAt(0)
            selectedBoxes.add(first)
            sortedBoxes.removeAll { calculateIoU(first, it) >= IOU_THRESHOLD }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: Output0, box2: Output0): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    @SuppressLint("UseKtx")
    private fun preProcess(frame: Bitmap): Array<ByteBuffer> {
        // Utiliser Bitmap.createScaledBitmap qui est plus efficace
        val resizedBitmap = Bitmap.createScaledBitmap(
            frame,
            tensorWidth,
            tensorHeight,
            true // Filtre bilinéaire pour de meilleurs résultats
        )

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        return arrayOf(imageProcessor.process(tensorImage).buffer)
    }

    interface InstanceSegmentationListener {
        fun onError(error: String)
        fun onEmpty()
        fun onDetect(
            interfaceTime: Long,
            results: List<ApiSegmentationResult>,
            preProcessTime: Long,
            postProcessTime: Long
        )
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.45F
        private const val IOU_THRESHOLD = 0.5F
    }
}