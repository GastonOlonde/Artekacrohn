package com.example.arteka_crohn.model

import android.content.Context

object ModelManager {
    fun listModels(context: Context): List<String> {
        return context.assets.list("models")?.filter { it.endsWith(".tflite") } ?: emptyList()
    }
}
