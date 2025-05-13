package com.example.arteka_crohn

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.arteka_crohn.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import androidx.core.graphics.createBitmap

/**
 * Activité principale de l'application Arteka Crohn.
 * Gère la capture d'images via la caméra et leur analyse par segmentation d'instance.
 *
 * Implémente InstanceSegmentationListener pour recevoir les résultats de la segmentation.
 */
class MainActivity : AppCompatActivity(), InstanceSegmentation.InstanceSegmentationListener {

    // Binding de vue généré automatiquement pour accéder aux éléments UI
    private lateinit var binding: ActivityMainBinding

    // Module de segmentation d'instance pour détecter les régions d'intérêt
    private lateinit var instanceSegmentation: InstanceSegmentation

    // Utilitaire pour dessiner les résultats de segmentation sur les images
    private lateinit var drawImages: DrawImages

    // Vue pour afficher le flux de la caméra en temps réel
    private lateinit var previewView: PreviewView

    /**
     * Méthode appelée lors de la création de l'activité.
     * Initialise l'interface utilisateur et les composants principaux.
     *
     * @param savedInstanceState État sauvegardé de l'instance (peut être null)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Active l'affichage edge-to-edge (plein écran)
        enableEdgeToEdge()

        // Charge le layout principal
        setContentView(R.layout.activity_main)

        // Initialise le view binding
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Gère les insets système (barre de statut/navigation)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Initialise la vue de prévisualisation de la caméra
        previewView = binding.previewView

        // Vérifie et demande les permissions nécessaires
        checkPermission()

        // Initialise le module de dessin des résultats
        drawImages = DrawImages(applicationContext)

        // Initialise le module de segmentation avec :
        // - Contexte de l'application
        // - Chemin vers le modèle TensorFlow Lite
        // - Chemin vers le fichier des labels
        // - Listener pour les résultats
        // - Callback pour les messages d'information
        instanceSegmentation = InstanceSegmentation(
            context = applicationContext,
            modelPath = "yolo11v1_float16.tflite",
            labelPath = "labels.txt",
            instanceSegmentationListener = this,
            message = { Toast.makeText(applicationContext, it, Toast.LENGTH_SHORT).show() }
        )
    }

    /**
     * Configure et démarre la caméra en utilisant CameraX.
     * Initialise les use cases pour :
     * - La prévisualisation en temps réel
     * - L'analyse d'image pour la segmentation
     */
    private fun startCamera() {
        // Obtient un Future du ProcessCameraProvider
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        // Ajoute un listener qui s'exécute quand le Future est complet
        cameraProviderFuture.addListener({
            try {
                // Obtient le cameraProvider une fois disponible
                val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

                // Configure le use case de prévisualisation
                val preview = Preview.Builder()
                    .build()
                    .also {
                        // Connecte le surfaceProvider à notre PreviewView
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                // Configure le use case d'analyse d'image
                val imageAnalyzer = ImageAnalysis.Builder()
                    // Ne garde que la dernière image pour éviter la surcharge
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    // Format RGBA_8888 pour la compatibilité avec Bitmap
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also {
                        // Définit notre ImageAnalyzer avec un executor dédié
                        it.setAnalyzer(Executors.newSingleThreadExecutor(), ImageAnalyzer())
                    }

                // Sélectionne la caméra arrière par défaut
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                try {
                    // Détache tous les use cases précédents
                    cameraProvider.unbindAll()
                    // Lie les use cases au lifecycle de l'activité
                    cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalyzer
                    )
                } catch (exc: Exception) {
                    Log.e("CameraX", "Échec de la liaison des use cases", exc)
                }
            } catch (exc: Exception) {
                Log.e("CameraX", "Échec de l'obtention du camera provider", exc)
            }
        }, ContextCompat.getMainExecutor(this)) // Exécute sur le thread principal
    }

    /**
     * Analyseur d'image personnalisé qui traite chaque frame capturée.
     * Convertit l'ImageProxy en Bitmap et l'envoie au module de segmentation.
     */
    inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        /**
         * Méthode appelée pour chaque frame capturée.
         *
         * @param imageProxy Conteneur des données de l'image et métadonnées
         */
        override fun analyze(imageProxy: ImageProxy) {
            // Crée un bitmap de la même taille que l'image capturée
            val bitmapBuffer = createBitmap(imageProxy.width, imageProxy.height)

            // Copie les pixels du buffer de l'image dans le bitmap
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            // Crée une matrice de transformation pour corriger l'orientation
            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            }

            // Applique la rotation au bitmap
            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            // Envoie l'image au module de segmentation
            instanceSegmentation.invoke(rotatedBitmap)
        }
    }

    /**
     * Vérifie si les permissions nécessaires sont accordées.
     * Si oui, démarre la caméra. Sinon, lance la demande de permissions.
     */
    private fun checkPermission() = lifecycleScope.launch(Dispatchers.IO) {
        val isGranted = REQUIRED_PERMISSIONS.all {
            ActivityCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }

        if (isGranted) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    // Lanceur pour la demande de permissions
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
        if (permissions.all { it.value }) {
            // Toutes les permissions accordées
            startCamera()
        } else {
            // Certaines permissions refusées
            Toast.makeText(baseContext,
                "Les permissions caméra sont requises",
                Toast.LENGTH_LONG).show()
        }
    }

    // Callback appelé en cas d'erreur lors de la segmentation
    override fun onError(error: String) {
        runOnUiThread {
            Toast.makeText(applicationContext, error, Toast.LENGTH_SHORT).show()
            // Efface l'image affichée
            binding.ivTop.setImageResource(0)
        }
    }

    // Callback appelé quand des objets sont détectés
    @SuppressLint("SetTextI18n")
    override fun onDetect(
        interfaceTime: Long,        // Temps d'inférence du modèle
        results: List<SegmentationResult>,  // Résultats de segmentation
        preProcessTime: Long,      // Temps de pré-traitement
        postProcessTime: Long      // Temps de post-traitement
    ) {
        // Dessine les résultats sur l'image
        val image = drawImages.invoke(results)

        runOnUiThread {
            // Met à jour les temps de traitement dans l'UI
            binding.tvPreprocess.text = "$preProcessTime ms"
            binding.tvInference.text = "$interfaceTime ms"
            binding.tvPostprocess.text = "$postProcessTime ms"
            binding.tvTotal.text = (interfaceTime + preProcessTime + postProcessTime).toString() + " ms"

            // Affiche l'image avec les résultats
            binding.ivTop.setImageBitmap(image)
        }
    }

    // Callback appelé quand aucun objet n'est détecté
    override fun onEmpty() {
        runOnUiThread {
            // Efface l'image affichée
            binding.ivTop.setImageResource(0)
        }
    }

    // Nettoyage lors de la destruction de l'activité
    override fun onDestroy() {
        super.onDestroy()
        // Ferme proprement le module de segmentation
        instanceSegmentation.close()
    }

    // Vérifie si toutes les permissions sont accordées
    companion object {
        // Permissions requises pour l'application
        val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA
        ).toTypedArray()
    }
}