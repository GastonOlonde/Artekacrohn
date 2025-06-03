package com.example.arteka_crohn

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Matrix
import android.graphics.Bitmap
import android.os.Bundle
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import android.util.Log
import android.view.View // Import pour View.GONE et View.VISIBLE
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.arteka_crohn.databinding.ActivityMainBinding
import com.google.firebase.auth.FirebaseAuth
import java.util.concurrent.Executors
import com.example.arteka_crohn.ApiSegmentationResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext // Import pour withContext


/**
 * MainActivity est responsable de l'interface utilisateur principale et de la coordination
 * entre les différents composants de l'application.
 * Elle délègue les responsabilités spécifiques à des classes dédiées.
 */
class MainActivity :
        AppCompatActivity(),
        ProfileDialogFragment.LogoutListener,
        InstanceSegmentation.InstanceSegmentationListener {

    // Verrou pour protéger l'accès à instanceSegmentation
    private val segmentationLock = Any()

    private lateinit var binding: ActivityMainBinding
    private var instanceSegmentation: InstanceSegmentation? = null // Initialisation différée
    private lateinit var drawImages: DrawImages

    private lateinit var previewView: PreviewView

    private var cameraProvider: ProcessCameraProvider? = null // Pour le garder en référence

    // Executor pour l'analyse d'image (peut être partagé ou réutilisé)
    private val imageAnalysisExecutor = Executors.newSingleThreadExecutor()

    // Nom du modèle sélectionné (persistant durant le cycle de vie de l'activité)
    private var selectedModelName: String = "yolo11v1_float16.tflite" // Valeur par défaut
    private var lastLoadedModelName: String = selectedModelName

    // Gestion du résultat de la sélection du modèle (plus nécessaire, navigation par bottomBar)
    // private val modelSelectionLauncher = ... (supprimé car plus utilisé)
    
    // État de l'activité
    @Volatile private var isChangingActivity = false

    /**
     * Arrête proprement tous les threads d'analyse d'image, libère la caméra et l'interpréteur. À
     * appeler lors d'un changement de page, pause ou logout.
     */
    private fun cleanupSegmentation() {
        isChangingActivity = true
        synchronized(segmentationLock) {
            try {
                imageAnalysisExecutor.shutdownNow()
            } catch (_: Exception) {}
            try {
                cameraProvider?.unbindAll()
            } catch (_: Exception) {}
            try {
                instanceSegmentation?.close()
            } catch (_: Exception) {}
            instanceSegmentation = null
        }
    }

    override fun onLogoutRequested() {
        cleanupAndLogout()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        isChangingActivity = false
        super.onCreate(savedInstanceState)

        // Vérifie si l'utilisateur est connecté
        val user = FirebaseAuth.getInstance().currentUser
        if (user == null) {
            val intent = android.content.Intent(this, LoginActivity::class.java)
            intent.flags =
                    android.content.Intent.FLAG_ACTIVITY_NEW_TASK or
                            android.content.Intent.FLAG_ACTIVITY_CLEAR_TASK
            startActivity(intent)
            finish()
            return
        }

        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        // tvTotal (résultat) reste affiché uniquement sur la caméra

        binding.bottomNavigation.selectedItemId = R.id.action_camera

        // --- Ajout navigation barre du bas ---
        // Corrigé : lambda reçoit un MenuItem
        binding.bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.action_profile -> {
                    cleanupSegmentation()
                    startActivity(Intent(this, ProfileActivity::class.java))
                    overridePendingTransition(R.anim.slide_in_right, R.anim.slide_out_left)
                    finish()
                    true
                }
                R.id.action_models -> {
                    val intent = Intent(this, ModelSelectionActivity::class.java)
                    startActivity(intent)
                    overridePendingTransition(R.anim.slide_in_left, R.anim.slide_out_right)
                    finish()
                    true
                }
                R.id.action_camera -> true // déjà sur camera
                else -> false
            }
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        previewView = binding.previewView
        drawImages = DrawImages(applicationContext)

        // Affiche un indicateur de chargement (ProgressBar)
        // Assure-toi d'avoir un ProgressBar dans ton layout activity_main.xml
        // avec par exemple l'id "progressBar"
        binding.progressBar.visibility =
                View.VISIBLE // Suppose que tu as un progressBar dans ton layout

        // Initialise le modèle sélectionné en arrière-plan
        initializeSegmentation()
    }

    // Fonction pour initialiser InstanceSegmentation avec le modèle choisi
    private fun initializeSegmentation() {
        binding.progressBar.visibility = View.VISIBLE
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d("MainActivity", "Starting InstanceSegmentation initialization with $selectedModelName ...")
                val segmenter = InstanceSegmentation(
                    context = applicationContext,
                    modelPath = "models/$selectedModelName",
                    labelPath = "labels/labels.txt",
                    instanceSegmentationListener = this@MainActivity,
                    message = { msg ->
                        lifecycleScope.launch(Dispatchers.Main) {
                            Toast.makeText(applicationContext, msg, Toast.LENGTH_SHORT).show()
                        }
                    }
                )
                instanceSegmentation = segmenter
                Log.d("MainActivity", "InstanceSegmentation initialized successfully.")
                withContext(Dispatchers.Main) {
                    binding.progressBar.visibility = View.GONE
                    checkPermissionAndStartCamera()
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error initializing InstanceSegmentation", e)
                withContext(Dispatchers.Main) {
                    binding.progressBar.visibility = View.GONE
                    Toast.makeText(
                        applicationContext,
                        "Failed to load model: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    private fun cleanupAndLogout() {
        cleanupSegmentation()
        // Déconnecte l’utilisateur et redirige
        FirebaseAuth.getInstance().signOut()
        val intent = Intent(this, LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }

    private fun goToLogin() {
        val intent = Intent(this, LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }
    /**
     * Vérifie les permissions et démarre la caméra si elles sont accordées. Doit être appelé APRÈS
     * que instanceSegmentation soit initialisé.
     */
    private fun checkPermissionAndStartCamera() {
        // Vérifie que instanceSegmentation est initialisé avant de continuer
        if (instanceSegmentation == null) {
            Log.e("MainActivity", "InstanceSegmentation not initialized, cannot start camera.")
            Toast.makeText(this, "Model not ready.", Toast.LENGTH_SHORT).show()
            return
        }

        val isGranted =
                REQUIRED_PERMISSIONS.all {
                    ContextCompat.checkSelfPermission(baseContext, it) ==
                            PackageManager.PERMISSION_GRANTED
                }

        if (isGranted) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(
                {
                    try {
                        cameraProvider = cameraProviderFuture.get() // Stocke la référence
                        val preview =
                                Preview.Builder().build().also {
                                    it.surfaceProvider = previewView.surfaceProvider
                                }
                        val imageAnalyzerUseCase =
                                ImageAnalysis.Builder()
                                        .setBackpressureStrategy(
                                                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                                        )
                                        .setOutputImageFormat(
                                                ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                                        )
                                        .build()
                                        .also {
                                            it.setAnalyzer(imageAnalysisExecutor, ImageAnalyzer())
                                        }
                        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                        cameraProvider?.unbindAll()
                        cameraProvider?.bindToLifecycle(
                                this,
                                cameraSelector,
                                preview,
                                imageAnalyzerUseCase
                        )
                    } catch (exc: Exception) {
                        Log.e("CameraX", "Failed to bind/get camera provider", exc)
                    }
                },
                ContextCompat.getMainExecutor(this)
        )
    }

    inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        @SuppressLint("UnsafeOptInUsageError") // Nécessaire pour imageProxy.image si tu l'utilises
        override fun analyze(imageProxy: ImageProxy) {
            synchronized(segmentationLock) {
                if (isChangingActivity) {
                    imageProxy.close()
                    return
                }
                val currentSegmenter = instanceSegmentation
                if (currentSegmenter == null) {
                    imageProxy.close()
                    return
                }

                val bitmapBuffer = createBitmap(imageProxy.width, imageProxy.height)
                imageProxy.use { proxy ->
                    val buffer = proxy.planes[0].buffer
                    buffer.rewind()
                    bitmapBuffer.copyPixelsFromBuffer(buffer)
                }
                val matrix =
                        Matrix().apply {
                            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                        }
                val rotatedBitmap =
                        Bitmap.createBitmap(
                                bitmapBuffer,
                                0,
                                0,
                                bitmapBuffer.width,
                                bitmapBuffer.height,
                                matrix,
                                true
                        )
                currentSegmenter.invoke(rotatedBitmap)
            }
        }
    }

    // `checkPermission` est maintenant fusionné dans `checkPermissionAndStartCamera`
    // et n'est plus lancé directement avec lifecycleScope.launch(Dispatchers.IO) dans onCreate

    private val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
                    permissions ->
                if (permissions.all { it.value }) {
                    startCamera()
                } else {
                    Toast.makeText(
                                    baseContext,
                                    "Les permissions caméra sont requises",
                                    Toast.LENGTH_LONG
                            )
                            .show()
                }
            }

    override fun onError(onError: String) {
        runOnUiThread {
            Toast.makeText(applicationContext, onError, Toast.LENGTH_SHORT).show()
            binding.ivTop.setImageResource(0)
        }
    }

    @SuppressLint("SetTextI18n")
    override fun onDetect(
            interfaceTime: Long,
            results: List<ApiSegmentationResult>,
            preProcessTime: Long,
            postProcessTime: Long
    ) {
        // Récupérer les dimensions actuelles de l'imageView
        val imageWidth = binding.ivTop.width
        val imageHeight = binding.ivTop.height

        // Utiliser les dimensions de l'écran pour l'affichage des masques
        val resultBitmap =
                drawImages.invoke(
                        results = results,
                        contourThickness = DEFAULT_CONTOUR_THICKNESS,
                        screenWidth = imageWidth,
                        screenHeight = imageHeight,
                        scaleFactor =
                                1.0f // Ajustez selon les besoins (0.5f pour plus rapide, 1.0f pour
                        // meilleure qualité)
                        )

        runOnUiThread {
            binding.tvInferenceTime.text =
                    (interfaceTime + preProcessTime + postProcessTime).toString() + " ms"
            binding.ivTop.setImageBitmap(resultBitmap)
        }
    }

    companion object {
        val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA) // Simplifié
        private const val DEFAULT_CONTOUR_THICKNESS = 2 // Épaisseur du contour en pixels augmentée
    }

    override fun onEmpty() {
        runOnUiThread { binding.ivTop.setImageResource(0) }
    }

    override fun onPause() {
        super.onPause()
        cleanupSegmentation()
    }

    override fun onDestroy() {
        super.onDestroy()
        cleanupSegmentation()
    }
}
