package com.example.arteka_crohn

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.example.arteka_crohn.segmentation.ApiSegmentationResult
import com.example.arteka_crohn.camera.CameraManager
import com.example.arteka_crohn.camera.ImageAnalysisCallback
import com.example.arteka_crohn.databinding.ActivityMainBinding
import com.example.arteka_crohn.segmentation.InstanceSegmentationListener
import com.example.arteka_crohn.segmentation.SegmentationManager
import com.google.firebase.auth.FirebaseAuth
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * MainActivity est responsable de l'interface utilisateur principale et de la coordination
 * entre les différents composants de l'application.
 * Elle délègue les responsabilités spécifiques à des classes dédiées.
 */
class MainActivity :
        AppCompatActivity(),
        InstanceSegmentationListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var drawImages: DrawImages
    private lateinit var cameraManager: CameraManager
    private lateinit var segmentationManager: SegmentationManager
    private lateinit var spinnerAnimation: android.view.animation.Animation

    // État de l'activité
    @Volatile private var isChangingActivity = false

    // Nom du modèle sélectionné (persistant durant le cycle de vie de l'activité)
    private var selectedModelName: String = "yolo11v1_float16.tflite" // Valeur par défaut

    /**
     * Appelé lors de la création de l'activité
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        
        isChangingActivity = false // Flage pour indiquer si on change d'activité

        installSplashScreen()

        super.onCreate(savedInstanceState) // Appel de la méthode onCreate de la classe parente

        // Vérifie si l'utilisateur est connecté
        checkUserAuthentication()

        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialisation de l'animation du spinner
        spinnerAnimation = android.view.animation.AnimationUtils.loadAnimation(this, R.anim.logo_spinner_rotation)

        // Récupérer le modèle sélectionné depuis les préférences
        val prefs = getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
        selectedModelName = prefs.getString("SELECTED_MODEL", selectedModelName) ?: selectedModelName
        
        // S'il n'y a pas de modèle sélectionné dans les préférences, définir le modèle par défaut
        if (!prefs.contains("SELECTED_MODEL")) {
            prefs.edit().putString("SELECTED_MODEL", selectedModelName).apply()
        }

        setupUI()
        setupBottomNavigation()
        setupWindowInsets()
        checkPermissionAndStartCamera()  // Déplacé ici après l'initialisation de l'UI
    }

    /**
     * Vérifie si l'utilisateur est connecté
     */
    private fun checkUserAuthentication() {
        val user = FirebaseAuth.getInstance().currentUser
        if (user == null) {
            goToLogin()
        }
    }

    /**
     * Configure l'interface utilisateur
     */
    private fun setupUI() {
        drawImages = DrawImages(applicationContext)
        
        // Affiche un indicateur de chargement avec animation
        binding.progressBar.visibility = View.VISIBLE
        binding.progressBar.startAnimation(spinnerAnimation)
    }

    /**
     * Initialise les gestionnaires
     */
    private fun initializeManagers() {
        // Initialisation des gestionnaires
        cameraManager = CameraManager(
            this,
            this,
            binding.previewView,
            object : ImageAnalysisCallback {
                override fun onImageAnalyzed(bitmap: android.graphics.Bitmap) {
                    segmentationManager.onImageAnalyzed(bitmap)
                }
            }
        )
        
        segmentationManager = SegmentationManager(
            this,
            this,
            selectedModelName,
            this
        )
        
        // Initialiser la segmentation
        binding.previewView.post {
            // Initialiser la segmentation
            lifecycleScope.launch(Dispatchers.IO) {
                segmentationManager.initializeSegmentation()
                lifecycleScope.launch(Dispatchers.Main) {
                    binding.progressBar.clearAnimation()
                    binding.progressBar.visibility = View.GONE
                }
            }
        }
    }

    /**
     * Configure la navigation
     */
    private fun setupBottomNavigation() {
        binding.bottomNavigation.selectedItemId = R.id.action_camera
        
        binding.bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.action_profile -> {
                    isChangingActivity = true
                    val intent = Intent(this, ProfileActivity::class.java)
                    startActivity(intent)
                    overridePendingTransition(R.anim.slide_in_right, R.anim.slide_out_left)
                    finish()
                    true
                }
                R.id.action_models -> {
                    isChangingActivity = true
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
    }
    
    /**
     * Configure les insets de la fenêtre
     */
    private fun setupWindowInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

    /**
     * Vérifie les permissions et démarre la caméra si elles sont accordées
     */
    private fun checkPermissionAndStartCamera() {
        when {
            REQUIRED_PERMISSIONS.all {
                ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
            } -> {
                Log.d(TAG, "Toutes les permissions sont accordées, démarrage de la caméra")
                lifecycleScope.launch(Dispatchers.Main) {
                    initializeManagers()
                    startCamera()
                }
            }
            else -> {
                Log.d(TAG, "Demande des permissions")
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }
    }

    /**
     * Démarre la caméra
     */
    private fun startCamera() {
        // Vérifier que la vue de prévisualisation est initialisée
        if (binding.previewView.width == 0 || binding.previewView.height == 0) {
            // Si la vue n'est pas encore mesurée, attendre qu'elle le soit
            binding.previewView.post {
                cameraManager.startCamera()
            }
        } else {
            cameraManager.startCamera()
        }
    }

    /**
     * Navigue vers l'activité de connexion
     */
    private fun goToLogin() {
        val intent = Intent(this, LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }

    /**
     * Lance une demande de permissions et démarre la caméra si elles sont accordées
     */
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
            Log.d(TAG, "Permissions accordées par l'utilisateur")
            lifecycleScope.launch(Dispatchers.Main) {
                initializeManagers()
                startCamera()
            }
        } else {
            Toast.makeText(
                baseContext,
                "Permissions refusées. La caméra ne peut pas être utilisée.",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    /**
     * Gestionnaire d'erreur
     */
    override fun onError(error: String) {
        runOnUiThread {
            Toast.makeText(applicationContext, error, Toast.LENGTH_SHORT).show()
            binding.ivTop.setImageResource(0)
        }
    }

    /**
     * Appelé lorsqu'un résultat de segmentation est détecté
     */
    @SuppressLint("SetTextI18n")
    override fun onDetect(
        interfaceTime: Long,
        results: List<ApiSegmentationResult>,
        preProcessTime: Long,
        postProcessTime: Long
    ) {
        // Arrêter l'animation et cacher le spinner
        runOnUiThread {
            binding.progressBar.clearAnimation()
            binding.progressBar.visibility = View.GONE
        }
        // Récupérer les dimensions actuelles de l'imageView
        val imageWidth = binding.ivTop.width
        val imageHeight = binding.ivTop.height

        // Utiliser les dimensions de l'écran pour l'affichage des masques
        val drawImages = DrawImages(applicationContext)
        val resultBitmap = drawImages.invoke(
            results = results,
            contourThickness = DEFAULT_CONTOUR_THICKNESS,
            screenWidth = imageWidth,
            screenHeight = imageHeight,
            scaleFactor = 1.0f // Ajustez selon les besoins
        )

        runOnUiThread {
            binding.tvInferenceTime.text = (interfaceTime + preProcessTime + postProcessTime).toString() + " ms"
            binding.ivTop.setImageBitmap(resultBitmap)
        }
    }

    /**
     * Appelé lorsqu'il n'y a pas de résultats de segmentation
     */
    override fun onEmpty() {
        runOnUiThread { binding.ivTop.setImageResource(0) }
    }

    /**
     * Appelé lorsqu'on "pause" l'activité
     */
    override fun onPause() {
        super.onPause()
        // Nettoyer la caméra uniquement si elle est initialisée
        if (::cameraManager.isInitialized) {
            cameraManager.cleanup()
        }
    }

    /**
     * Appelé lorsque l'activité reprend le focus
     */
    override fun onResume() {
        super.onResume()
        
        if (isChangingActivity) {
            isChangingActivity = false
            return
        }

        if (::cameraManager.isInitialized && REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }) {
            Log.d(TAG, "Reprise de la caméra dans onResume")
            cameraManager.resumeCamera()
        }
    }

    /**
     * Appelé lorsqu'on "destroy" l'activité
     */
    override fun onDestroy() {
        super.onDestroy()
        if (::segmentationManager.isInitialized) {
            segmentationManager.cleanup()
        }
        if (::cameraManager.isInitialized) {
            cameraManager.cleanup()
        }
    }

    /**
     * Constantes
     */
    companion object {
        private const val TAG = "MainActivity"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val DEFAULT_CONTOUR_THICKNESS = 1 // Épaisseur du contour en pixels
    }
}
