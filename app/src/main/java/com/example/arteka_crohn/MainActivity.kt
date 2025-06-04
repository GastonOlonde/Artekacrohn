package com.example.arteka_crohn

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
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

    // État de l'activité
    @Volatile private var isChangingActivity = false

    // Nom du modèle sélectionné (persistant durant le cycle de vie de l'activité)
    private var selectedModelName: String = "yolo11v1_float16.tflite" // Valeur par défaut

    /**
     * Appelé lors de la création de l'activité
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        isChangingActivity = false // Flage pour indiquer si on change d'activité
        super.onCreate(savedInstanceState) // Appel de la méthode onCreate de la classe parente

        // Vérifie si l'utilisateur est connecté
        checkUserAuthentication()

        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        initializeManagers()
        setupBottomNavigation()
        setupWindowInsets()
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
        
        // Affiche un indicateur de chargement
        binding.progressBar.visibility = View.VISIBLE
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
        lifecycleScope.launch(Dispatchers.IO) {
            segmentationManager.initializeSegmentation()
            lifecycleScope.launch(Dispatchers.Main) {
                binding.progressBar.visibility = View.GONE
                checkPermissionAndStartCamera()
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
        val isGranted = REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }

        if (isGranted) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    /**
     * Démarre la caméra
     */
    private fun startCamera() {
        cameraManager.startCamera()
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
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            if (permissions.all { it.value }) {
                startCamera()
            } else {
                Toast.makeText(
                    baseContext,
                    "Les permissions caméra sont requises",
                    Toast.LENGTH_LONG
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
        // Récupérer les dimensions actuelles de l'imageView
        val imageWidth = binding.ivTop.width
        val imageHeight = binding.ivTop.height

        // Utiliser les dimensions de l'écran pour l'affichage des masques
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
        // Nettoyer la caméra uniquement
        cameraManager.cleanup()
    }

    /**
     * Appelé lorsqu'on "resume" l'activité
     */
    override fun onResume() {
        super.onResume()
        
        // Ne pas redémarrer si on change d'activité
        if (isChangingActivity) {
            return
        }
        
        // Redémarrer la caméra si les permissions sont accordées
        if (REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }) {
            startCamera()
        }
        
        // Réinitialiser le flag
        isChangingActivity = false
    }

    /**
     * Appelé lorsqu'on "destroy" l'activité
     */
    override fun onDestroy() {
        super.onDestroy()
        segmentationManager.cleanup()
        cameraManager.cleanup()
    }

    /**
     * Constantes
     */
    companion object {
        val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val DEFAULT_CONTOUR_THICKNESS = 1 // Épaisseur du contour en pixels
    }
}
