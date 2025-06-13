package com.example.arteka_crohn

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.example.arteka_crohn.databinding.ActivityWelcomeBinding
import com.google.firebase.auth.FirebaseAuth

class WelcomeActivity : AppCompatActivity() {
    private lateinit var binding: ActivityWelcomeBinding
    private lateinit var loginButton: Button
    private lateinit var registerButton: Button
    
    override fun onCreate(savedInstanceState: Bundle?) {
        // Installer l'écran de splash avant de configurer le contenu
        installSplashScreen()
        
        super.onCreate(savedInstanceState)
        binding = ActivityWelcomeBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        loginButton = binding.buttonLogin
        registerButton = binding.buttonRegister
        
        // Configuration des boutons
        loginButton.setOnClickListener {
            val intent = Intent(this, LoginActivity::class.java)
            startActivity(intent)
        }
        
        registerButton.setOnClickListener {
            val intent = Intent(this, RegisterActivity::class.java)
            startActivity(intent)
        }
    }
    
    override fun onStart() {
        super.onStart()
        // Si l'utilisateur est déjà connecté, rediriger directement vers MainActivity
        val currentUser = FirebaseAuth.getInstance().currentUser
        if (currentUser != null) {
            navigateToMain()
        }
    }
    
    private fun navigateToMain() {
        val intent = Intent(this, MainActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }
}
