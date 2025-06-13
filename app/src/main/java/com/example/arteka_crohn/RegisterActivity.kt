package com.example.arteka_crohn

import android.content.Context
import android.content.Intent
import android.graphics.Rect
import android.os.Bundle
import android.util.Patterns
import android.view.MotionEvent
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.google.firebase.auth.FirebaseAuth
import com.example.arteka_crohn.databinding.ActivityRegisterBinding

class RegisterActivity : AppCompatActivity() {
    private lateinit var binding: ActivityRegisterBinding
    private lateinit var auth: FirebaseAuth
    private lateinit var emailEditText: EditText
    private lateinit var passwordEditText: EditText
    private lateinit var confirmPasswordEditText: EditText
    private lateinit var emailErrorTextView: android.widget.TextView
    private lateinit var passwordErrorTextView: android.widget.TextView
    private lateinit var confirmPasswordErrorTextView: android.widget.TextView
    private lateinit var registerButton: Button
    private lateinit var loginButton: Button
    private lateinit var progressBar: ImageView
    private lateinit var togglePasswordImageView: android.widget.ImageView
    private lateinit var toggleConfirmPasswordImageView: android.widget.ImageView
    private var isPasswordVisible = false
    private var isConfirmPasswordVisible = false
    private lateinit var spinnerAnimation: android.view.animation.Animation
    
    private fun showAlertDialog(title: String, message: String, errorCode: String? = null) {
        val dialog = SimpleAlertDialogFragment.newInstance(title, message, errorCode)
        dialog.show(supportFragmentManager, "SimpleAlertDialog")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRegisterBinding.inflate(layoutInflater)
        setContentView(binding.root)

        auth = FirebaseAuth.getInstance()
        emailEditText = binding.editTextEmail
        passwordEditText = binding.editTextPassword
        confirmPasswordEditText = binding.editTextConfirmPassword
        emailErrorTextView = binding.textViewEmailError
        passwordErrorTextView = binding.textViewPasswordError
        confirmPasswordErrorTextView = binding.textViewConfirmPasswordError
        registerButton = binding.buttonRegister
        loginButton = binding.buttonLogin
        progressBar = binding.progressBarRegister
        togglePasswordImageView = binding.imageViewTogglePassword
        toggleConfirmPasswordImageView = binding.imageViewToggleConfirmPassword
        
        // Initialisation de l'animation du spinner
        spinnerAnimation = android.view.animation.AnimationUtils.loadAnimation(this, R.anim.logo_spinner_rotation)

        // Désactive le bouton d'inscription par défaut
        registerButton.isEnabled = false

        // TextWatcher pour vérifier les champs email et mot de passe
        val textWatcher = object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                validateFields()
            }

            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        }

        // Ajouter les TextWatcher aux champs
        emailEditText.addTextChangedListener(textWatcher)
        passwordEditText.addTextChangedListener(textWatcher)
        confirmPasswordEditText.addTextChangedListener(textWatcher)

        // Gestionnaire de clic pour le bouton d'inscription
        registerButton.setOnClickListener {
            registerUser()
        }

        // Gestionnaire de clic pour le bouton de connexion (redirection)
        loginButton.setOnClickListener {
            finish() // Retourner à l'activité de connexion
        }

        // Configuration du basculement de visibilité du mot de passe
        togglePasswordImageView.setOnClickListener {
            isPasswordVisible = !isPasswordVisible
            togglePasswordVisibility(passwordEditText, isPasswordVisible, togglePasswordImageView)
        }

        // Configuration du basculement de visibilité de la confirmation du mot de passe
        toggleConfirmPasswordImageView.setOnClickListener {
            isConfirmPasswordVisible = !isConfirmPasswordVisible
            togglePasswordVisibility(confirmPasswordEditText, isConfirmPasswordVisible, toggleConfirmPasswordImageView)
        }
    }

    private fun togglePasswordVisibility(editText: EditText, isVisible: Boolean, toggleView: ImageView) {
        if (isVisible) {
            // Afficher le mot de passe
            editText.transformationMethod = android.text.method.HideReturnsTransformationMethod.getInstance()
            toggleView.setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
        } else {
            // Masquer le mot de passe
            editText.transformationMethod = android.text.method.PasswordTransformationMethod.getInstance()
            toggleView.setImageResource(android.R.drawable.ic_menu_view)
        }
        // Placer le curseur à la fin
        editText.setSelection(editText.text.length)
    }

    private fun validateFields() {
        val email = emailEditText.text.toString().trim()
        val password = passwordEditText.text.toString().trim()
        val confirmPassword = confirmPasswordEditText.text.toString().trim()
        
        // Validation de l'email
        if (email.isEmpty() || !Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            emailErrorTextView.text = "Format d'email invalide"
            emailErrorTextView.visibility = View.VISIBLE
        } else {
            emailErrorTextView.visibility = View.GONE
        }
        
        // Validation du mot de passe
        if (password.length < 6) {
            passwordErrorTextView.text = "Le mot de passe doit contenir au moins 6 caractères"
            passwordErrorTextView.visibility = View.VISIBLE
        } else {
            passwordErrorTextView.visibility = View.GONE
        }
        
        // Validation de la confirmation du mot de passe
        if (password != confirmPassword) {
            confirmPasswordErrorTextView.text = "Les mots de passe ne correspondent pas"
            confirmPasswordErrorTextView.visibility = View.VISIBLE
        } else {
            confirmPasswordErrorTextView.visibility = View.GONE
        }
        
        // Activer le bouton d'inscription si tous les champs sont valides
        registerButton.isEnabled = emailErrorTextView.visibility == View.GONE &&
                                  passwordErrorTextView.visibility == View.GONE &&
                                  confirmPasswordErrorTextView.visibility == View.GONE &&
                                  email.isNotEmpty() &&
                                  password.isNotEmpty() &&
                                  confirmPassword.isNotEmpty()
    }

    private fun registerUser() {
        val email = emailEditText.text.toString().trim()
        val password = passwordEditText.text.toString().trim()
        
        progressBar.visibility = View.VISIBLE
        progressBar.startAnimation(spinnerAnimation)
        
        auth.createUserWithEmailAndPassword(email, password)
            .addOnCompleteListener(this) { task ->
                progressBar.clearAnimation()
                progressBar.visibility = View.GONE
                
                if (task.isSuccessful) {
                    // Inscription réussie, rediriger vers MainActivity
                    goToMain()
                } else {
                    // Gestion des erreurs
                    val exception = task.exception
                    val errorCode = (exception as? com.google.firebase.auth.FirebaseAuthException)?.errorCode
                    val errorMessage = exception?.localizedMessage ?: "Erreur inconnue"
                    showAlertDialog("Erreur d'inscription", errorMessage, errorCode)
                }
            }
    }

    private fun goToMain() {
        val intent = Intent(this, MainActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }

    /**
     * Appelé lorsqu'un événement de touche est dispatché
     */
    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN) {
            val v = currentFocus
            if (v is EditText) {
                // Vérifier si le clic est en dehors du champ
                val outRect = Rect()
                v.getGlobalVisibleRect(outRect)
                if (!outRect.contains(event.rawX.toInt(), event.rawY.toInt())) {
                    hideKeyboard()
                }
            }
        }
        return super.dispatchTouchEvent(event)
    }

    private fun hideKeyboard() {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        currentFocus?.let { view ->
            imm.hideSoftInputFromWindow(view.windowToken, 0)
            view.clearFocus()
        }
    }
}
