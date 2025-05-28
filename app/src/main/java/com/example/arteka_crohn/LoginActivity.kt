package com.example.arteka_crohn

import android.content.Intent
import android.os.Bundle
import android.util.Patterns
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.firebase.auth.FirebaseAuth

class LoginActivity : AppCompatActivity() {
    private fun showAlertDialog(title: String, message: String, errorCode: String? = null) {
        val dialog = SimpleAlertDialogFragment.newInstance(title, message, errorCode)
        dialog.show(supportFragmentManager, "SimpleAlertDialog")
    }
    private lateinit var auth: FirebaseAuth
    private lateinit var emailEditText: EditText
    private lateinit var passwordEditText: EditText
    private lateinit var emailErrorTextView: android.widget.TextView
    private lateinit var passwordErrorTextView: android.widget.TextView
    private lateinit var loginButton: Button
    private lateinit var signupButton: Button
    private lateinit var forgotPasswordButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var togglePasswordImageView: android.widget.ImageView
    private lateinit var scrollView: android.widget.ScrollView
    private var isPasswordVisible = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        auth = FirebaseAuth.getInstance()
        emailEditText = findViewById(R.id.editTextEmail)
        passwordEditText = findViewById(R.id.editTextPassword)
        emailErrorTextView = findViewById(R.id.textViewEmailError)
        passwordErrorTextView = findViewById(R.id.textViewPasswordError)
        loginButton = findViewById(R.id.buttonLogin)
        signupButton = findViewById(R.id.buttonSignup)
        forgotPasswordButton = findViewById(R.id.buttonForgotPassword)
        progressBar = findViewById(R.id.progressBarLogin)
        togglePasswordImageView = findViewById(R.id.imageViewTogglePassword)
        scrollView = findViewById(R.id.scrollViewLogin)
        // Suppression de toute logique de scroll automatique liée au clavier/focus


        forgotPasswordButton.setOnClickListener {
            val email = emailEditText.text.toString().trim()
            if (email.isEmpty()) {
                // Afficher un dialog pour saisir l'email
                val input = EditText(this)
                input.inputType = android.text.InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS
                input.hint = "Entrez votre adresse e-mail"
                androidx.appcompat.app.AlertDialog.Builder(this)
                    .setTitle("Réinitialiser le mot de passe")
                    .setMessage("Veuillez saisir votre adresse e-mail pour recevoir un lien de réinitialisation.")
                    .setView(input)
                    .setPositiveButton("Envoyer") { _, _ ->
                        val enteredEmail = input.text.toString().trim()
                        if (enteredEmail.isNotEmpty()) {
                            sendPasswordReset(enteredEmail)
                        } else {
                            showAlertDialog("Erreur", "Veuillez entrer une adresse e-mail valide.")
                        }
                    }
                    .setNegativeButton("Annuler", null)
                    .show()
            } else {
                sendPasswordReset(email)
            }
        }
    

        // Désactive le bouton de connexion par défaut
        loginButton.isEnabled = false

        // Ajoute un TextWatcher pour vérifier l'existence de l'email
        // Active le bouton Connexion seulement si email et mot de passe sont valides
        val watcher = object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                val email = emailEditText.text.toString().trim()
                val password = passwordEditText.text.toString()

                // Email
                if (email.isEmpty() || !android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
                    emailErrorTextView.text = "Format d'email invalide"
                    emailErrorTextView.visibility = android.view.View.VISIBLE
                } else {
                    emailErrorTextView.text = ""
                    emailErrorTextView.visibility = android.view.View.GONE
                }

                // Mot de passe
                if (password.isNotEmpty() && password.length < 6) {
                    passwordErrorTextView.text = "Le mot de passe doit contenir au moins 6 caractères"
                    passwordErrorTextView.visibility = android.view.View.VISIBLE
                } else {
                    passwordErrorTextView.text = ""
                    passwordErrorTextView.visibility = android.view.View.GONE
                }

                loginButton.isEnabled = email.isNotEmpty() &&
                        android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches() &&
                        password.length >= 6
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        }
        emailEditText.addTextChangedListener(watcher)
        passwordEditText.addTextChangedListener(watcher)


        // Désactive le bouton de connexion si le mot de passe est vide ou trop court
        passwordEditText.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                val password = s.toString()
                loginButton.isEnabled = loginButton.isEnabled && password.length >= 6
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })

        togglePasswordImageView.setOnClickListener {
            isPasswordVisible = !isPasswordVisible
            if (isPasswordVisible) {
                passwordEditText.inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
                togglePasswordImageView.setImageResource(android.R.drawable.ic_menu_view)
            } else {
                passwordEditText.inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
                togglePasswordImageView.setImageResource(android.R.drawable.ic_menu_view)
            }
            // Pour garder le curseur à la fin
            passwordEditText.setSelection(passwordEditText.text.length)
        }

        loginButton.setOnClickListener {
            loginUser()
        }

        signupButton.setOnClickListener {
            signupUser()
        }
    }

    override fun onStart() {
        super.onStart()
        if (auth.currentUser != null) {
            goToMain()
        }
    }

    private fun sendPasswordReset(email: String) {
        progressBar.visibility = View.VISIBLE
        auth.sendPasswordResetEmail(email)
            .addOnCompleteListener { task ->
                progressBar.visibility = View.GONE
                if (task.isSuccessful) {
                    showAlertDialog("Réinitialisation envoyée", "Un email de réinitialisation a été envoyé à $email.")
                } else {
                    val exception = task.exception
                    val errorCode = (exception as? com.google.firebase.auth.FirebaseAuthException)?.errorCode
                    val errorMessage = exception?.localizedMessage ?: "Erreur inconnue"
                    showAlertDialog("Erreur", errorMessage, errorCode)
                }
            }
    }

    private fun loginUser() {
        val email = emailEditText.text.toString().trim()
        val password = passwordEditText.text.toString().trim()
        if (!validateInput(email, password)) return
        progressBar.visibility = View.VISIBLE
        auth.signInWithEmailAndPassword(email, password)
            .addOnCompleteListener(this) { task ->
                progressBar.visibility = View.GONE
                if (task.isSuccessful) {
                    goToMain()
                } else {
                    // Message global pour éviter toute fuite d'info
                    showAlertDialog(
                        "Erreur de connexion",
                        "L’identifiant ou le mot de passe est incorrect, ou le compte n’existe pas."
                    )
                }
            }
    }

    private fun signupUser() {
        val email = emailEditText.text.toString().trim()
        val password = passwordEditText.text.toString().trim()
        if (!validateInput(email, password)) return
        progressBar.visibility = View.VISIBLE
        auth.createUserWithEmailAndPassword(email, password)
            .addOnCompleteListener(this) { task ->
                progressBar.visibility = View.GONE
                if (task.isSuccessful) {
                    goToMain()
                } else {
                    val exception = task.exception
                    val errorCode = (exception as? com.google.firebase.auth.FirebaseAuthException)?.errorCode
                    val errorMessage = exception?.localizedMessage ?: "Erreur inconnue"
                    showAlertDialog("Erreur d'inscription", errorMessage, errorCode)
                }
            }
    }

    private fun validateInput(email: String, password: String): Boolean {
        var valid = true
        if (email.isEmpty() || !Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            emailErrorTextView.text = "Format d'email invalide"
            emailErrorTextView.visibility = View.VISIBLE
            valid = false
        } else {
            emailErrorTextView.text = ""
            emailErrorTextView.visibility = View.GONE
        }
        if (password.length < 6) {
            passwordErrorTextView.text = "Le mot de passe doit contenir au moins 6 caractères"
            passwordErrorTextView.visibility = View.VISIBLE
            valid = false
        } else {
            passwordErrorTextView.text = ""
            passwordErrorTextView.visibility = View.GONE
        }
        return valid
    }

    private fun goToMain() {
        val intent = Intent(this, MainActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }
}
