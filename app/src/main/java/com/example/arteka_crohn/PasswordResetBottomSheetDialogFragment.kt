package com.example.arteka_crohn

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.bottomsheet.BottomSheetDialogFragment

class PasswordResetBottomSheetDialogFragment : BottomSheetDialogFragment() {

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.dialog_password_reset, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
    
        val emailInput = view.findViewById<EditText>(R.id.emailInput)
        val emailErrorTextView = view.findViewById<TextView>(R.id.emailErrorTextView)
        val sendButton = view.findViewById<Button>(R.id.sendButton)
    
        // Ajouter un TextWatcher pour surveiller les changements de texte
        emailInput.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                val email = s.toString().trim()
                val isValidEmail = android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()
                sendButton.isEnabled = isValidEmail
    
                if (!isValidEmail && email.isNotEmpty()) {
                    emailErrorTextView.text = "Format d'email invalide"
                    emailErrorTextView.visibility = View.VISIBLE
                } else {
                    emailErrorTextView.text = ""
                    emailErrorTextView.visibility = View.GONE
                }
            }
    
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })
    
        sendButton.setOnClickListener {
            val email = emailInput.text.toString().trim()
            if (email.isNotEmpty()) {
                (activity as? LoginActivity)?.sendPasswordReset(email)
                dismiss()
            } else {
                Toast.makeText(context, "Veuillez entrer une adresse e-mail valide.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    companion object {
        fun newInstance(): PasswordResetBottomSheetDialogFragment {
            return PasswordResetBottomSheetDialogFragment()
        }
    }
}