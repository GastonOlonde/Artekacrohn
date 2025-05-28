package com.example.arteka_crohn

import android.app.Dialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.DialogFragment
import com.example.arteka_crohn.R

class SimpleAlertDialogFragment : DialogFragment() {
    companion object {
        private const val ARG_TITLE = "arg_title"
        private const val ARG_MESSAGE = "arg_message"
        private const val ARG_ERROR_CODE = "arg_error_code"

        fun newInstance(title: String, message: String, errorCode: String? = null): SimpleAlertDialogFragment {
            val fragment = SimpleAlertDialogFragment()
            val args = Bundle()
            args.putString(ARG_TITLE, title)
            args.putString(ARG_MESSAGE, message)
            if (errorCode != null) args.putString(ARG_ERROR_CODE, errorCode)
            fragment.arguments = args
            return fragment
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.dialog_simple_alert, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val title = arguments?.getString(ARG_TITLE) ?: ""
        val rawMessage = arguments?.getString(ARG_MESSAGE) ?: ""
        val errorCode = arguments?.getString(ARG_ERROR_CODE)
        val userMessage = translateErrorMessage(rawMessage, errorCode)

        view.findViewById<TextView>(R.id.tvAlertTitle).text = title
        view.findViewById<TextView>(R.id.tvAlertMessage).text = userMessage

        view.findViewById<Button>(R.id.btnOk).setOnClickListener {
            dismiss()
        }
    }

    private fun translateErrorMessage(message: String, errorCode: String?): String {
        // Priorité au code d'erreur Firebase
        when (errorCode) {
            "ERROR_USER_NOT_FOUND" -> return "Ce compte n'existe pas. Veuillez créer un compte."
            "ERROR_WRONG_PASSWORD" -> return "Le mot de passe est incorrect."
            "ERROR_INVALID_EMAIL" -> return "L'adresse e-mail n'est pas valide."
            "ERROR_EMAIL_ALREADY_IN_USE" -> return "Cette adresse e-mail est déjà utilisée. Utilisez-en une autre ou connectez-vous."
            "ERROR_NETWORK_REQUEST_FAILED" -> return "Problème de connexion réseau. Vérifiez votre accès internet."
            "ERROR_TOO_MANY_REQUESTS" -> return "Trop de tentatives. Veuillez réessayer plus tard."
            "ERROR_PERMISSION_DENIED" -> return "Vous n'avez pas la permission d'effectuer cette action."
            "ERROR_MISSING_PASSWORD" -> return "Veuillez entrer un mot de passe."
            "ERROR_MISSING_EMAIL" -> return "Veuillez entrer une adresse e-mail."
        }
        // Sinon, fallback sur le texte du message
        val msg = message.lowercase()
        // Si le message contient 'auth credential is incorrect' ou 'malformed or has expired', on affiche un message neutre
        if (msg.contains("auth credential is incorrect") || msg.contains("malformed or has expired")) {
            return "L’identifiant ou le mot de passe est incorrect."
        }
        // On ne montre "compte inexistant" que si le code est explicite ou le message contient 'no user record' ou 'user does not exist'
        if (msg.contains("no user record") || msg.contains("user does not exist") || msg.contains("doesn't exist")) {
            return "Ce compte n'existe pas. Veuillez créer un compte."
        }
        return when {
            msg.contains("password is invalid") || msg.contains("invalid password") || msg.contains("wrong password") ->
                "Le mot de passe est incorrect."
            msg.contains("email address is badly formatted") || msg.contains("badly formatted") ->
                "L'adresse e-mail n'est pas valide."
            msg.contains("email address is already in use") || msg.contains("already in use") ->
                "Cette adresse e-mail est déjà utilisée. Utilisez-en une autre ou connectez-vous."
            msg.contains("network error") || msg.contains("unable to resolve host") ->
                "Problème de connexion réseau. Vérifiez votre accès internet."
            msg.contains("too many requests") ->
                "Trop de tentatives. Veuillez réessayer plus tard."
            msg.contains("permission denied") ->
                "Vous n'avez pas la permission d'effectuer cette action."
            msg.contains("missing password") ->
                "Veuillez entrer un mot de passe."
            msg.contains("missing email") ->
                "Veuillez entrer une adresse e-mail."
            else ->
                message.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
        }
    }

    override fun onStart() {
        super.onStart()
        dialog?.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
    }
}
