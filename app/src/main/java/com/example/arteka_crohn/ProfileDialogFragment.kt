// Fichier supprimé : ProfileDialogFragment n'est plus utilisé.

import android.app.Dialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.DialogFragment
import com.example.arteka_crohn.R

class ProfileDialogFragment : DialogFragment() {
    companion object {
        private const val ARG_NAME = "arg_name"
        private const val ARG_EMAIL = "arg_email"

        fun newInstance(name: String, email: String): ProfileDialogFragment {
            val fragment = ProfileDialogFragment()
            val args = Bundle()
            args.putString(ARG_NAME, name)
            args.putString(ARG_EMAIL, email)
            fragment.arguments = args
            return fragment
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.profile_dialog, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val name = arguments?.getString(ARG_NAME) ?: ""
        val email = arguments?.getString(ARG_EMAIL) ?: ""

        view.findViewById<TextView>(R.id.tvProfileName).text = "Nom : $name"
        view.findViewById<TextView>(R.id.tvProfileEmail).text = "Email : $email"

        view.findViewById<Button>(R.id.btnLogout).setOnClickListener {
            // Callback de déconnexion, à adapter selon ta logique
            (activity as? LogoutListener)?.onLogoutRequested()
            dismiss()
        }
    }

    interface LogoutListener {
        fun onLogoutRequested()
    }

    override fun onStart() {
        super.onStart()
        dialog?.window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
    }
}
