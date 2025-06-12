package com.example.arteka_crohn

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.firebase.auth.FirebaseAuth
import com.example.arteka_crohn.segmentation.InstanceSegmentation

class ProfileActivity : AppCompatActivity() {
    private var instanceSegmentation: InstanceSegmentation? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_profile)

        val user = FirebaseAuth.getInstance().currentUser
        val email = user?.email ?: "Email inconnu"

        findViewById<TextView>(R.id.tvProfileEmail).text = email

        findViewById<Button>(R.id.btnLogout).setOnClickListener {
            FirebaseAuth.getInstance().signOut()
            val intent = Intent(this, LoginActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            startActivity(intent)
        }

        findViewById<TextView>(R.id.tvDeleteAccount).setOnClickListener {
            val bottomSheet = com.google.android.material.bottomsheet.BottomSheetDialog(this)
            val view = layoutInflater.inflate(R.layout.bottomsheet_delete_account, null)
            bottomSheet.setContentView(view)

            view.findViewById<Button>(R.id.btnConfirmDelete).setOnClickListener {
                user?.delete()?.addOnCompleteListener { task ->
                    if (task.isSuccessful) {
                        val intent = Intent(this, LoginActivity::class.java)
                        intent.flags =
                                Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                        startActivity(intent)
                        bottomSheet.dismiss()
                    } else {
                        Toast.makeText(
                                        this,
                                        "Erreur lors de la suppression du compte. Veuillez vous reconnecter et réessayer.",
                                        Toast.LENGTH_LONG
                                )
                                .show()
                    }
                }
            }
            view.findViewById<Button>(R.id.btnCancel).setOnClickListener { bottomSheet.dismiss() }
            bottomSheet.show()
        }

        val bottomNavigation =
                findViewById<com.google.android.material.bottomnavigation.BottomNavigationView>(
                        R.id.bottom_navigation
                )
        bottomNavigation.selectedItemId = R.id.action_profile
        bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.action_profile -> true
                R.id.action_models -> {
                    cleanupModel()
                    startActivity(Intent(this, ModelSelectionActivity::class.java))

                    // Utilisation de overridePendingTransition pour la compatibilité avec toutes
                    // les versions.
                    // Cette méthode est dépréciée mais il n’existe pas d’alternative officielle à
                    // ce jour.
                    overridePendingTransition(R.anim.slide_in_left, R.anim.slide_out_right)
                    finish()
                    true
                }
                R.id.action_camera -> {
                    cleanupModel()
                    startActivity(Intent(this, MainActivity::class.java))

                    // Utilisation de overridePendingTransition pour la compatibilité avec toutes
                    // les versions.
                    // Cette méthode est dépréciée mais il n’existe pas d’alternative officielle à
                    // ce jour.
                    overridePendingTransition(R.anim.slide_in_left, R.anim.slide_out_right)
                    finish()
                    true
                }
                else -> false
            }
        }
    }

    private fun cleanupModel() {
        instanceSegmentation?.close()
    }
}
