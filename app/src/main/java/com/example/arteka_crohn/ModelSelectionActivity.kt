package com.example.arteka_crohn

import android.os.Bundle
import android.content.Intent
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.arteka_crohn.R


class ModelSelectionActivity : AppCompatActivity() {
    private var instanceSegmentation: InstanceSegmentation? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_selection)

        val bottomNavigation = findViewById<com.google.android.material.bottomnavigation.BottomNavigationView>(R.id.bottom_navigation)
        bottomNavigation.selectedItemId = R.id.action_models
        bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.action_models -> true
                R.id.action_profile -> {
    cleanupModel()
    startActivity(Intent(this, ProfileActivity::class.java))

    // Utilisation de overridePendingTransition pour la compatibilité avec toutes les versions.
    // Cette méthode est dépréciée mais il n’existe pas d’alternative officielle à ce jour.
    overridePendingTransition(R.anim.slide_in_right, R.anim.slide_out_left)
    finish()
    true
}
                R.id.action_camera -> {
    cleanupModel()
    startActivity(Intent(this, MainActivity::class.java))
    
    // Utilisation de overridePendingTransition pour la compatibilité avec toutes les versions.
    // Cette méthode est dépréciée mais il n’existe pas d’alternative officielle à ce jour.
    overridePendingTransition(R.anim.slide_in_right, R.anim.slide_out_left)
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
