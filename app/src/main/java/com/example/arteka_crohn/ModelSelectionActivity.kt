package com.example.arteka_crohn

import android.os.Bundle
import android.content.Intent
import android.widget.TextView
import android.view.View
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import com.example.arteka_crohn.R


class ModelSelectionActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_selection)

        // Affichage des modèles dans une liste de CardView (RecyclerView)
        val recyclerView = findViewById<androidx.recyclerview.widget.RecyclerView>(R.id.model_recycler_view)
        val models = ModelManager.listModels(this)
        recyclerView.layoutManager = androidx.recyclerview.widget.LinearLayoutManager(this)
        val adapter = ModelAdapter(this, models) { selectedModel ->
            showModernToast(this, "Modèle sélectionné : $selectedModel")
        }
        recyclerView.adapter = adapter
        recyclerView.setHasFixedSize(true)

        val bottomNavigation = findViewById<com.google.android.material.bottomnavigation.BottomNavigationView>(R.id.bottom_navigation)
        bottomNavigation.selectedItemId = R.id.action_models
        bottomNavigation.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.action_models -> true
                R.id.action_profile -> {
                    startActivity(Intent(this, ProfileActivity::class.java))
                    overridePendingTransition(R.anim.slide_in_right, R.anim.slide_out_left)
                    finish()
                    true
                }
                R.id.action_camera -> {
                    startActivity(Intent(this, MainActivity::class.java))
                    overridePendingTransition(R.anim.slide_in_right, R.anim.slide_out_left)
                    finish()
                    true
                }
                else -> false
            }
        }
    }
}

