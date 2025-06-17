package com.example.arteka_crohn

import android.os.Bundle
import android.content.Intent
import android.widget.TextView
import android.view.View
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.arteka_crohn.R
import android.widget.ImageView


class ModelSelectionActivity : AppCompatActivity() {
    private lateinit var recyclerView: RecyclerView
    private lateinit var scrollIndicatorUp: ImageView
    private lateinit var scrollIndicatorDown: ImageView
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_selection)

        // Initialiser les vues
        scrollIndicatorUp = findViewById(R.id.scroll_indicator_up)
        scrollIndicatorDown = findViewById(R.id.scroll_indicator_down)
        
        // Affichage des modèles dans une liste de CardView (RecyclerView)
        recyclerView = findViewById(R.id.model_recycler_view)
        val models = ModelManager.listModels(this)
        val layoutManager = LinearLayoutManager(this)
        recyclerView.layoutManager = layoutManager
        val adapter = ModelAdapter(this, models) { selectedModel ->
            //showModernToast(this, "Modèle sélectionné : $selectedModel")
        }
        recyclerView.adapter = adapter
        recyclerView.setHasFixedSize(true)
        
        // Configuration des indicateurs de défilement
        setupScrollIndicators(layoutManager)

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
    
    private fun setupScrollIndicators(layoutManager: LinearLayoutManager) {
        // Vérifier initialement si des indicateurs doivent être affichés
        checkScrollableStatus(layoutManager)
        
        // Ajouter un listener de défilement pour mettre à jour les indicateurs
        recyclerView.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                super.onScrolled(recyclerView, dx, dy)
                checkScrollableStatus(layoutManager)
            }
        })
        
        // Configurer les clics sur les indicateurs pour faire défiler automatiquement
        scrollIndicatorUp.setOnClickListener {
            val targetPosition = maxOf(0, layoutManager.findFirstVisibleItemPosition() - 3)
            recyclerView.smoothScrollToPosition(targetPosition)
        }
        
        scrollIndicatorDown.setOnClickListener {
            val lastPosition = recyclerView.adapter?.itemCount?.minus(1) ?: 0
            val targetPosition = minOf(lastPosition, layoutManager.findLastVisibleItemPosition() + 3)
            recyclerView.smoothScrollToPosition(targetPosition)
        }
    }
    
    private fun checkScrollableStatus(layoutManager: LinearLayoutManager) {
        val itemCount = recyclerView.adapter?.itemCount ?: 0
        
        // Vérifier si on peut défiler vers le haut
        val canScrollUp = layoutManager.findFirstVisibleItemPosition() > 0
        scrollIndicatorUp.visibility = if (canScrollUp) View.VISIBLE else View.GONE
        
        // Vérifier si on peut défiler vers le bas
        val canScrollDown = layoutManager.findLastCompletelyVisibleItemPosition() < itemCount - 1
        scrollIndicatorDown.visibility = if (canScrollDown) View.VISIBLE else View.GONE
    }
}
