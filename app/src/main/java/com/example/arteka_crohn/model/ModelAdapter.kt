package com.example.arteka_crohn.model

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.Context
import android.content.SharedPreferences
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.OvershootInterpolator
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.card.MaterialCardView
import androidx.recyclerview.widget.RecyclerView
import com.example.arteka_crohn.R

class ModelAdapter(
    private val context: Context,
    private val models: List<String>,
    private val onModelSelected: (String) -> Unit
) : RecyclerView.Adapter<ModelAdapter.ModelViewHolder>() {
    private val prefs: SharedPreferences = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_model, parent, false)
        return ModelViewHolder(view)
    }

    override fun onBindViewHolder(holder: ModelViewHolder, position: Int) {
        val model = models[position]
        holder.bind(model, isSelected(model))
        holder.itemView.setOnClickListener {
            prefs.edit().putString("SELECTED_MODEL", model).apply()
            notifyDataSetChanged()
            onModelSelected(model)
        }
    }

    override fun getItemCount(): Int = models.size

    private fun isSelected(model: String): Boolean {
        val selected = prefs.getString("SELECTED_MODEL", null)
        return selected == model
    }

    class ModelViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val cardView = itemView as MaterialCardView
        private val nameView: TextView = itemView.findViewById(R.id.tvModelName)
        private val checkMarkView: ImageView = itemView.findViewById(R.id.ivCheckMark)
        //private val descView: TextView = itemView.findViewById(R.id.tvModelDescription)
        
        fun bind(model: String, selected: Boolean) {
            nameView.text = model
            //descView.text = "" // Ajoute une description si tu veux
            if (selected) {
                // cardView.setCardBackgroundColor(itemView.context.getColor(R.color.selected_model))
                nameView.setTextColor(itemView.context.getColor(R.color.overlay_green))
                showCheckMarkWithAnimation()
            } else {
                //cardView.setCardBackgroundColor(itemView.context.getColor(R.color.overlay_gray))
                nameView.setTextColor(ContextCompat.getColor(itemView.context, R.color.text))
                checkMarkView.visibility = View.GONE
            }
        }
        
        private fun showCheckMarkWithAnimation() {
            // Réinitialiser l'état de la vue avant l'animation
            checkMarkView.visibility = View.VISIBLE
            checkMarkView.alpha = 0f
            checkMarkView.scaleX = 0.1f
            checkMarkView.scaleY = 0.1f
            checkMarkView.rotation = 0f
            
            // Créer une animation de rotation
            val rotateAnimation = ObjectAnimator.ofFloat(checkMarkView, "rotation", 0f, 360f)
            rotateAnimation.duration = 500
            
            // Créer une animation de mise à l'échelle
            val scaleXAnimation = ObjectAnimator.ofFloat(checkMarkView, "scaleX", 0.1f, 1.2f, 1.0f)
            val scaleYAnimation = ObjectAnimator.ofFloat(checkMarkView, "scaleY", 0.1f, 1.2f, 1.0f)
            scaleXAnimation.duration = 500
            scaleYAnimation.duration = 500
            
            // Animation d'apparition progressive
            val fadeAnimation = ObjectAnimator.ofFloat(checkMarkView, "alpha", 0f, 1f)
            fadeAnimation.duration = 400
            
            // Ajouter un effet de rebond à la fin pour le "saut"
            val bounceInterpolator = OvershootInterpolator(2.0f)
            scaleXAnimation.interpolator = bounceInterpolator
            scaleYAnimation.interpolator = bounceInterpolator
            
            // Combiner toutes les animations ensemble
            val animSet = AnimatorSet()
            animSet.playTogether(rotateAnimation, scaleXAnimation, scaleYAnimation, fadeAnimation)
            
            // Démarrer l'animation
            animSet.start()
        }
    }
}
