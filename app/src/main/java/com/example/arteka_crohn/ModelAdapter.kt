package com.example.arteka_crohn

import android.content.Context
import android.content.SharedPreferences
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
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
        private val cardView = itemView as com.google.android.material.card.MaterialCardView
        private val nameView: TextView = itemView.findViewById(R.id.tvModelName)
        //private val descView: TextView = itemView.findViewById(R.id.tvModelDescription)
        fun bind(model: String, selected: Boolean) {
            nameView.text = model
            //descView.text = "" // Ajoute une description si tu veux
            if (selected) {
                cardView.setCardBackgroundColor(itemView.context.getColor(R.color.colorSecondary))
                nameView.setTextColor(itemView.context.getColor(android.R.color.white))
            } else {
                cardView.setCardBackgroundColor(itemView.context.getColor(R.color.overlay_gray))
                nameView.setTextColor(androidx.core.content.ContextCompat.getColor(itemView.context, R.color.text))
            }
        }
    }
}
