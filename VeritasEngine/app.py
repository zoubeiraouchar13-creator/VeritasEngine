import streamlit as st
import os
from agents_lc import VeritasEngineOrchestrator

# Configuration de la page avec un style professionnel élargi
st.set_page_config(page_title="VeritasEngine | Plateforme d'Audit d'Information", layout="wide")

# Injection CSS pour le titre effet néon technologique et épuré
st.markdown(
    """
    <style>
    .neon-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 5px #00d2ff, 0 0 10px #00d2ff, 0 0 20px #00d2ff;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #a1a1aa;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# En-tête de l'application
st.markdown('<div class="neon-title">VeritasEngine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Système d\'analyse croisée par Deep Learning, Réseaux Multi-Agents et Contextualisation Factuelle</div>', unsafe_allow_html=True)
st.write("---")

# Organisation des sections en deux colonnes équilibrées
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Entrées du Système")
    
    # On enlève la valeur par défaut pour laisser l'utilisateur libre
    input_text = st.text_area(
        "Contenu textuel à auditer (Article, dépêche ou affirmation) :", 
        height=220, 
        value=""
    )
    
    uploaded_file = st.file_uploader(
        "Fichier image associé pour analyse multimodale (Optionnel)", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Bouton d'action principal
    analyze_button = st.button("Lancer l'audit analytique", type="primary", use_container_width=True)

with col2:
    st.subheader("Tableau de Bord de Crédibilité")
    
    # MODIFICATION DE LA CONDITION : On s'assure qu'au moins une des deux entrées est présente
    if analyze_button:
        if not input_text and not uploaded_file:
            st.error("Veuillez fournir au moins une entrée (un texte ou une image) pour lancer l'analyse.")
        else:
            with st.spinner("Orchestration des agents spécialisés et calculs d'inférence..."):
                
                # Gestion de l'infrastructure de fichiers temporaires
                temp_img_path = None
                if uploaded_file:
                    os.makedirs("data", exist_ok=True)
                    temp_img_path = os.path.join("data", uploaded_file.name)
                    with open(temp_img_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Exécution du pipeline complet
                results = VeritasEngineOrchestrator.run_pipeline(input_text, temp_img_path)
                
                # Distribution et affichage des métriques de la Partie A (Deep Learning)
                m1, m2 = st.columns(2)
                m1.metric(label="Indice de manipulation textuelle (Bi-LSTM)", value=f"{results['text_score']:.2%}")
                m2.metric(label="Indice d'anomalie d'image (CNN)", value=f"{results['vision_score']:.2%}")
                
                st.write("---")
                
                # Affichage conditionnel de la partie sémantique / textuelle
                if input_text:
                    st.markdown("**Vérification Contextuelle et Historique**")
                    if results.get('contradictions'):
                        for conflict in results['contradictions']:
                            st.warning(conflict)
                    else:
                        st.info("Aucune anomalie flagrante relevée par les moteurs de recherche contextuels.")
                    st.write("---")
                    
                # Rapport d'explication finale structuré par l'IA Générative (Partie B)
                st.markdown("**Rapport d'Explication Générative**")
                st.markdown(''' Score > 0.70: Forte probabilité d'image générée par IA ou manipulée.\n
                                Score 0.70 >Score >= 0.45: Anomalies visuelles détectées (Zone d'incertitude).\n
                                Score < 0.45 :  L'image présente des caractéristiques de capture réelle."
                            ''')
                st.markdown(results['explanation'])
                
                # Nettoyage sécurisé du fichier
                if temp_img_path and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
    else:
        st.caption("En attente de soumission de données dans le panneau de gauche.")