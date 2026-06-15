# agents_lc.py (Version Affinée)
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from models_dl import inference_text, inference_image
from fact_checker_kb import fetch_live_news, fetch_historical_context

llm = Ollama(model="llama3.2", temperature=0.0) # Température à 0 pour un comportement déterministe

class VeritasEngineOrchestrator:
    @staticmethod
    def run_pipeline(text: str, image_path: str = None) -> dict:
        # Si aucun texte n'est fourni, on initialise une chaîne vide
        text = text if text else ""
        
        # 1. Analyse Deep Learning (Partie A)
        text_score = inference_text(text) if text else 0.0
        vision_score = inference_image(image_path) if image_path else 0.0
        
        # Détermination du statut de l'image (Seuils calibrés)
        if vision_score > 0.70:
            vision_status = "ALERTE : Forte probabilité d'image générée par IA ou manipulée."
        elif vision_score > 0.45:
            vision_status = "ATTENTION : Anomalies visuelles détectées (Zone d'incertitude)."
        else:
            vision_status = "RAS : L'image présente des caractéristiques de capture réelle."

        # Initialisation des variables par défaut
        current_news_context = "Aucun texte fourni pour l'analyse d'actualité."
        historical_context = "Aucun texte fourni pour l'analyse historique."
        history_query = "N/A"
        contradictions = []

        # Exécution des briques de recherche UNIQUEMENT si un texte existe
        if text:
            # Extraction via LangChain
            extraction_prompt = ChatPromptTemplate.from_template(
                "Analyse le texte et isole UNIQUEMENT le nom de la personne principale ou de l'événement historique précis.\n"
                "Texte : {text}"
            )
            extraction_chain = extraction_prompt | llm
            history_query = extraction_chain.invoke({"text": text}).strip()

            live_query = f"{text[:60]} actualité officielle récentes 2026"
            current_news_context = fetch_live_news(live_query)
            historical_context = fetch_historical_context(history_query)
            contradictions = [f"Recherche Actualité : {live_query}", f"Entité Wikipédia calibrée : {history_query}"]
        else:
            # Si image seule, on demande au LLM de faire une analyse basée UNIQUEMENT sur le constat du CNN
            contradictions = ["Mode Image Seule : Analyse factuelle textuelle désactivée."]

        # Prompt final adapté aux deux situations
        synthesis_prompt = ChatPromptTemplate.from_template(
            "Tu es 'VeritasEngine'. Rédige un rapport d'audit.\n\n"
            "📊 ANALYSE COGNITIVE (DEEP LEARNING) :\n"
            "- Score de suspicion du texte : {t_score}\n"
            "- Diagnostic de l'image ({v_score}) : {v_status}\n\n"
            "🌐 CONTEXTUALISATION RECHERCHÉE :\n"
            "- Actualités chaudes : {news}\n"
            "- Vérification historique (Wikipédia) : {history}\n\n"
            "Rédige en français un verdict technique expliquant les scores et une conclusion sur la crédibilité des éléments fournis."
        )
        
        synthesis_chain = synthesis_prompt | llm
        explanation = synthesis_chain.invoke({
            "t_score": f"{text_score:.2%}",
            "v_score": f"{vision_score:.2%}",
            "v_status": vision_status,
            "news": current_news_context,
            "history": historical_context
        })

        return {
            "text_score": text_score,
            "vision_score": vision_score,
            "contradictions": contradictions,
            "explanation": explanation
        }