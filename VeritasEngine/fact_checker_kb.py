import os
import wikipediaapi
from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Charger les variables du fichier .env
load_dotenv()

# Récupérer le User-Agent ou mettre une valeur par défaut de secours
user_agent_env = os.getenv("WIKIPEDIA_USER_AGENT", "VeritasEngine/1.0 (contact@default.com)")

# Configuration de l'API Wikipédia sécurisée
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent=user_agent_env,
    language="fr"
)

# Wrapper de recherche DuckDuckGo
search_engine = DuckDuckGoSearchAPIWrapper(max_results=2)

def fetch_live_news(query: str) -> str:
    """Recherche les dernières actualités officielles sur le web."""
    try:
        results = search_engine.run(f"{query} actualité officielle")
        return results if results else "Aucune dépêche officielle récente trouvée."
    except Exception as e:
        return f"Impossible de joindre les flux d'actualité : {str(e)}"

def fetch_historical_context(entity: str) -> str:
    """Interroge Wikipédia pour valider les faits historiques liés à une entité."""
    page = wiki_wiki.page(entity)
    if page.exists():
        # On récupère les 300 premiers mots pour le résumé historique
        return page.summary[:1500] + "..."
    return f"Aucun antécédent historique majeur trouvé sur Wikipédia pour '{entity}'."