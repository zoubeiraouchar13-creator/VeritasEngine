# VeritasEngine

Système multimodal de détection de fausses informations et d'images générées par IA, combinant Deep Learning, architecture multi-agents et génération d'explications par LLM.

---

## Description

VeritasEngine est une plateforme d'audit informationnel développée dans le cadre du module IA Avancée et Applications (Master 1 SDIA). Elle permet d'analyser un texte, une image, ou les deux simultanément, et produit un rapport de crédibilité structuré.

Le système repose sur trois couches complémentaires :

- Un modèle Bi-LSTM pour scorer la probabilité de manipulation textuelle
- Un CNN basé sur ResNet-18 (transfer learning) pour détecter les images manipulées
  ou générées par IA
- Une architecture multi-agents orchestrée via LangChain, couplant recherche web
  en temps réel (DuckDuckGo), vérification historique (Wikipedia) et génération
  d'un rapport explicatif par Llama 3.2 (Ollama)

---

## Architecture:  

Entrée utilisateur (texte / image)  
|  
+---> Bi-LSTM (score suspicion textuelle)  
|  
+---> CNN ResNet-18 (score anomalie image)  
|  
v  
Orchestrateur LangChain (VeritasEngineOrchestrator)
|  
+---> Agent DuckDuckGo  (actualités récentes)  
|  
+---> Agent Wikipedia   (contexte historique)  
|  
v  
LLM Llama 3.2 via Ollama (rapport d'audit)  
|  
v  
Dashboard Streamlit (scores + verdict + explication)  
---

## Structure du projet  

.  
├── app.py                # Interface Streamlit et gestion du pipeline  
├── agents_lc.py          # Orchestrateur LangChain et agents multi-tâches  
├── models_dl.py          # Architectures PyTorch (Bi-LSTM, CNN) et inférences  
├── fact_checker_kb.py    # Modules de recherche DuckDuckGo et Wikipedia  
├── .env                  # Variables d'environnement (non versionné)  
└── data/                 # Dossier temporaire pour les images uploadées  
---

## Prérequis

- Python 3.10 ou supérieur
- [Ollama](https://ollama.com) installé et en cours d'exécution en local
- Le modèle Llama 3.2 téléchargé via Ollama :

```bash
ollama pull llama3.2
```

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/veritasengine.git
cd veritasengine

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Configuration

Créer un fichier `.env` à la racine du projet :
WIKIPEDIA_USER_AGENT=VeritasEngine/1.0 (votre-email@exemple.com)
---

## Lancement

```bash
streamlit run app.py
```

L'interface est accessible à l'adresse `http://localhost:8501`.

---

## Dépendances principales

| Bibliothèque            | Usage                                              |
|-------------------------|----------------------------------------------------|
| torch / torchvision     | Architectures Bi-LSTM et CNN, inférence            |
| transformers            | Tokenizer BERT (bert-base-uncased)                 |
| langchain-community     | Orchestration multi-agents, DuckDuckGo             |
| ollama                  | Interface locale avec Llama 3.2                    |
| streamlit               | Interface utilisateur web                          |
| wikipedia-api           | Interrogation de Wikipedia en français             |
| pillow                  | Chargement et transformation des images            |
| python-dotenv           | Gestion des variables d'environnement              |

Générer le fichier `requirements.txt` :

```bash
pip freeze > requirements.txt
```

---

## Fonctionnement

L'interface propose deux zones de saisie : un champ texte pour soumettre un article ou une affirmation, et un uploader d'image. Au moins une des deux entrées est requise.

Une fois l'audit lancé, le système affiche :

- L'indice de manipulation textuelle (Bi-LSTM), exprimé en pourcentage
- L'indice d'anomalie d'image (CNN), avec interprétation selon trois seuils :
  - Score > 0.70 : forte probabilité d'image générée par IA ou manipulée
  - Score entre 0.45 et 0.70 : anomalies visuelles détectées, zone d'incertitude
  - Score < 0.45 : image présentant des caractéristiques de capture réelle
- Le contexte factuel récupéré par les agents de recherche
- Le rapport d'audit rédigé par le LLM

---

## Limites connues

Les modèles Deep Learning sont initialisés avec des poids aléatoires dans cette version. Pour obtenir des performances métriques exploitables, un entraînement supervisé sur des jeux de données dédiés est nécessaire :

- Texte : LIAR dataset, FakeNewsNet
- Images : CIFAKE
