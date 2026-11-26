# Suivi des rendez-vous et risque d'absence

Projet du module **Innovations technologiques en IA** (L3 Ingénierie Informatique, UASZ).

## Problème
Dans de nombreux centres de santé, le suivi des patients repose encore sur des dossiers papier. Le centre ne sait pas à l'avance quels patients risquent de manquer leur prochain rendez-vous, et ne peut donc pas les relancer à temps.

**Question :** parmi les patients suivis, lesquels risquent de manquer leur prochain rendez-vous ?

## Solution
Un modèle de Machine Learning (classification) estime un niveau de risque d'absence (faible, moyen, élevé) pour chaque rendez-vous. Une application web permet de :

1. enregistrer les patients ;
2. gérer les rendez-vous ;
3. voir le risque d'absence de chaque rendez-vous ;
4. préparer les rappels, en relançant en priorité les patients à risque ;
5. interroger un **assistant IA** (« Quels patients relancer en priorité ? »).

## Assistant IA
L'assistant (menu « Assistant IA ») aide le personnel en interrogeant les prédictions du modèle et les données enregistrées. Il fonctionne par outils : il comprend la question, appelle la bonne fonction de consultation (patients à relancer, risque et historique d'un patient, explication d'un score, chiffres de l'application) ou répond aux questions sur le fonctionnement du modèle, des données et des limites.

Garde-fous :
- il n'invente aucun résultat : chaque réponse vient directement du score du modèle et des fichiers CSV ;
- il ne donne aucun avis médical : les questions de diagnostic ou de traitement sont refusées ;
- il ne décide pas à la place du personnel : il propose une priorité de relance.

## Contenu du dépôt
| Fichier | Rôle |
|---|---|
| `projet.ipynb` | Notebook : nettoyage des données, comparaison de 3 algorithmes, évaluation, niveaux de risque, export du modèle |
| `app.py` | Application web (Streamlit) |
| `modele_absences.json` | Modèle exporté (régression logistique) utilisé par l'application |
| `requirements.txt` | Bibliothèques nécessaires à l'application |

## Données
Le modèle est entraîné sur le jeu de données **Medical Appointment No Shows** (Joni Hoppen, Kaggle, licence CC BY-NC-SA 4.0) : 110 527 rendez-vous médicaux anonymisés au Brésil. Le fichier n'est pas inclus dans ce dépôt ; il se télécharge sur Kaggle (`joniarroba/noshowappointments`).

L'application ne contient que des **patients fictifs** : ne jamais y saisir de vraies données de santé.

## Lancer l'application
```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Limites
- Données du Brésil (2016), qui ne reflètent pas forcément la réalité des centres de santé sénégalais.
- Le modèle reste modeste : une règle simple sur le délai fait presque aussi bien (F1 de 0,43 contre 0,41 à 0,45 pour les modèles).
- Le niveau de risque est une aide à la décision : l'agent du centre garde toujours la décision finale.

## Activer le modèle de langage (optionnel)
Par défaut, l'assistant comprend les questions grâce à des règles. Pour qu'il comprenne des phrases libres, on peut lui brancher un modèle de langage (Gemini, clé gratuite sur Google AI Studio) : il choisit seulement l'outil à appeler, et la réponse vient toujours des données et du modèle de prédiction. Ajouter dans les *Secrets* de Streamlit (ou dans `.streamlit/secrets.toml` en local, fichier à ne jamais déposer sur GitHub) :
```
GEMINI_API_KEY = "votre_clé"
```
Sans clé, ou en cas de panne, l'assistant revient automatiquement aux règles.
