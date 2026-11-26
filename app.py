import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Suivi des rendez-vous", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------
# Fichiers de données (CSV) et constantes
# ---------------------------------------------------------------
DOSSIER = Path("data")
DOSSIER.mkdir(exist_ok=True)
F_PATIENTS = DOSSIER / "patients.csv"
F_RDV = DOSSIER / "rendezvous.csv"

COLS_PATIENTS = ["id", "nom", "date_naissance", "sexe", "telephone",
                 "aide_sociale", "hypertension", "diabete", "alcoolisme", "handicap"]
COLS_RDV = ["id_rdv", "id_patient", "date_prise", "date_rdv",
            "type_consultation", "statut", "rappel_envoye"]
TYPES = ["Consultation générale", "Vaccination",
         "Suivi maladie chronique", "Consultation prénatale"]

# taux d'absence observés dans le notebook pour chaque niveau (jeu de test)
TAUX_OBSERVE = {"Faible": 10, "Moyen": 22, "Élevé": 35}


# ---------------------------------------------------------------
# Modèle : régression logistique exportée depuis le notebook
# probabilité = 1 / (1 + exp(-(intercept + somme(coef * valeur))))
# ---------------------------------------------------------------
@st.cache_resource
def charger_modele():
    with open("modele_absences.json", encoding="utf-8") as f:
        return json.load(f)


def score_absence(valeurs: dict) -> float:
    m = charger_modele()
    z = m["intercept"] + sum(c * valeurs[nom] for nom, c in zip(m["colonnes"], m["coef"]))
    return 1 / (1 + math.exp(-z))


def niveau_risque(score: float) -> str:
    if score < 0.4:
        return "Faible"
    if score < 0.6:
        return "Moyen"
    return "Élevé"




# ---------------------------------------------------------------
# Lecture / écriture des CSV
# ---------------------------------------------------------------
def donnees_demo():
    """Crée des données FICTIVES pour la démonstration."""
    aujourdhui = date.today()
    alea = random.Random(7)
    noms = ["Awa D.", "Moussa S.", "Fatou N.", "Ibrahima S.", "Mariama D.", "Cheikh F.",
            "Aminata B.", "Ousmane S.", "Khady G.", "Abdou M.", "Rokhaya C.", "Modou S."]
    pats = []
    for i, nom in enumerate(noms, start=1):
        pats.append({
            "id": f"P{i:03d}", "nom": nom,
            "date_naissance": str(date(alea.randint(1950, 2015), alea.randint(1, 12), alea.randint(1, 28))),
            "sexe": "Femme" if nom.split()[0] in ("Awa", "Fatou", "Mariama", "Aminata", "Khady", "Rokhaya") else "Homme",
            "telephone": f"77 000 00 {i:02d}",
            "aide_sociale": alea.choice(["Non", "Non", "Oui"]),
            "hypertension": alea.choice(["Non", "Non", "Oui"]),
            "diabete": alea.choice(["Non", "Non", "Non", "Oui"]),
            "alcoolisme": "Non", "handicap": "Non",
        })
    rdvs = []
    n = 1
    # patients qui ont déjà manqué plusieurs rendez-vous
    fragiles = {"P003": 0.6, "P009": 0.5}
    for p in pats:
        for _ in range(alea.randint(2, 4)):
            jour = aujourdhui - timedelta(days=alea.randint(20, 300))
            absent = alea.random() < fragiles.get(p["id"], 0.12)
            rdvs.append({"id_rdv": f"R{n:03d}", "id_patient": p["id"],
                         "date_prise": str(jour - timedelta(days=alea.randint(0, 30))),
                         "date_rdv": str(jour), "type_consultation": alea.choice(TYPES),
                         "statut": "Absent" if absent else "Présent", "rappel_envoye": "Non"})
            n += 1
    # rendez-vous à venir
    for p in pats[:10]:
        jour = aujourdhui + timedelta(days=alea.randint(1, 20))
        delai = alea.choice([0, 0, 1, 3, 5, 7, 10, 15, 25])
        prise = min(aujourdhui, jour - timedelta(days=delai))
        rdvs.append({"id_rdv": f"R{n:03d}", "id_patient": p["id"], "date_prise": str(prise),
                     "date_rdv": str(jour), "type_consultation": alea.choice(TYPES),
                     "statut": "À venir", "rappel_envoye": "Non"})
        n += 1
    pd.DataFrame(pats, columns=COLS_PATIENTS).to_csv(F_PATIENTS, index=False)
    pd.DataFrame(rdvs, columns=COLS_RDV).to_csv(F_RDV, index=False)


def lire(fichier, colonnes):
    if not fichier.exists():
        donnees_demo()
    return pd.read_csv(fichier, dtype=str).fillna("")[colonnes]


def sauver(df, fichier):
    df.to_csv(fichier, index=False)


def en_date(texte):
    return pd.to_datetime(texte).date()


def age_a(naissance: date, jour: date) -> int:
    return jour.year - naissance.year - ((jour.month, jour.day) < (naissance.month, naissance.day))


def prochain_id(df, colonne, prefixe):
    nums = [int(x[1:]) for x in df[colonne] if x[1:].isdigit()]
    return f"{prefixe}{(max(nums) if nums else 0) + 1:03d}"


# ---------------------------------------------------------------
# Calcul du risque d'un rendez-vous
# ---------------------------------------------------------------
def valeurs_risque(patient, rdv, tous_rdv):
    jour = en_date(rdv["date_rdv"])
    deja_vus = tous_rdv[(tous_rdv["id_patient"] == rdv["id_patient"]) & (tous_rdv["statut"] == "Absent")]
    absences = sum(en_date(d) < jour for d in deja_vus["date_rdv"])
    return {
        "Age": min(max(age_a(en_date(patient["date_naissance"]), jour), 0), 100),
        "delai": max((jour - en_date(rdv["date_prise"])).days, 0),
        "Scholarship": int(patient["aide_sociale"] == "Oui"),
        "Hipertension": int(patient["hypertension"] == "Oui"),
        "Diabetes": int(patient["diabete"] == "Oui"),
        "Alcoholism": int(patient["alcoolisme"] == "Oui"),
        "handicap": int(patient["handicap"] == "Oui"),
        "genre": int(patient["sexe"] == "Femme"),
        "absences_passees": absences,
    }


def risque_rdv(patient, rdv, tous_rdv):
    valeurs = valeurs_risque(patient, rdv, tous_rdv)
    score = score_absence(valeurs)
    return score, niveau_risque(score), valeurs["delai"], valeurs["absences_passees"]


def rdv_a_venir_avec_risque():
    patients = lire(F_PATIENTS, COLS_PATIENTS)
    rdvs = lire(F_RDV, COLS_RDV)
    lignes = []
    for _, r in rdvs[rdvs["statut"] == "À venir"].iterrows():
        p = patients[patients["id"] == r["id_patient"]]
        if p.empty:
            continue
        p = p.iloc[0]
        score, niveau, delai, absences = risque_rdv(p, r, rdvs)
        lignes.append({"id_rdv": r["id_rdv"], "patient": p["nom"], "telephone": p["telephone"],
                       "date_rdv": en_date(r["date_rdv"]), "type": r["type_consultation"],
                       "delai (jours)": delai, "absences passées": absences,
                       "score": score, "niveau": niveau, "rappel_envoye": r["rappel_envoye"]})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------
# Couleurs et style de l'interface
# ---------------------------------------------------------------
COULEURS = {"Faible": "#16A34A", "Moyen": "#F59E0B", "Élevé": "#DC2626"}
FONDS = {"Faible": "#DCFCE7", "Moyen": "#FEF3C7", "Élevé": "#FEE2E2"}
TEXTES = {"Faible": "#166534", "Moyen": "#92400E", "Élevé": "#991B1B"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {font-family: 'Inter', sans-serif;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding: 1.2rem 2.4rem 2rem 2.4rem; max-width: 100%;}
h2, h3 {font-weight: 600; letter-spacing: -0.01em;}
h3 {font-size: 1.15rem !important; margin-bottom: .4rem;}
.entete {display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;}
.entete .titre {font-size: 1.6rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;}
.entete .sous-titre {color: #64748B; font-size: .95rem; margin-top: .1rem;}
.puce {background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; padding: .2rem .8rem;
       border-radius: 999px; font-size: .78rem; font-weight: 600;}
.kpi {background: white; border: 1px solid #E2E8F0; border-left: 4px solid #2563EB; border-radius: 12px;
      padding: .9rem 1.1rem; box-shadow: 0 1px 3px rgba(15,23,42,.06);}
.kpi-val {font-size: 1.9rem; font-weight: 700; color: #0F172A; line-height: 1.1;}
.kpi-lib {font-size: .82rem; color: #64748B; margin-top: .2rem;}
.marque {font-size: 1.15rem; font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: .6rem;}
.logo {background: #2563EB; color: white; width: 30px; height: 30px; border-radius: 9px; display: inline-flex;
       align-items: center; justify-content: center; font-weight: 700;}
.pastille {display: inline-block; padding: .15rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 700;}
.discret {color: #64748B; font-size: .88rem;}
[data-testid="stTabs"] button {font-size: .95rem; font-weight: 500; padding: .5rem .9rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
button[kind="primary"], button[kind="primaryFormSubmit"] {border-radius: 9px; font-weight: 600;}
[data-testid="stDataFrame"] {border: 1px solid #E2E8F0; border-radius: 10px; overflow: hidden;}
</style>
""", unsafe_allow_html=True)


def pastille(niveau: str) -> str:
    return (f'<span class="pastille" style="background:{FONDS[niveau]};color:{TEXTES[niveau]}">'
            f'{niveau}</span>')


def colorier(valeur):
    return f"background-color: {FONDS[valeur]}; color: {TEXTES[valeur]}; font-weight: 600"


# ---------------------------------------------------------------
# Assistant IA : il ne fait que CONSULTER les résultats du modèle et les données
# (il ne calcule rien lui-même, n'invente rien, ne donne aucun avis médical)
# ---------------------------------------------------------------
import re
import unicodedata
import time
import urllib.request

MOTS_MEDICAUX = ["diagnostic", "medicament", "traitement", "symptome", "douleur", "dose", "posologie",
                 "ordonnance", "guerir", "soigner", "prescri", "antibio", "grave", "est-ce dangereux",
                 "dois-je prendre", "dois je prendre", "quelle maladie", "est malade", "vaccin a faire"]

SUGGESTIONS = [
    "Quels sont les patients qui nécessitent une relance prioritaire ?",
    "Pourquoi Awa est-elle à risque élevé ?",
    "Quels patients ont déjà manqué un rendez-vous ?",
    "Comment le niveau de risque est-il calculé ?",
]

ETIQUETTES = {"Age": "l'âge", "delai": "le délai de prise du rendez-vous", "Scholarship": "l'aide sociale",
              "Hipertension": "l'hypertension", "Diabetes": "le diabète", "Alcoholism": "l'alcoolisme",
              "handicap": "le handicap", "genre": "le sexe", "absences_passees": "les absences passées"}

# Questions sur l'application : (nom de l'outil, description pour le LLM, mots-clés, réponse)
FAQ = [
    ("faq_sms", "pourquoi les SMS ne sont pas utilisés comme facteur du modèle",
     ["sms"],
     "Le modèle n'utilise pas l'envoi de SMS comme facteur : dans les données, les patients qui recevaient un SMS "
     "manquaient plus souvent leur rendez-vous, mais ce n'est pas le SMS qui en est la cause (on rappelait surtout "
     "les patients déjà jugés à risque). Corrélation n'est pas causalité, donc cette colonne a été retirée. "
     "L'envoi réel de SMS n'est pas inclus dans ce prototype."),
    ("faq_fiabilite", "fiabilité, précision et performance du modèle, confiance dans les prédictions",
     ["fiable", "precis", "performance", "f1", "auc", "exact", "se tromp", "confiance", "efficace"],
     "Le modèle reste modeste : sur les données de test, sa régression logistique obtient une AUC d'environ 0,67 "
     "(0,5 = hasard), et une règle simple sur le délai fait presque aussi bien. Il se trompe donc régulièrement. "
     "C'est pourquoi le niveau de risque sert à **prioriser les relances** et non à affirmer qu'un patient viendra "
     "ou non. Dans les données de test, environ 10 % des patients de niveau faible, 22 % de niveau moyen et 35 % de "
     "niveau élevé ont manqué leur rendez-vous."),
    ("faq_calcul", "comment le niveau de risque / le score est calculé, différence entre faible, moyen et élevé",
     ["calcule", "calcul", "score", "difference", "signifie", "que veut dire", "c'est quoi le niveau",
      "niveau de risque", "seuil"],
     "Le modèle (régression logistique) a appris sur 110 527 rendez-vous réels anonymisés quels facteurs augmentent "
     "le risque d'absence. Pour chaque rendez-vous, l'application additionne ces facteurs pondérés (âge, délai, "
     "absences passées, aide sociale, hypertension, diabète, alcoolisme, handicap, sexe) et obtient un **score** "
     "entre 0 et 1. Score inférieur à 0,4 : risque **faible** ; de 0,4 à 0,6 : **moyen** ; supérieur à 0,6 : **élevé**. "
     "Le score classe les patients, ce n'est pas une probabilité exacte."),
    ("faq_modele", "quel modèle ou algorithme de Machine Learning est utilisé",
     ["modele", "algorithme", "regression", "machine learning", "apprentissage", "intelligence artificielle",
      "arbre", "foret"],
     "L'application utilise un modèle de Machine Learning de classification : une **régression logistique** "
     "entraînée dans le notebook `projet.ipynb`, puis exportée dans `modele_absences.json`. Un arbre de décision "
     "et une forêt aléatoire ont été comparés ; la régression logistique a été gardée car on peut expliquer "
     "chaque score. Je suis un assistant qui consulte ce modèle : je ne fais aucun calcul de mon côté."),
    ("faq_donnees", "d'où viennent les données, jeu de données d'entraînement, données réelles ou fictives",
     ["donnees", "dataset", "kaggle", "bresil", "reelle", "fictive", "entraine", "source", "jeu de"],
     "Le modèle a été entraîné sur le jeu public **Medical Appointment No Shows** (Kaggle, Joni Hoppen, "
     "CC BY-NC-SA 4.0) : 110 527 rendez-vous anonymisés au Brésil, dont environ 20 % d'absences. "
     "Les patients affichés dans l'application sont **fictifs** et servent uniquement à la démonstration."),
    ("faq_ethique", "éthique, confidentialité, vie privée, loi, sécurité des données",
     ["remplace", "medecin", "decision", "ethique", "confidentiel", "vie privee", "loi", "cdp", "anonym", "securite", "securise", "protege"],
     "Les données utilisées pour entraîner le modèle sont anonymisées et les patients de l'application sont fictifs. "
     "Avec de vraies données de santé, il faudrait respecter la loi sénégalaise n° 2008-12 sur les données "
     "personnelles (Commission de protection des données personnelles). L'outil aide à décider, la décision "
     "finale reste humaine, et je ne donne aucun avis médical."),
    ("faq_limites", "limites et points faibles de l'application",
     ["limite", "inconvenient", "defaut", "faiblesse", "ameliorer", "amelioration"],
     "Limites : données du Brésil (2016) qui ne reflètent pas forcément les centres de santé sénégalais ; modèle "
     "modeste ; données de l'application enregistrées dans des fichiers CSV (remises à zéro sur un hébergement "
     "gratuit) ; pas d'envoi réel de SMS. Pour aller plus loin : une vraie base de données et l'envoi de SMS."),
    ("faq_ajouter_patient", "comment enregistrer / ajouter un nouveau patient",
     ["ajouter un patient", "enregistrer un patient", "nouveau patient", "creer un patient", "ajouter patient",
      "inscrire un patient", "ajouter une personne"],
     "Pour enregistrer un patient : menu **Patients**, formulaire « Nouveau patient » à droite (nom, naissance, "
     "sexe, téléphone, aide sociale, suivi particulier), puis **Enregistrer**. Il apparaît tout de suite dans la liste."),
    ("faq_creer_rdv", "comment créer / prendre un rendez-vous",
     ["creer un rendez", "ajouter un rendez", "nouveau rendez", "prendre un rendez", "enregistrer un rendez",
      "planifier", "programmer", "fixer un rendez"],
     "Pour créer un rendez-vous : menu **Rendez-vous**, formulaire « Nouveau rendez-vous » (patient, date, type de "
     "consultation), puis **Créer le rendez-vous**. Son risque d'absence est calculé automatiquement."),
    ("faq_resultat_rdv", "comment enregistrer qu'un patient est venu ou absent",
     ["patient est absent", "patient est present", "enregistrer l'absence", "saisir le resultat", "marquer absent",
      "marquer present", "resultat d'un rendez", "enregistrer qu", "est venu", "n'est pas venu", "signaler une absence"],
     "Après la consultation : menu **Rendez-vous**, encadré « Résultat d'un rendez-vous » : choisissez le rendez-vous, "
     "indiquez **Présent** ou **Absent**, puis **Enregistrer le résultat**. Les absences passées sont ensuite "
     "prises en compte dans les prochaines prédictions."),
    ("faq_envoyer_rappel", "comment envoyer ou marquer un rappel SMS",
     ["envoyer un rappel", "envoyer un sms", "comment rappeler", "marquer le rappel", "envoyer les rappels",
      "comment envoyer", "rappel envoye"],
     "Menu **Rappels** : chaque patient à relancer a une carte avec le message SMS prêt à copier (bouton « Message "
     "SMS »). Après avoir contacté le patient, cliquez sur **Rappel envoyé**. L'envoi réel de SMS n'est pas inclus "
     "dans ce prototype."),
    ("faq_fonctionnement", "présentation de l'application, comment elle fonctionne, mode d'emploi, ce que sait faire l'assistant",
     ["comment fonctionne", "comment ca marche", "comment utiliser", "fonctionnement", "mode d'emploi", "aide",
      "que peux", "que sais", "tu peux faire", "sert a quoi", "a quoi sert", "presente", "explique l'application"],
     "L'application aide un centre de santé à repérer les patients qui risquent de manquer leur rendez-vous.\n\n"
     "- **Patients** : enregistrer et consulter les patients\n"
     "- **Rendez-vous** : créer les rendez-vous et saisir si le patient est venu\n"
     "- **Prédiction** : risque d'absence de chaque rendez-vous à venir (faible, moyen, élevé)\n"
     "- **Rappels** : patients à relancer en priorité, avec le message SMS\n"
     "- **Assistant IA** : moi. Je réponds sur les relances, un patient, les chiffres et le fonctionnement du modèle.\n\n"
     "Exemples : « Qui dois-je relancer ? », « Pourquoi Awa est-elle à risque ? », « Quel est le taux d'absence ? »"),
    ("faq_createur", "qui a créé ou développé l'application",
     ["qui a cree", "qui a fait", "qui a developpe", "createur", "auteur", "qui t'a"],
     "C'est un prototype réalisé pour le module « Innovations technologiques en IA » (L3 Ingénierie Informatique, "
     "UASZ). Le modèle de prédiction a été entraîné dans le notebook du projet."),
    ("faq_merci", "remerciements, salutations, politesse",
     ["merci", "bravo", "parfait", "super", "bonjour", "salut", "bonsoir", "coucou", "au revoir"],
     "Avec plaisir. Posez-moi une autre question quand vous voulez : relances, patients, chiffres ou fonctionnement du modèle."),
]
FAQ_TEXTES = {nom: texte for nom, _, _, texte in FAQ}

# conditions de santé enregistrées : racine du mot -> (colonne, libellé)
CONDITIONS = {"diabet": ("diabete", "un diabète"), "hypertend": ("hypertension", "une hypertension"),
              "hypertension": ("hypertension", "une hypertension"), "alcool": ("alcoolisme", "de l'alcoolisme"),
              "handicap": ("handicap", "un handicap"), "aide sociale": ("aide_sociale", "une aide sociale"),
              "bourse": ("aide_sociale", "une aide sociale"), "social": ("aide_sociale", "une aide sociale")}


def normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFD", texte.lower().replace("’", "'"))
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def ligne_rdv(r) -> str:
    return (f"- **{r['patient']}** ({r['telephone']}) : rendez-vous le {r['date_rdv']:%d/%m/%Y}, "
            f"{r['type'].lower()}, risque **{r['niveau'].lower()}**, "
            f"{r['absences passées']} absence(s) passée(s), délai de {r['delai (jours)']} jour(s)"
            f"{', rappel déjà envoyé' if r['rappel_envoye'] == 'Oui' else ''}")


def liste_rdv(titre, sel):
    return f"{titre}\n\n" + "\n".join(ligne_rdv(r) for _, r in sel.iterrows())


def outil_relances(tableau):
    """Risque moyen ou élevé, rappel pas encore envoyé (même règle que l'onglet Rappels)."""
    a_faire = tableau[(tableau["niveau"] != "Faible") & (tableau["rappel_envoye"] == "Non")]
    a_faire = a_faire.sort_values("score", ascending=False)
    if a_faire.empty:
        return "D'après les prédictions actuelles, aucun patient ne nécessite de relance pour le moment."
    haut = a_faire[a_faire["niveau"] == "Élevé"]
    moyen = a_faire[a_faire["niveau"] == "Moyen"]
    texte = f"{len(a_faire)} patient(s) à relancer d'après le score du modèle.\n\n"
    if not haut.empty:
        texte += "**À relancer en priorité (risque élevé) :**\n" + "\n".join(ligne_rdv(r) for _, r in haut.iterrows()) + "\n\n"
    if not moyen.empty:
        texte += "**À relancer ensuite (risque moyen) :**\n" + "\n".join(ligne_rdv(r) for _, r in moyen.iterrows())
    return texte


def outil_par_niveau(tableau, niveau):
    sel = tableau[tableau["niveau"] == niveau].sort_values("score", ascending=False)
    if sel.empty:
        return f"Aucun rendez-vous à venir n'a un risque {niveau.lower()}."
    return liste_rdv(f"{len(sel)} rendez-vous à risque {niveau.lower()} :", sel)


def outil_periode(tableau, jours, libelle):
    sel = tableau[tableau["date_rdv"] <= date.today() + timedelta(days=jours)].sort_values("date_rdv")
    if sel.empty:
        return f"Aucun rendez-vous à venir {libelle}."
    return liste_rdv(f"{len(sel)} rendez-vous {libelle} :", sel)


def outil_type(tableau, type_c):
    sel = tableau[tableau["type"] == type_c].sort_values("score", ascending=False)
    if sel.empty:
        return f"Aucun rendez-vous à venir de type « {type_c.lower()} »."
    return liste_rdv(f"{len(sel)} rendez-vous de type « {type_c.lower()} » :", sel)


def outil_patient(tableau, patient):
    sel = tableau[tableau["patient"] == patient["nom"]]
    if sel.empty:
        return f"**{patient['nom']}** n'a aucun rendez-vous à venir enregistré."
    return liste_rdv(f"Rendez-vous à venir de {patient['nom']} :", sel)


def outil_historique(patient):
    rdvs = lire(F_RDV, COLS_RDV)
    passes = rdvs[(rdvs["id_patient"] == patient["id"]) & (rdvs["statut"].isin(["Présent", "Absent"]))]
    if passes.empty:
        return f"Aucun rendez-vous passé enregistré pour **{patient['nom']}**."
    nb_abs = int((passes["statut"] == "Absent").sum())
    lignes = [f"- {en_date(r['date_rdv']):%d/%m/%Y} : {r['type_consultation'].lower()}, **{r['statut'].lower()}**"
              for _, r in passes.sort_values("date_rdv").iterrows()]
    return (f"**{patient['nom']}** : {len(passes)} rendez-vous passé(s), dont {nb_abs} absence(s).\n\n"
            + "\n".join(lignes))


def outil_pourquoi(patient):
    """Explique le score : contribution de chaque facteur = coefficient du modèle × valeur."""
    rdvs = lire(F_RDV, COLS_RDV)
    futurs = rdvs[(rdvs["id_patient"] == patient["id"]) & (rdvs["statut"] == "À venir")]
    if futurs.empty:
        return f"**{patient['nom']}** n'a aucun rendez-vous à venir : il n'y a pas de score à expliquer."
    r = futurs.sort_values("date_rdv").iloc[0]
    valeurs = valeurs_risque(patient, r, rdvs)
    modele = charger_modele()
    score = score_absence(valeurs)
    contrib = {nom: c * valeurs[nom] for nom, c in zip(modele["colonnes"], modele["coef"])}
    haut = sorted([(n, v) for n, v in contrib.items() if v > 0.02], key=lambda x: -x[1])
    bas = sorted([(n, v) for n, v in contrib.items() if v < -0.02], key=lambda x: x[1])

    def detail(nom):
        v = valeurs[nom]
        if nom == "delai":
            return f"{ETIQUETTES[nom]} ({v} jour(s))"
        if nom == "absences_passees":
            return f"{ETIQUETTES[nom]} ({v})"
        if nom == "Age":
            return f"{ETIQUETTES[nom]} ({v} ans)"
        return ETIQUETTES[nom]

    texte = (f"Pour le rendez-vous du {en_date(r['date_rdv']):%d/%m/%Y}, **{patient['nom']}** a un score de "
             f"{score:.2f}, donc un risque **{niveau_risque(score).lower()}**.\n\n")
    if haut:
        texte += "**Facteurs qui augmentent le score :** " + ", ".join(detail(n) for n, _ in haut) + ".\n\n"
    if bas:
        texte += "**Facteurs qui le diminuent :** " + ", ".join(detail(n) for n, _ in bas) + ".\n\n"
    texte += "Ce sont les poids appris par le modèle sur les données du Brésil : ils indiquent une tendance, pas une certitude."
    return texte


def outil_resume(tableau):
    if tableau.empty:
        return "Aucun rendez-vous à venir."
    c = tableau["niveau"].value_counts()
    a_relancer = len(tableau[(tableau["niveau"] != "Faible") & (tableau["rappel_envoye"] == "Non")])
    return (f"Il y a **{len(tableau)}** rendez-vous à venir : {int(c.get('Élevé', 0))} à risque élevé, "
            f"{int(c.get('Moyen', 0))} à risque moyen, {int(c.get('Faible', 0))} à risque faible. "
            f"**{a_relancer}** patient(s) restent à relancer (rappel non envoyé).")


def outil_taux_absence():
    rdvs = lire(F_RDV, COLS_RDV)
    passes = rdvs[rdvs["statut"].isin(["Présent", "Absent"])]
    if passes.empty:
        return "Aucun rendez-vous passé enregistré : impossible de calculer un taux d'absence."
    nb_abs = int((passes["statut"] == "Absent").sum())
    return (f"Dans les données de l'application, {nb_abs} rendez-vous sur {len(passes)} ont été manqués, "
            f"soit **{100 * nb_abs / len(passes):.0f} %**. (Ce sont des données fictives de démonstration.)")


def outil_absents_habituels():
    patients = lire(F_PATIENTS, COLS_PATIENTS)
    rdvs = lire(F_RDV, COLS_RDV)
    abs_par_patient = rdvs[rdvs["statut"] == "Absent"].groupby("id_patient").size().sort_values(ascending=False)
    if abs_par_patient.empty:
        return "Aucun patient n'a manqué de rendez-vous dans les données enregistrées."
    noms = dict(zip(patients["id"], patients["nom"]))
    lignes = [f"- **{noms.get(i, i)}** : {n} absence(s)" for i, n in abs_par_patient.items()]
    return "Patients ayant déjà manqué des rendez-vous :\n\n" + "\n".join(lignes)


def outil_patients_condition(cle, libelle):
    patients = lire(F_PATIENTS, COLS_PATIENTS)
    sel = patients[patients[cle] == "Oui"]
    if sel.empty:
        return f"Aucun patient enregistré avec : {libelle}."
    return f"{len(sel)} patient(s) avec {libelle} :\n\n" + "\n".join(f"- {n} ({i})" for n, i in zip(sel["nom"], sel["id"]))


def outil_rappels(tableau, envoyes):
    etat = "Oui" if envoyes else "Non"
    sel = tableau[tableau["rappel_envoye"] == etat].sort_values("date_rdv")
    if sel.empty:
        return "Aucun rappel déjà envoyé pour le moment." if envoyes else "Tous les rappels ont déjà été envoyés."
    titre = "Rappels déjà envoyés :" if envoyes else "Rendez-vous dont le rappel n'est pas encore envoyé :"
    return liste_rdv(titre, sel)



def outil_fiche(patient):
    age = age_a(en_date(patient["date_naissance"]), date.today())
    suivis = [n for n, c in [("hypertension", "hypertension"), ("diabète", "diabete"),
                             ("alcoolisme", "alcoolisme"), ("handicap", "handicap")] if patient[c] == "Oui"]
    rdvs = lire(F_RDV, COLS_RDV)
    passes = rdvs[(rdvs["id_patient"] == patient["id"]) & (rdvs["statut"].isin(["Présent", "Absent"]))]
    nb_abs = int((passes["statut"] == "Absent").sum())
    return (f"**{patient['nom']}** ({patient['id']}) · {patient['sexe']}, {age} ans · {patient['telephone']}\n\n"
            f"- Aide sociale : {patient['aide_sociale'].lower()}\n"
            f"- Suivi particulier : {', '.join(suivis) if suivis else 'aucun enregistré'}\n"
            f"- Rendez-vous passés : {len(passes)}, dont {nb_abs} absence(s)")


def outil_facteurs():
    m = charger_modele()
    positifs = sorted([(c, n) for n, c in zip(m["colonnes"], m["coef"]) if c > 0.01 and n != "genre"], reverse=True)
    lignes = [f"- {ETIQUETTES[n]}" for _, n in positifs]
    return ("Facteurs qui **augmentent** le plus le risque d'absence d'après le modèle, du plus au moins important "
            "par unité :\n\n" + "\n".join(lignes) +
            "\n\nLe délai de prise du rendez-vous pèse peu par jour, mais s'additionne : un rendez-vous pris "
            "longtemps à l'avance est plus risqué. Ce sont des tendances apprises sur les données du Brésil.")


def outil_types():
    return "Types de consultation enregistrés : " + ", ".join(t.lower() for t in TYPES) + "."


def outil_compte_rdv(manques):
    rdvs = lire(F_RDV, COLS_RDV)
    if manques:
        n = int((rdvs["statut"] == "Absent").sum())
        return f"**{n}** rendez-vous ont été manqués sur {len(rdvs)} enregistrés."
    n_av = int((rdvs["statut"] == "À venir").sum())
    return f"**{len(rdvs)}** rendez-vous sont enregistrés, dont {n_av} à venir."


def outil_prevision(tableau):
    c = tableau["niveau"].value_counts()
    return (f"Le modèle ne peut pas annoncer un nombre exact d'absences. Il classe les {len(tableau)} rendez-vous à venir : "
            f"{int(c.get('Élevé', 0))} à risque élevé, {int(c.get('Moyen', 0))} à risque moyen, "
            f"{int(c.get('Faible', 0))} à risque faible. Dans les données de test, environ 10 % des patients de niveau "
            "faible, 22 % de niveau moyen et 35 % de niveau élevé ont manqué leur rendez-vous.")


def outil_action_refusee():
    return ("Je ne peux pas modifier, annuler ou supprimer de données : je consulte seulement. "
            "Pour agir sur un rendez-vous ou un patient, utilisez les menus **Patients** et **Rendez-vous**.")


def trouver_patient(q, patients):
    for _, p in patients.iterrows():
        prenom = normaliser(p["nom"].split()[0])
        if normaliser(p["nom"]) in q or re.search(rf"\b{re.escape(prenom)}\b", q):
            return p
    return None


def a(q, *mots):
    """Vrai si l'un des mots apparaît dans la question (mots courts : mot entier)."""
    return any((re.search(rf"\b{re.escape(m)}\b", q) if len(m) <= 4 else m in q) for m in mots)


def faq_trouvee(q):
    """FAQ la plus proche : la somme des longueurs des mots-clés trouvés sert de score."""
    meilleur, score_max = None, 0
    for nom, _, mots, texte in FAQ:
        score = sum(len(m) for m in mots if (re.search(rf"\b{re.escape(m)}\b", q) if len(m) <= 4 else m in q))
        if score > score_max:
            meilleur, score_max = (nom, texte), score
    return meilleur if score_max >= 3 else None


MSG_MEDICAL = ("Je ne peux pas répondre à cette question : je ne donne aucun avis médical, diagnostic ou conseil "
               "de traitement. Pour cela, adressez-vous à un professionnel de santé. Je peux en revanche vous montrer "
               "quels patients risquent de manquer leur rendez-vous.")
MSG_INCONNU = ("Je n'ai pas trouvé de réponse à cette question dans les données ou les résultats du modèle, "
               "et je préfère ne rien inventer. Je peux répondre sur les relances, les rendez-vous, un patient "
               "(« Pourquoi Awa est-elle à risque ? »), les chiffres de l'application ou le fonctionnement du modèle. "
               "Écrivez « aide » pour voir tout ce que je sais faire.")


def repondre_regles(question: str) -> str:
    q = normaliser(question)
    comment = a(q, "comment")
    if any(mot in q for mot in MOTS_MEDICAUX):
        return MSG_MEDICAL
    tableau = rdv_a_venir_avec_risque()
    patients = lire(F_PATIENTS, COLS_PATIENTS)

    # ---- 1. un patient est cité ----
    p = trouver_patient(q, patients)
    if p is not None:
        if a(q, "annuler", "supprimer", "modifier", "effacer", "changer", "deplacer", "reporter"):
            return outil_action_refusee()
        if a(q, "telephone", "numero", "joindre", "contact", "tel"):
            return f"Téléphone de **{p['nom']}** : {p['telephone'] or 'non renseigné'}."
        if a(q, "age", "ans", "naissance", "ne le", "nee"):
            return f"**{p['nom']}** est né(e) le {en_date(p['date_naissance']):%d/%m/%Y} ({age_a(en_date(p['date_naissance']), date.today())} ans)."
        if a(q, "pourquoi", "explique", "facteur", "raison", "niveau", "risque", "risques", "score", "viendra",
             "venir", "manquer", "probab"):
            return outil_pourquoi(p)
        if a(q, "historique", "absence", "absences", "manque", "rate", "passe", "deja", "venu", "present", "absent"):
            return outil_historique(p)
        if a(q, "dossier", "fiche", "informations", "infos", "profil", "tout sur", "sexe", "aide"):
            return outil_fiche(p)
        if a(q, "quand", "date", "prochain", "rendez-vous", "rdv", "consultation"):
            return outil_patient(tableau, p)
        return outil_fiche(p) + "\n\n" + outil_patient(tableau, p)

    # ---- 2. questions chiffrées et listes précises ----
    if a(q, "quelle date", "date d'aujourd'hui", "quel jour", "date du jour"):
        return f"Nous sommes le {date.today():%d/%m/%Y}."
    if a(q, "sms") and a(q, "pourquoi", "retire", "utilise", "utilisez", "facteur") and not comment:
        return FAQ_TEXTES["faq_sms"]
    m_age = re.search(r"(plus|moins) de (\d+) ans", q)
    if m_age:
        seuil = int(m_age.group(2))
        ages = patients.assign(age=patients["date_naissance"].map(lambda d: age_a(en_date(d), date.today())))
        sel = ages[ages["age"] > seuil] if m_age.group(1) == "plus" else ages[ages["age"] < seuil]
        if sel.empty:
            return f"Aucun patient enregistré avec {m_age.group(1)} de {seuil} ans."
        return (f"{len(sel)} patient(s) avec {m_age.group(1)} de {seuil} ans :\n\n"
                + "\n".join(f"- {n} ({ag} ans)" for n, ag in zip(sel["nom"], sel["age"])))
    if a(q, "sans rendez-vous", "sans rdv", "aucun rendez-vous"):
        rdvs_a_venir = lire(F_RDV, COLS_RDV)
        avec = set(rdvs_a_venir[rdvs_a_venir["statut"] == "À venir"]["id_patient"])
        sel = patients[~patients["id"].isin(avec)]
        if sel.empty:
            return "Tous les patients ont au moins un rendez-vous à venir."
        return f"{len(sel)} patient(s) sans rendez-vous à venir :\n\n" + "\n".join(f"- {n} ({i})" for n, i in zip(sel["nom"], sel["id"]))
    if a(q, "vont", "risquent", "probablement", "prochainement") and a(q, "manquer", "absent", "absents", "venir"):
        return outil_prevision(tableau) if not tableau.empty else "Aucun rendez-vous à venir."
    if a(q, "taux", "pourcentage", "proportion"):
        return outil_taux_absence()
    for mot, (cle, libelle) in CONDITIONS.items():
        if mot in q and not comment:
            return outil_patients_condition(cle, libelle)
    if a(q, "combien") and a(q, "vont", "va", "risquent", "probablement", "manquer", "absents"):
        return outil_prevision(tableau) if not tableau.empty else "Aucun rendez-vous à venir."
    if a(q, "combien") and a(q, "rendez-vous", "rdv", "rendez", "consultations") and a(q, "manque", "manques", "absent", "rate", "rates"):
        return outil_compte_rdv(True)
    if a(q, "combien") and a(q, "rappel", "rappels") and not comment:
        return outil_rappels(tableau, True) if not tableau.empty else "Aucun rendez-vous à venir."
    if a(q, "combien") and a(q, "patient", "patients"):
        return f"**{len(patients)}** patients sont enregistrés dans l'application."
    if a(q, "combien") and a(q, "rendez-vous", "rdv", "rendez"):
        return outil_compte_rdv(False)
    for t in TYPES:
        if normaliser(t).split()[-1] in q and not a(q, "types", "type"):
            return outil_type(tableau, t) if not tableau.empty else "Aucun rendez-vous à venir."
    if a(q, "types", "type") and a(q, "consultation", "consultations", "rendez-vous"):
        return outil_types()
    if a(q, "facteur", "facteurs", "influence", "influencent") and not comment:
        return outil_facteurs()
    if a(q, "pas ete rappele", "pas rappele", "pas recu", "sans rappel", "non envoye", "pas encore", "pas ete relance"):
        return outil_rappels(tableau, False) if not tableau.empty else "Aucun rendez-vous à venir."
    if a(q, "annuler", "supprimer", "modifier", "effacer") and not comment:
        return outil_action_refusee()

    # ---- 3. questions sur l'application (comment, pourquoi, modèle, données…) ----
    faq = faq_trouvee(q)
    if faq and (comment or not a(q, "patients", "rendez-vous", "relance", "relancer", "risque eleve", "appeler")):
        return faq[1]

    # ---- 4. consultation des prédictions ----
    if tableau.empty:
        return "Il n'y a aucun rendez-vous à venir, donc rien à analyser pour le moment."
    if a(q, "relance", "relancer", "rappeler", "prioritaire", "priorite", "contacter", "appeler", "joindre"):
        return outil_relances(tableau)
    if a(q, "eleve", "haut", "danger", "critique", "urgent", "plus a risque", "a risque", "risque", "inquiet",
         "inquietant", "inquietants", "preoccupant", "preoccupants", "alarmant", "alarmants", "sensible", "difficile") and not a(q, "moyen", "faible"):
        return outil_par_niveau(tableau, "Élevé")
    if "moyen" in q:
        return outil_par_niveau(tableau, "Moyen")
    if a(q, "faible", "bas", "sans risque"):
        return outil_par_niveau(tableau, "Faible")
    if a(q, "envoye", "envoyes", "rappele", "rappeles"):
        return outil_rappels(tableau, True)
    if a(q, "deja manque", "ont manque", "ont rate", "souvent absent", "absents habituels", "ont ete absents",
         "deja ete absent", "raté", "rate", "absent", "absents", "absence", "absences", "manque"):
        return outil_absents_habituels()
    if a(q, "semaine"):
        return outil_periode(tableau, 7, "dans les 7 prochains jours")
    if a(q, "mois"):
        return outil_periode(tableau, 30, "dans les 30 prochains jours")
    if a(q, "demain"):
        return outil_periode(tableau, 1, "d'ici demain")
    if a(q, "aujourd'hui", "aujourdhui", "aujourd"):
        return outil_periode(tableau, 0, "aujourd'hui")
    if a(q, "prochain", "prochains"):
        return liste_rdv("Prochain rendez-vous :", tableau.sort_values("date_rdv").head(1))
    if a(q, "liste", "tous", "patients"):
        return (f"{len(patients)} patients enregistrés :\n\n"
                + "\n".join(f"- {n} ({i})" for n, i in zip(patients["nom"], patients["id"])))
    if a(q, "resume", "situation", "bilan", "synthese", "statistique", "statistiques", "rendez-vous"):
        return outil_resume(tableau)
    return MSG_INCONNU


# ---------------------------------------------------------------
# Mode modèle de langage (optionnel) : un LLM comprend la question et CHOISIT l'outil à appeler.
# Il ne rédige aucune réponse : le texte affiché vient toujours des outils ci-dessus.
# Sans clé ou en cas d'erreur, l'assistant revient automatiquement aux règles.
# ---------------------------------------------------------------
OUTILS_LLM = {
    "relances": "patients à relancer en priorité (risque moyen ou élevé, rappel pas encore envoyé)",
    "risque_eleve": "rendez-vous à risque élevé, patients les plus à risque",
    "risque_moyen": "rendez-vous à risque moyen", "risque_faible": "rendez-vous à risque faible",
    "rdv_semaine": "rendez-vous des 7 prochains jours", "rdv_mois": "rendez-vous des 30 prochains jours",
    "rdv_demain": "rendez-vous de demain", "rdv_aujourdhui": "rendez-vous d'aujourd'hui",
    "rdv_prochain": "le prochain rendez-vous",
    "rdv_par_type": "rendez-vous d'un type (parametre = Consultation générale, Vaccination, "
                    "Suivi maladie chronique ou Consultation prénatale)",
    "types_consultation": "quels types de consultation existent",
    "resume": "résumé de la situation (nombre de rendez-vous à venir par niveau)",
    "prevision": "combien de patients vont manquer leur rendez-vous (prévision globale)",
    "taux_absence": "taux / pourcentage d'absence observé dans les données de l'application",
    "absents_habituels": "patients qui ont déjà manqué des rendez-vous",
    "nb_patients": "nombre de patients enregistrés", "liste_patients": "liste de tous les patients",
    "nb_rdv": "nombre de rendez-vous enregistrés", "nb_rdv_manques": "nombre de rendez-vous manqués",
    "patients_condition": "patients ayant une condition (parametre = diabete, hypertension, alcoolisme, "
                          "handicap ou aide_sociale)",
    "facteurs_modele": "quels facteurs influencent le plus le risque dans le modèle",
    "rappels_envoyes": "rappels déjà envoyés", "rappels_non_envoyes": "rappels pas encore envoyés",
    "patient_rdv": "rendez-vous à venir d'un patient (patient = son prénom)",
    "patient_historique": "historique et absences passées d'un patient (patient = son prénom)",
    "patient_pourquoi": "niveau de risque d'un patient et pourquoi (patient = son prénom)",
    "patient_fiche": "informations enregistrées sur un patient : âge, sexe, suivi (patient = son prénom)",
    "patient_telephone": "téléphone d'un patient (patient = son prénom)",
    "action_refusee": "demande de modifier, annuler ou supprimer des données",
    "refus_medical": "question médicale (diagnostic, médicament, traitement, gravité d'une maladie)",
    "hors_sujet": "question sans rapport avec l'application ou introuvable dans les données",
}
for _nom, _desc, _mots, _texte in FAQ:
    OUTILS_LLM[_nom] = _desc


def cle_llm():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def appeler_llm(prompt: str) -> str:
    """Appelle Gemini (API REST). Renvoie le texte brut, ou lève une exception."""
    modele = "gemini-2.5-flash"
    try:
        modele = st.secrets.get("GEMINI_MODEL", modele)
    except Exception:
        pass
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modele}:generateContent"
    corps = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}).encode()
    requete = urllib.request.Request(url, data=corps, headers={"Content-Type": "application/json",
                                                               "x-goog-api-key": cle_llm()})
    with urllib.request.urlopen(requete, timeout=6) as rep:
        return json.load(rep)["candidates"][0]["content"]["parts"][0]["text"]


def choisir_outil(question: str):
    """Demande au LLM quel outil appeler. Renvoie (outil, patient, parametre) ou None."""
    catalogue = "\n".join(f"- {nom} : {desc}" for nom, desc in OUTILS_LLM.items())
    prompt = ("Tu es le routeur d'un assistant de centre de santé. Choisis l'outil qui répond à la question "
              "du membre du personnel. Tu ne réponds jamais toi-même à la question.\n"
              f"Outils :\n{catalogue}\n\nQuestion : {question}\n\n"
              'Réponds uniquement en JSON : {"outil": "...", "patient": "prénom ou vide", "parametre": "valeur ou vide"}')
    donnees = json.loads(appeler_llm(prompt))
    if donnees.get("outil") not in OUTILS_LLM:
        return None
    return donnees["outil"], str(donnees.get("patient") or ""), str(donnees.get("parametre") or "")


def executer_outil(outil, patient_nom, parametre):
    tableau = rdv_a_venir_avec_risque()
    patients = lire(F_PATIENTS, COLS_PATIENTS)
    if outil in FAQ_TEXTES:
        return FAQ_TEXTES[outil]
    if outil == "refus_medical":
        return MSG_MEDICAL
    if outil == "hors_sujet":
        return MSG_INCONNU
    if outil == "action_refusee":
        return outil_action_refusee()
    if outil.startswith("patient_"):
        cible = normaliser(patient_nom).split()[:1]
        trouves = [p for _, p in patients.iterrows() if cible and normaliser(p["nom"].split()[0]) == cible[0]]
        if not trouves:
            return "Je ne trouve pas ce patient dans l'application."
        p = trouves[0]
        if outil == "patient_telephone":
            return f"Téléphone de **{p['nom']}** : {p['telephone'] or 'non renseigné'}."
        return {"patient_rdv": lambda: outil_patient(tableau, p), "patient_historique": lambda: outil_historique(p),
                "patient_pourquoi": lambda: outil_pourquoi(p), "patient_fiche": lambda: outil_fiche(p)}[outil]()
    if outil == "nb_patients":
        return f"**{len(patients)}** patients sont enregistrés dans l'application."
    if outil == "liste_patients":
        return (f"{len(patients)} patients enregistrés :\n\n"
                + "\n".join(f"- {n} ({i})" for n, i in zip(patients["nom"], patients["id"])))
    if outil == "nb_rdv":
        return outil_compte_rdv(False)
    if outil == "nb_rdv_manques":
        return outil_compte_rdv(True)
    if outil == "taux_absence":
        return outil_taux_absence()
    if outil == "absents_habituels":
        return outil_absents_habituels()
    if outil == "types_consultation":
        return outil_types()
    if outil == "facteurs_modele":
        return outil_facteurs()
    if outil == "patients_condition":
        cles = {"diabete": "un diabète", "hypertension": "une hypertension", "alcoolisme": "de l'alcoolisme",
                "handicap": "un handicap", "aide_sociale": "une aide sociale"}
        if parametre not in cles:
            return "Je ne connais pas cette condition."
        return outil_patients_condition(parametre, cles[parametre])
    if tableau.empty:
        return "Il n'y a aucun rendez-vous à venir, donc rien à analyser pour le moment."
    return {
        "relances": lambda: outil_relances(tableau),
        "risque_eleve": lambda: outil_par_niveau(tableau, "Élevé"),
        "risque_moyen": lambda: outil_par_niveau(tableau, "Moyen"),
        "risque_faible": lambda: outil_par_niveau(tableau, "Faible"),
        "rdv_semaine": lambda: outil_periode(tableau, 7, "dans les 7 prochains jours"),
        "rdv_mois": lambda: outil_periode(tableau, 30, "dans les 30 prochains jours"),
        "rdv_demain": lambda: outil_periode(tableau, 1, "d'ici demain"),
        "rdv_aujourdhui": lambda: outil_periode(tableau, 0, "aujourd'hui"),
        "rdv_prochain": lambda: liste_rdv("Prochain rendez-vous :", tableau.sort_values("date_rdv").head(1)),
        "rdv_par_type": lambda: outil_type(tableau, parametre) if parametre in TYPES else "Type de consultation inconnu.",
        "resume": lambda: outil_resume(tableau),
        "prevision": lambda: outil_prevision(tableau),
        "rappels_envoyes": lambda: outil_rappels(tableau, True),
        "rappels_non_envoyes": lambda: outil_rappels(tableau, False),
    }[outil]()


def repondre(question: str) -> str:
    """Le modèle de langage choisit l'outil ; s'il échoue ou se déclare incapable, les règles prennent le relais."""
    st.session_state.mode_assistant = "règles"
    st.session_state.erreur_llm = ""
    if any(m in normaliser(question) for m in MOTS_MEDICAUX):
        return MSG_MEDICAL  # refus médical immédiat, sans passer par le LLM
    if cle_llm() and time.time() > st.session_state.get("llm_pause_jusqua", 0):
        try:
            choix = choisir_outil(question)
            if choix and choix[0] not in ("hors_sujet", "refus_medical"):
                texte = executer_outil(*choix)
                st.session_state.mode_assistant = "modèle de langage"
                return texte
        except Exception as erreur:
            st.session_state.erreur_llm = f"{type(erreur).__name__}: {str(erreur)[:80]}"
            st.session_state.llm_pause_jusqua = time.time() + 300  # on n'insiste pas pendant 5 minutes
    return repondre_regles(question)


# ---------------------------------------------------------------
# Interface
# ---------------------------------------------------------------
if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"))


def flash(message):
    """Affiche un message de confirmation après le rechargement de la page."""
    st.session_state.flash = message
    st.rerun()


def carte(libelle, valeur, couleur):
    return (f'<div class="kpi" style="border-left-color:{couleur}"><div class="kpi-val">{valeur}</div>'
            f'<div class="kpi-lib">{libelle}</div></div>')


PAGES = [("Patients", ":material/group:"), ("Rendez-vous", ":material/event:"),
         ("Prédiction", ":material/monitoring:"), ("Rappels", ":material/notifications:"),
         ("Assistant IA", ":material/smart_toy:")]
if "page" not in st.session_state:
    st.session_state.page = "Patients"

with st.sidebar:
    st.markdown('<div class="marque"><span class="logo">+</span> Suivi RDV</div>', unsafe_allow_html=True)
    st.caption("Centre de santé · prototype")
    st.write("")
    for nom_page, icone in PAGES:
        actif = st.session_state.page == nom_page
        if st.button(nom_page, icon=icone, key=f"nav_{nom_page}", width="stretch",
                     type="primary" if actif else "tertiary"):
            st.session_state.page = nom_page
            st.rerun()
    st.divider()
    with st.expander("À propos"):
        st.write("Projet du module *Innovations technologiques en IA* (L3 Ingénierie Informatique, UASZ). "
                 "Modèle entraîné sur le jeu Kaggle « Medical Appointment No Shows » (Brésil). "
                 "Outil d'aide à la décision : la décision finale reste humaine.")
    if st.button("Réinitialiser la démo", icon=":material/restart_alt:", width="stretch"):
        donnees_demo()
        st.rerun()
    st.caption("Données fictives, stockées en CSV.")

page = st.session_state.page

tableau_global = rdv_a_venir_avec_risque()
nb_patients = len(lire(F_PATIENTS, COLS_PATIENTS))
if tableau_global.empty:
    nb_avenir = nb_eleve = nb_relance = 0
else:
    nb_avenir = len(tableau_global)
    nb_eleve = int((tableau_global["niveau"] == "Élevé").sum())
    nb_relance = int(((tableau_global["niveau"] != "Faible") & (tableau_global["rappel_envoye"] == "Non")).sum())

SOUS_TITRES = {"Patients": "Dossiers des patients du centre", "Rendez-vous": "Planning et résultats des consultations",
               "Prédiction": "Risque d'absence des rendez-vous à venir", "Rappels": "Patients à relancer en priorité",
               "Assistant IA": "Posez vos questions sur les patients et les prédictions"}
st.markdown(f'<div class="entete"><div><div class="titre">{page}</div>'
            f'<div class="sous-titre">{SOUS_TITRES[page]}</div></div>'
            '<span class="puce">Données fictives</span></div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.markdown(carte("Patients", nb_patients, "#2563EB"), unsafe_allow_html=True)
k2.markdown(carte("Rendez-vous à venir", nb_avenir, "#0EA5E9"), unsafe_allow_html=True)
k3.markdown(carte("Risque élevé", nb_eleve, COULEURS["Élevé"]), unsafe_allow_html=True)
k4.markdown(carte("Patients à relancer", nb_relance, COULEURS["Moyen"]), unsafe_allow_html=True)
st.write("")


# ---------------- Onglet 1 : patients ----------------
if page == "Patients":
    liste_col, form_col = st.columns([3, 2], gap="large")
    with liste_col:
        st.subheader("Patients")
        liste = lire(F_PATIENTS, COLS_PATIENTS)
        recherche = st.text_input("Rechercher", placeholder="Nom ou numéro", label_visibility="collapsed",
                                  icon=":material/search:")
        vue = liste.copy()
        vue["suivi"] = vue.apply(lambda r: ", ".join(
            n for n, c in [("Hypertension", "hypertension"), ("Diabète", "diabete"),
                           ("Alcoolisme", "alcoolisme"), ("Handicap", "handicap")] if r[c] == "Oui"), axis=1)
        if recherche:
            vue = vue[vue["nom"].str.contains(recherche, case=False) | vue["id"].str.contains(recherche, case=False)]
        st.dataframe(vue[["id", "nom", "sexe", "telephone", "suivi"]], width="stretch",
                     hide_index=True, height=420,
                     column_config={"id": "N°", "nom": "Nom", "sexe": "Sexe",
                                    "telephone": "Téléphone", "suivi": "Suivi particulier"})
    with form_col:
        with st.container(border=True):
            st.subheader("Nouveau patient")
            with st.form("form_patient", clear_on_submit=True, border=False):
                nom = st.text_input("Nom (fictif)")
                c1, c2 = st.columns(2)
                naissance = c1.date_input("Naissance", value=date(1995, 1, 1),
                                          min_value=date(1920, 1, 1), max_value=date.today())
                sexe = c2.selectbox("Sexe", ["Femme", "Homme"])
                tel = st.text_input("Téléphone (fictif)")
                aide = st.selectbox("Aide sociale", ["Non", "Oui"])
                s1, s2 = st.columns(2)
                hyper = s1.checkbox("Hypertension")
                diab = s2.checkbox("Diabète")
                alcool = s1.checkbox("Alcoolisme")
                handi = s2.checkbox("Handicap")
                envoyer = st.form_submit_button("Enregistrer", type="primary", width="stretch")
            if envoyer:
                if not nom.strip():
                    st.error("Le nom est obligatoire.")
                else:
                    patients = lire(F_PATIENTS, COLS_PATIENTS)
                    nouveau = {"id": prochain_id(patients, "id", "P"), "nom": nom.strip(),
                               "date_naissance": str(naissance), "sexe": sexe, "telephone": tel.strip(),
                               "aide_sociale": aide, "hypertension": "Oui" if hyper else "Non",
                               "diabete": "Oui" if diab else "Non", "alcoolisme": "Oui" if alcool else "Non",
                               "handicap": "Oui" if handi else "Non"}
                    sauver(pd.concat([patients, pd.DataFrame([nouveau])], ignore_index=True), F_PATIENTS)
                    flash(f"Patient enregistré : {nouveau['nom']}")

# ---------------- Onglet 2 : rendez-vous ----------------
if page == "Rendez-vous":
    patients = lire(F_PATIENTS, COLS_PATIENTS)
    rdvs = lire(F_RDV, COLS_RDV)
    noms_patients = {r["id"]: f'{r["id"]} – {r["nom"]}' for _, r in patients.iterrows()}

    gauche, droite = st.columns(2, gap="large")
    with gauche:
        with st.container(border=True):
            st.subheader("Nouveau rendez-vous")
            if patients.empty:
                st.info("Enregistrez d'abord un patient.")
            else:
                with st.form("form_rdv", clear_on_submit=True, border=False):
                    choix = st.selectbox("Patient", list(noms_patients), format_func=lambda i: noms_patients[i])
                    jour = st.date_input("Date", value=date.today() + timedelta(days=7), min_value=date.today())
                    type_c = st.selectbox("Type de consultation", TYPES)
                    valider = st.form_submit_button("Créer le rendez-vous", type="primary", width="stretch")
                if valider:
                    nouveau = {"id_rdv": prochain_id(rdvs, "id_rdv", "R"), "id_patient": choix,
                               "date_prise": str(date.today()), "date_rdv": str(jour),
                               "type_consultation": type_c, "statut": "À venir", "rappel_envoye": "Non"}
                    sauver(pd.concat([rdvs, pd.DataFrame([nouveau])], ignore_index=True), F_RDV)
                    flash(f"Rendez-vous créé pour le {jour:%d/%m/%Y}")
    with droite:
        with st.container(border=True):
            st.subheader("Résultat d'un rendez-vous")
            a_venir = rdvs[rdvs["statut"] == "À venir"]
            if a_venir.empty:
                st.info("Aucun rendez-vous à venir.")
            else:
                libelles = {r["id_rdv"]: f'{noms_patients.get(r["id_patient"], r["id_patient"])} – {r["date_rdv"]}'
                            for _, r in a_venir.iterrows()}
                with st.form("form_resultat", border=False):
                    rid = st.selectbox("Rendez-vous", list(libelles), format_func=lambda i: libelles[i])
                    resultat = st.radio("Le patient est…", ["Présent", "Absent"], horizontal=True)
                    ok = st.form_submit_button("Enregistrer le résultat", type="primary", width="stretch")
                if ok:
                    rdvs.loc[rdvs["id_rdv"] == rid, "statut"] = resultat
                    sauver(rdvs, F_RDV)
                    flash("Résultat enregistré")

    st.subheader("Tous les rendez-vous")
    filtre = st.segmented_control("Statut", ["Tous", "À venir", "Présent", "Absent"], default="Tous",
                                  label_visibility="collapsed")
    affichage = rdvs.copy()
    affichage["patient"] = affichage["id_patient"].map(lambda i: noms_patients.get(i, i).split(" – ")[-1])
    if filtre and filtre != "Tous":
        affichage = affichage[affichage["statut"] == filtre]
    affichage = affichage.sort_values("date_rdv", ascending=False)
    couleurs_statut = {"À venir": "background-color:#DBEAFE;color:#1E40AF;font-weight:600",
                       "Présent": "background-color:#DCFCE7;color:#166534;font-weight:600",
                       "Absent": "background-color:#FEE2E2;color:#991B1B;font-weight:600"}
    st.dataframe(affichage[["id_rdv", "patient", "date_rdv", "type_consultation", "statut", "rappel_envoye"]]
                 .style.map(lambda v: couleurs_statut.get(v, ""), subset=["statut"]),
                 width="stretch", hide_index=True, height=320,
                 column_config={"id_rdv": "N°", "patient": "Patient", "date_rdv": "Date",
                                "type_consultation": "Type", "statut": "Statut", "rappel_envoye": "Rappel envoyé"})

# ---------------- Onglet 3 : prédiction ----------------
if page == "Prédiction":
    tableau = tableau_global
    if tableau.empty:
        st.info("Aucun rendez-vous à venir.")
    else:
        tableau = tableau.sort_values("score", ascending=False)
        compte = tableau["niveau"].value_counts()
        g1, g2 = st.columns([2, 5], gap="large")
        with g1:
            st.subheader("Répartition")
            donut = alt.Chart(pd.DataFrame({
                "niveau": ["Élevé", "Moyen", "Faible"],
                "rendez-vous": [int(compte.get(n, 0)) for n in ["Élevé", "Moyen", "Faible"]],
            })).mark_arc(innerRadius=60, outerRadius=100).encode(
                theta="rendez-vous:Q",
                color=alt.Color("niveau:N", scale=alt.Scale(domain=list(COULEURS), range=list(COULEURS.values())),
                                legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["niveau", "rendez-vous"],
            ).properties(height=250)
            st.altair_chart(donut, width="stretch")
        with g2:
            st.subheader("Rendez-vous à venir")
            vue = tableau[["patient", "date_rdv", "delai (jours)", "absences passées", "score", "niveau"]]
            st.dataframe(vue.style.map(colorier, subset=["niveau"]), width="stretch", hide_index=True, height=300,
                         column_config={"patient": "Patient", "date_rdv": "Date",
                                        "delai (jours)": "Délai (j)", "absences passées": "Absences",
                                        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1,
                                                                                 format="%.2f"),
                                        "niveau": "Risque"})
        st.caption(f"Parmi les données de test du modèle, {TAUX_OBSERVE['Faible']} % des patients de niveau faible, "
                   f"{TAUX_OBSERVE['Moyen']} % de niveau moyen et {TAUX_OBSERVE['Élevé']} % de niveau élevé "
                   "ont manqué leur rendez-vous.")
    with st.expander("Comment le niveau est-il calculé ?"):
        st.markdown(
            "Le modèle a appris sur 110 000 rendez-vous anonymisés quels facteurs augmentent le risque "
            "(délai, absences passées, âge…). Chaque rendez-vous reçoit un **score** entre 0 et 1, rangé en "
            "niveau : **faible** (< 0,4), **moyen** (0,4 à 0,6) ou **élevé** (> 0,6). "
            "Le score classe les patients, ce n'est pas une probabilité exacte.")

# ---------------- Onglet 4 : rappels et alertes ----------------
if page == "Rappels":
    tete, filtre_col = st.columns([3, 2])
    tete.subheader("Rappels à envoyer")
    jours = filtre_col.segmented_control("Période", [7, 14, 30], default=7, format_func=lambda j: f"{j} jours",
                                         label_visibility="collapsed") or 7
    tableau = tableau_global
    if tableau.empty:
        st.info("Aucun rendez-vous à venir.")
    else:
        limite = date.today() + timedelta(days=jours)
        a_relancer = tableau[(tableau["date_rdv"] <= limite) & (tableau["niveau"] != "Faible")
                             & (tableau["rappel_envoye"] == "Non")].sort_values("score", ascending=False)
        if a_relancer.empty:
            st.success("Aucun patient à relancer pour cette période.")
        for _, r in a_relancer.iterrows():
            with st.container(border=True):
                c_info, c_niveau, c_action = st.columns([5, 1.3, 2.2], vertical_alignment="center")
                c_info.markdown(f"**{r['patient']}** · {r['telephone']}<br>"
                                f"<span class='discret'>{r['date_rdv']:%d/%m/%Y} · {r['type']}</span>",
                                unsafe_allow_html=True)
                c_niveau.markdown(pastille(r["niveau"]), unsafe_allow_html=True)
                if c_action.button("Rappel envoyé", key=f"envoye_{r['id_rdv']}", type="primary",
                                   icon=":material/check:", width="stretch"):
                    rdvs = lire(F_RDV, COLS_RDV)
                    rdvs.loc[rdvs["id_rdv"] == r["id_rdv"], "rappel_envoye"] = "Oui"
                    sauver(rdvs, F_RDV)
                    st.rerun()
                with st.expander("Message SMS"):
                    st.code(f"Bonjour {r['patient'].split()[0]}, nous vous rappelons votre rendez-vous "
                            f"({r['type'].lower()}) au centre de santé le {r['date_rdv']:%d/%m/%Y}. "
                            "Merci de venir ou de nous prévenir si vous ne pouvez pas.",
                            language=None, wrap_lines=True)
        deja = tableau[tableau["rappel_envoye"] == "Oui"]
        if not deja.empty:
            with st.expander(f"Rappels déjà envoyés ({len(deja)})"):
                st.dataframe(deja[["patient", "date_rdv", "type"]], width="stretch", hide_index=True)

# ---------------- Onglet 5 : assistant IA ----------------
if page == "Assistant IA":
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content":
            "Bonjour. Je consulte les prédictions du modèle et les données du centre. Que voulez-vous savoir ?"}]

    def choisir_suggestion():
        st.session_state.question_en_attente = st.session_state.suggestion
        st.session_state.suggestion = None

    st.pills("Suggestions", SUGGESTIONS, key="suggestion", on_change=choisir_suggestion,
             label_visibility="collapsed")
    saisie = st.chat_input("Posez votre question…")
    question = saisie or st.session_state.pop("question_en_attente", None)
    with st.container(height=480, border=True):
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Je consulte les données…"):
                    reponse = repondre(question)
                st.markdown(reponse)
            st.session_state.messages.append({"role": "assistant", "content": reponse})
    st.caption(f"Les réponses viennent des données et du modèle, jamais d'une invention · "
               f"compréhension : {st.session_state.get('mode_assistant', 'règles')}"
               + (f" · LLM indisponible ({st.session_state.erreur_llm})" if st.session_state.get("erreur_llm")
                  else ("" if cle_llm() else " · clé Gemini non détectée")))
