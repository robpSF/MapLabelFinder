import streamlit as st
from collections import defaultdict, Counter
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the stop words
nltk.download('stopwords')

# Crisis categories and associated keywords
CRISIS_CATEGORIES = {
    "Demonstrations": ["Protests", "Protest", "Riots Rallies", "Rally", "Demonstration", "Demonstrations", "March", "Crowds", "boycott", "picket", "resist"],
    "Politician": ["Prime Minister", "PM", "President", "Ambassador", "Governor", "Senator", "MP", "Member of Parliament", "Congressman", "Congresswoman", "Delegate"],
    "Non-governmental Individuals": ["Union Leader", "Representative", "General-Secretary", "Secretary", "Councillor", "Elder"],
    "Party": ["Party", "Coalition"],
    "Political Events": ["Election", "Council Meeting", "Prime Ministers Questions", "State-visit", "Speech", "Address"],
    "Military Areas": ["Base", "Fort", "Airbase", "Naval Base", "Port", "IED sites", "Mine field", "Nuclear Silo", "installation"],
    "Military Figures": ["General", "Commander", "Admiral", "Squadron Leader", "Captain"],
    "Land Assets (Personnel)": ["Soldiers", "Special Forces", "Military Police", "Security Forces", "Border Guards"],
    "Land Assets (Technical)": ["Tanks", "Armoured Vehicles", "Artillery", "Air-Defence", "Rocket", "APC", "Trucks", "Convoy"],
    "Land Assets (Physical)": ["Camp", "Tents", "roadblock", "checkpoint", "vehicle inspection", "border"],
    "Maritime Assets": ["Ship", "Warship", "Destroyer", "Submarine", "Battleship", "Aircraft Carrier", "Minesweeper", "Frigate", "Corvette"],
    "Air Assets": ["Jet", "Fighter", "Helicopter", "Bomber", "Drone", "Gunship", "Plane", "Aircraft"],
    "CRBN": ["Chemical", "Biological", "Radiological", "Nuclear", "CBRN", "non-conventional", "WMD", "weapons of mass destruction", "terror weapons"],
    "Combat": ["Battle", "skirmish", "fighting", "conflict", "combat", "war", "assault", "attack", "bombardment", "shelling", "bombing", "advance", "defeat", "victory"],
    "Political structures": ["Town hall", "Parliament", "Congress", "Senate", "Legislature", "Headquarters", "HQ"],
    "Legal structures": ["Courts", "magistrates", "municipal", "supreme court"],
    "Major structures": ["Stadium", "airport", "port", "docks", "train station", "landmark", "prison"],
    "Emergency structures": ["Hospital", "clinic", "medical facility", "field hospital", "firestation", "police station", "precinct"],
    "Economic Structures": ["Bank", "market", "stock exchange", "stock market", "vault", "bullion", "factory", "production", "warehouse", "workshop", "storage facility"],
    "Social Structures": ["Church", "Mosque", "Synagogue", "Temple", "Restaurant", "Bar", "Shopping Centre", "Shop", "Mall", "Park"],
    "Information Structures": ["Cell tower", "radio tower", "antennae", "TV tower", "server building", "internet café", "print shop", "news offices"],
    "Infrastructure": ["Roads", "Trainlines", "Tracks", "Bridge", "Dams", "Power line", "wall", "underground", "canals", "tunnels"],
    "Heritage Site": ["memorial", "heritage site"],
    "Responders": ["Police", "Fire", "Ambulance", "Specialists", "Officers", "Paramedics"],
    "NGO": ["NGO", "non-governmental organisation", "human rights group", "environment group", "activists", "charity", "private sector"],
    "Groups": ["Tribes", "Clans", "Community", "collective", "commune"],
    "Legal": ["Court", "Judge", "Supreme Court Justice", "Barrister", "Solicitor", "Attorney", "Attorney-General", "Jury"],
    "Media": ["News", "media", "press", "journalists", "reporter"],
    "Refugees": ["Refugees", "migrants", "immigrants", "asylum seekers", "stateless", "IDPs", "displaced"],
    "Fire": ["Fire", "Flame", "Blaze", "Wildfire", "Forest Fire", "Burning", "Smoke", "Ash"],
    "Explosion": ["Explosion", "Blast", "Detonation", "Bang", "Blow up", "Boom"],
    "Shooting": ["Shooting", "shots", "gunshots", "gun", "shots fired", "armed", "shooter"],
    "Traffic Incident": ["Crash", "Vehicle Fire", "Collision", "Pile-up", "Jack-knife", "traffic accident", "breakdown"],
    "Power": ["Power outage", "power cut", "blackout"],
    "Targeted": ["Assassination", "Kidnap", "ransom", "sabotage", "targeted", "torture", "imprisonment"],
    "Hurricane": ["Hurricane", "Storm", "Strong Winds", "Gale", "Tempest", "Tornado", "Squall", "Typhoon", "Disaster"],
    "Tsunami": ["Tsunami", "Tidal Wave", "Wave", "Disaster"],
    "Earthquake": ["Earthquake", "Tremor", "Aftershock", "Shake", "Convulsion", "Quake", "Seismic"],
    "Flood": ["Flood", "Torrent", "Landslide", "Disaster"],
    "Cyber": ["Cyber", "hack", "hacker", "malware", "breach", "data", "server"],
    "Phone": ["Phone", "telephone", "cell phone", "call"],
    "Disinformation": ["Disinformation", "Misinformation", "hostile messaging", "suspicious", "inauthentic behaviour", "bots", "information laundering"],
    "Law": ["Law", "legal", "legislature", "courtroom", "court decision", "ruling", "doctrine", "decision"]
}

# Load stop words from nltk
STOP_WORDS = set(stopwords.words('english'))

# Initialize PorterStemmer
stemmer = PorterStemmer()

# Preprocess crisis categories to store stemmed keywords
STEMMED_CRISIS_CATEGORIES = {
    category: [stemmer.stem(word.lower()) for word in keywords]
    for category, keywords in CRISIS_CATEGORIES.items()
}

def main():
    st.title("Crisis-Themed Word Categorization")

    # Text input for messages
    st.subheader("Input Your Messages")
    messages = st.text_area(
        "Paste your collection of messages below (one per line):",
        placeholder="Enter your messages here..."
    )

    if st.button("Categorize"):
        if messages.strip():
            # Combine input into one text blob
            text_blob = " ".join(messages.splitlines())

            # Preprocess the text and categorize words
            words = preprocess_text(text_blob)
            words = remove_stop_words(words)
            categorized_words = categorize_by_crisis_theme(words)

            # Remove empty categories
            categorized_words = {category: words for category, words in categorized_words.items() if words}

            # Convert results into a table
            category_table = create_category_table(categorized_words)

            # Display the categorized words
            st.subheader("Categorized Words")
            st.dataframe(category_table)

            # Display a warning with the top 100 words in Miscellaneous
            if "Miscellaneous" in categorized_words:
                display_miscellaneous_warning(categorized_words["Miscellaneous"])

            # Provide a download button
            st.download_button(
                label="Download Results as CSV",
                data=category_table.to_csv(index=False),
                file_name="categorized_words.csv",
                mime="text/csv",
            )

        else:
            st.error("Please enter some messages to analyze.")

def preprocess_text(text):
    """Clean and tokenize text."""
    # Remove special characters and digits, keep only words
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    cleaned_text = re.sub(r"\d+", "", cleaned_text)

    # Convert to lowercase and split into words
    words = cleaned_text.lower().split()

    return words

def remove_stop_words(words):
    """Remove stop words from the list of words."""
    return [word for word in words if word not in STOP_WORDS]

def categorize_by_crisis_theme(words):
    """Categorize words into predefined crisis themes."""
    categories = defaultdict(list)

    # Stem each word in the input
    stemmed_words = [stemmer.stem(word) for word in words]

    for word, stemmed_word in zip(words, stemmed_words):
        categorized = False
        for category, stemmed_keywords in STEMMED_CRISIS_CATEGORIES.items():
            if stemmed_word in stemmed_keywords:  # Match with stemmed keywords
                categories[category].append(word)
                categorized = True
                break
        if not categorized:
            categories["Miscellaneous"].append(word)

    return categories

def create_category_table(categorized_words):
    """Create a DataFrame from categorized words."""
    data = {
        "Category": [],
        "Words": []
    }

    # Ensure categories appear in the same order as in CRISIS_CATEGORIES
    for category in list(CRISIS_CATEGORIES.keys()) + ["Miscellaneous"]:
        if category in categorized_words:
            data["Category"].append(category)
            data["Words"].append(", ".join(sorted(set(categorized_words[category]))))  # Deduplicate and sort words

    return pd.DataFrame(data)

def display_miscellaneous_warning(miscellaneous_words):
    """Display a warning with the 100 most popular words in Miscellaneous."""
    word_counts = Counter(miscellaneous_words)
    top_100_words = word_counts.most_common(100)

    if top_100_words:
        st.warning(
            f"The following are the 100 most common words in the 'Miscellaneous' category:\n"
            + ", ".join([f"{word} ({count})" for word, count in top_100_words])
        )

if __name__ == "__main__":
    main()


