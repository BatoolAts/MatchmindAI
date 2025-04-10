import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gtts import gTTS
import streamlit as st
import base64
import random

# Load background image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

img_base64 = get_base64_image("image.png")

# Apply custom style for white text and background image
st.set_page_config(page_title="MatchMind AI", layout="centered")
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.9);  
            padding: 2rem;
            border-radius: 10px;
        }}
        html, body, [class*="css"] {{
            font-family: 'Arial', sans-serif;
            color: #ffffff !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #ffffff !important;
            text-shadow: 1px 1px 3px #000000;
        }}
        .stMarkdown, .stText, .stSubheader {{
            color: #ffffff !important;
        }}
        /* Metric component text */
        [data-testid="stMetric"] div, 
        [data-testid="stMetricLabel"] {{
            color: #ffffff !important;
            text-shadow: 1px 1px 2px #000000;
        }}
        /* Streamlit default labels and widget text */
        label, .stSlider label, .stSelectbox label, .stRadio label {{
            color: #ffffff !important;
            font-weight: 500;
        }}
        .css-1y4p8pa, .css-10trblm, .css-1cpxqw2 {{
            color: #ffffff !important;
        }}
    </style>
""", unsafe_allow_html=True)



st.markdown("<div class='title'>MatchMind AI - Smart Football Commentator</div>", unsafe_allow_html=True)
st.write("Tactical match analysis and voice commentary based on match outcome.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("Merged_Dataset_Cleaned_Optimized (1).xlsx")
    df = df.dropna(subset=[
        'home_team_goal', 'away_team_goal', 'stage',
        'Team_Points', 'Opponent_Points', 'Team_Score', 'Opponent_Score'
    ])
    return df

df = load_data()

# Label results
def label_result(row):
    if row['Team_Score'] > row['Opponent_Score']:
        return 2  # Win
    elif row['Team_Score'] < row['Opponent_Score']:
        return 0  # Loss
    else:
        return 1  # Draw

df['Result'] = df.apply(label_result, axis=1)

df = df.sort_values(by='season.1')
features = ['home_team_goal', 'away_team_goal', 'stage', 'Team_Points', 'Opponent_Points']


split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]


X_train = train_df[features]
y_train = train_df['Result']
X_test = test_df[features]
y_test = test_df['Result']

# Prepare train/test
X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    max_depth=3,  
    learning_rate=0.05,
    n_estimators=100,  
    subsample=0.7,     
    colsample_bytree=0.7,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Train XGBoost model
model.fit(
    X_train_main, y_train_main,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred = model.predict(X_test)
# Dictionary of real-like players per team
team_players = {
    "BARCELONA": ["Messi", "Xavi", "Iniesta", "Puyol", "Henry"],
    "REAL MADRID": ["Raul", "Robben", "Higuain", "Casillas", "Sneijder"],
    "VALENCIA": ["Villa", "Silva", "Marchena", "Joaquin", "Mata"],
    "SEVILLA": ["Kanoute", "Fabiano", "Navas", "Escude", "Palop"],
    "VILLARREAL": ["Rossi", "Senna", "Cazorla", "Godin", "Capdevila"],
    "ATLETICO MADRID": ["Forlan", "Aguero", "Simao", "Maxi", "Perea"],
    "ESPANYOL": ["Tamudo", "Luis Garcia", "De la Peña", "Jarque", "Kameni"],
    "MALLORCA": ["Arango", "Webo", "Aduriz", "Ayoze", "Lux"],
    "MALAGA": ["Baha", "Eliseu", "Lolo", "Goitia", "Gámez"],
    "DEPORTIVO LA CORUNA": ["Guardado", "Lafita", "Coloccini", "De Guzman", "Aranzubia"],
    "REAL SOCIEDAD": ["Xabi Prieto", "Gabilondo", "Ansotegi", "Bravo", "Agirretxe"],
    "ATHLETIC BILBAO": ["Llorente", "Yeste", "Susaeta", "Iraola", "Gorka"],
    "BETIS": ["Edu", "Mark Gonzalez", "Rivas", "Melli", "Casto"],
    "RACING SANTANDER": ["Tchite", "Colsa", "Serrano", "Pinillos", "Toño"],
    "OSASUNA": ["Pandiani", "Juanfran", "Sergio", "Puñal", "Ricardo"],
    "GETAFE": ["Soldado", "Albín", "Contra", "Granero", "Ustari"],
    "ALMERIA": ["Negredo", "Corona", "Ortiz", "Carlos Garcia", "Diego Alves"],
    "NUMANCIA": ["Goiria", "Barkero", "Culebras", "Gorka Brit", "Juanra"],
    "VALLADOLID": ["Sesma", "Canobbio", "Víctor", "Pedro León", "Asenjo"],
    "RECREATIVO HUELVA": ["Camuñas", "Aitor", "Javi Fuego", "Bouzon", "Riesgo"],
    "ZARAGOZA": ["Ewerthon", "Ayala", "Ponzio", "Zapater", "Carrizo"],
    "ALAVES": ["Bodipo", "Tomic", "Astudillo", "Pellegrino", "Herrera"],
    "MURCIA": ["Baiano", "Movilla", "Pena", "Iñaki Bea", "Notario"],
    "RAYO VALLECANO": ["Piti", "Michu", "Coke", "Lahcen", "Cobeño"]
}

# User selects match
st.subheader("Select a match to analyze:")
index = st.slider("Match Index", min_value=0, max_value=len(test_df)-1, value=0)
sample_row = test_df.iloc[index]
sample_input = sample_row[features].values.reshape(1, -1)
prediction = model.predict(sample_input)[0]

# Select commentary style
style = st.selectbox("Choose Commentary Style", ["Energetic", "Analytical", "Dramatic"])

# Generate simulated player stats
team = sample_row['Team']
opponent = sample_row['Opponent']
ts = int(sample_row['Team_Score'])
oscore = int(sample_row['Opponent_Score'])

players = team_players.get(team.upper(), ["Top Player"])
top_scorer = random.choice(players)

def generate_player_stats():
    goals = random.randint(0, 3)
    assists = random.randint(0, goals)
    tackles = random.randint(3, 10)
    saves = random.randint(0, 5)
    return {
        "Top Scorer": top_scorer,
        "Goals": goals,
        "Assists": assists,
        "Tackles": tackles,
        "Saves": saves
    }

player_stats = generate_player_stats()

# Commentary generation
if prediction == 2:
    commentary_base = f"{team} defeated {opponent} {ts}-{oscore}."
elif prediction == 0:
    commentary_base = f"{team} lost to {opponent} {ts}-{oscore}."
else:
    commentary_base = f"{team} and {opponent} drew {ts}-{oscore}."

if style == "Energetic":
    commentary = f"🔥 {commentary_base} An exciting game full of pace and passion! {player_stats['Top Scorer']} was unstoppable today!"
elif style == "Analytical":
    commentary = f"{commentary_base} Tactical formations and smart plays defined this match. Key tackles: {player_stats['Tackles']}, Saves: {player_stats['Saves']} by {player_stats['Top Scorer']}."
else:
    commentary = f"What a battle! {commentary_base} Drama unfolded on the pitch. {player_stats['Top Scorer']} brought the crowd to their feet!"

# Text to speech
audio_file = "match_commentary_en.mp3"
tts = gTTS(commentary, lang='en')
tts.save(audio_file)

# Display prediction
st.subheader("Predicted Match Result")
result_map = {0: "Loss", 1: "Draw", 2: "Win"}
st.success(f"{team} is predicted to: {result_map[prediction]}")

# Display commentary
st.subheader("AI Commentary")
st.write(commentary)
st.audio(audio_file)

# Match statistics
st.subheader("Match Statistics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Team", team)
    st.metric("Team Score", ts)
    st.metric("Team Points", int(sample_row['Team_Points']))
    st.metric("Location", sample_row['Location'])

with col2:
    st.metric("Opponent", opponent)
    st.metric("Opponent Score", oscore)
    st.metric("Stage", int(sample_row['stage']))

# Player Stats
st.subheader("Key Player Stats")
st.dataframe(pd.DataFrame([player_stats]))

# Match Timeline
st.subheader("Match Event Timeline")
events = []
for _ in range(random.randint(5, 8)):
    minute = random.randint(1, 90)
    event_type = random.choice(["⚽ Goal", "🟨 Yellow Card", "🟥 Red Card", "🧤 Save", "🔁 Substitution"])
    player = random.choice(players)
    events.append((minute, f"{minute}' {event_type} - {player}"))

# Sort by time
events.sort(key=lambda x: x[0])
for _, desc in events:
    st.markdown(f"- {desc}")
    
    # Suggested Tactical Strategy
st.subheader("AI Suggested Strategy for Next Match")
if prediction == 0:
    st.warning("Focus on tightening the defense and improving midfield control.")
elif prediction == 2:
    st.success("Keep your current attacking style. It’s working!")
else:
    st.info("Increase pressure in the final third and try more shots on goal.")
