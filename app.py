import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
from sentiment_analysis import analyze_sentiment_vader, get_keywords
from utils import save_entry, load_entries
import plotly.express as px 
import random  
from utils import export_to_pdf
import streamlit as st
from textblob import TextBlob
import nltk
import os

# Set a persistent path for NLTK downloads
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download essential corpora
for resource in ["punkt", "averaged_perceptron_tagger", "wordnet", "brown"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)



st.set_page_config(page_title="MoodMirror", page_icon="🪞", layout="centered")

# Inject manifest and service worker
st.markdown("""
<link rel="manifest" href="/static/manifest.json">
<script>
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/static/service-worker.js")
  }
</script>
""", unsafe_allow_html=True)


# Add cursor CSS immediately after
st.markdown("""
<style>
    /* Nuclear option - forces pointer cursor on ALL sidebar hover states */
    .stSidebar *:hover {
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 MoodMirror - AI Mental Health Journal")

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["New Entry", "View Emotional Trends", "WordCloud"])


if page == "New Entry":
    st.subheader("Write Your Journal Entry")

    journal_text = st.text_area("Today's Thoughts...", height=300)
    if st.button("Analyze and Save"):
        if journal_text.strip() != "":
            sentiment, score = analyze_sentiment_vader(journal_text)
            keywords = get_keywords(journal_text)
            save_entry(datetime.now().strftime("%Y-%m-%d"), journal_text, sentiment, score, keywords)
            st.success(f"Entry saved! Detected sentiment: **{sentiment}** (Score: {score:.2f})")
        else:
            st.warning("Please write something before saving!")


elif page == "View Emotional Trends":
    st.subheader("Your Emotional Trends Over Time")
    df = load_entries()
    
    if not df.empty:
        # Convert dates and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # --- [1. LINE CHART] --- (Keep your existing time series plot)
        fig_line = plt.figure(figsize=(10, 5))
        sns.lineplot(x='Date', y='Score', data=df, marker='o')
        plt.ylim(-1, 1)
        plt.axhline(0, color='gray', linestyle='--')
        st.pyplot(fig_line)

         # --- [2. ROLLING AVERAGE LINE CHART] ---
        st.subheader("7-Day Rolling Average of Sentiment")
        
        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Drop rows where 'Date' is NaT (invalid or missing)
        df = df.dropna(subset=['Date'])

        # Sort by date to ensure proper rolling behavior
        df = df.sort_values('Date')

        # Calculate 7-day rolling average
        rolling_avg = df.set_index('Date')['Score'].rolling('7D').mean()

        # Plot the rolling average
        st.line_chart(rolling_avg, use_container_width=True)

        
        # --- [3. NEW PIE CHART] ---
        st.subheader("Mood Distribution")
        mood_counts = df['Sentiment'].value_counts()

        
        # Create interactive pie chart
        fig_pie = px.pie(
            mood_counts,
            names=mood_counts.index,
            values=mood_counts.values,
            color=mood_counts.index,
            color_discrete_map={
                'Positive': '#4CAF50',  # Green
                'Neutral': '#FFC107',    # Amber
                'Negative': '#F44336'    # Red
            },
            hole=0.3  # Creates a donut chart
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
                # --- [MOOD STATS] ---
        st.subheader("Your Mood Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Entries", len(df))
        col2.metric("Most Common Mood", df['Sentiment'].mode()[0])
        col3.metric("Positivity Ratio", f"{(mood_counts.get('Positive', 0) / len(df) * 100):.1f}%")
        col4.metric("Avg. Sentiment Score", f"{df['Score'].mean():.2f}")


elif page == "WordCloud":
    st.subheader("Visualize Your Frequent Thoughts")
    df = load_entries()
    
    if not df.empty:
        # Sentiment filter dropdown
        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All"] + list(df['Sentiment'].unique())
        )
        
        # Filter the dataframe by selected sentiment
        if sentiment_filter != "All":
            filtered_df = df[df['Sentiment'] == sentiment_filter]
            text = " ".join(filtered_df['Entry'])
        else:
            text = " ".join(df['Entry'].dropna().astype(str))

        
        # Generate and display the WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Show the keyword frequency table
        st.subheader("Keyword Frequency")
        keywords = " ".join(df['Keywords'].fillna("").astype(str)).split(", ")
        freq = pd.Series(keywords).value_counts()
        st.dataframe(freq.head(10))
    else:
        st.info("No entries yet to generate a WordCloud.")

# ADDITIONAL FEATURES SECTION (ADD THIS AT THE END)

# --- Writing Prompts ---

st.sidebar.markdown("---")  # Visual separator
with st.sidebar.expander("💡 Writing Prompts"):
    if st.button("Get Random Prompt"):
        prompts = [
            "What made you smile today?",
            "What challenge did you overcome?",
            "What are you grateful for right now?",
            "Describe a moment you felt proud."
        ]
        st.session_state.prompt = random.choice(prompts)
    
    if 'prompt' in st.session_state:
        st.markdown(f"**Your Prompt:**\n\n{st.session_state.prompt}")
        if st.button("Clear Prompt"):
            del st.session_state.prompt

# --- PDF Export ---

st.sidebar.markdown("---")  # Visual separator
with st.sidebar.expander("📤 Export Journal"):
    if st.button("Generate PDF Report"):
        try:
            with st.spinner("Creating PDF..."):
                pdf_bytes = export_to_pdf(load_entries())
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=f"mood_journal_{datetime.now().date()}.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Export failed: {e}")

# [End of file - nothing should come after]
