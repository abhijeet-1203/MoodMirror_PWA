import pandas as pd
import os

DATA_PATH = "data/journal_entries.csv"

def load_entries():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame(columns=['Date', 'Entry', 'Sentiment', 'Score', 'Keywords'])

def save_entry(date, entry, sentiment, score, keywords):
    df = load_entries()
    new_entry = pd.DataFrame({
        'Date': [date],
        'Entry': [entry],
        'Sentiment': [sentiment],
        'Score': [score],
        'Keywords': [', '.join(keywords)]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(DATA_PATH, index=False) 

# Add to utils.py (at the end)
def export_to_pdf(df):
    """Export journal entries to PDF"""
    from fpdf import FPDF  # Import here to avoid dependency issues
    
    pdf = FPDF()
    pdf.add_page()
    
    # Add Unicode-compatible font (DejaVu Sans)
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.set_font('DejaVu', size=12)  # Set the font to DejaVu Sans

    # Header
    pdf.set_font(style='B')
    pdf.cell(200, 10, txt="Your MoodMirror Journal Entries", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font(style='')

    # Entries
    for _, row in df.iterrows():
        pdf.cell(200, 8, txt=f"📅 {row['Date']}", ln=True)
        pdf.multi_cell(0, 8, txt=f"• Mood: {row['Sentiment']} (Score: {row['Score']:.2f})")
        pdf.multi_cell(0, 8, txt=f"• Entry: {row['Entry']}")
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin1')
