mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"you@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

python -m nltk_setup.py
python -m textblob.download_corpora
