FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r law-chatbot-langchain/08_streamlit_app/requirements.txt

WORKDIR /app/law-chatbot-langchain/08_streamlit_app

ENV HOME=/app
EXPOSE 7860

CMD ["chainlit", "run", "appchainlit_search_v2.py", "--host", "0.0.0.0", "--port", "7860", "--headless"]
