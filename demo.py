# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
import re
import os
from dotenv import load_dotenv
import seaborn as sns
from sqlalchemy import create_engine
from pymongo import MongoClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model='gemma2-9b-it', groq_api_key=GROQ_API_KEY)

# === Utility Functions ===
def get_df_info(df):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.info()
    return buffer.getvalue()

def clean_code(response):
    if "```" in response:
        return re.sub(r"```.*?\n([\s\S]*?)```", r"\1", response).strip()
    return response.strip()

def execute_pandas_code(code, df):
    local_vars = {'df': df, 'plt': plt, 'pd': pd, 'np': np}
    try:
        lines = code.strip().split('\n')
        *body, last = lines
        exec("\n".join(body), {}, local_vars)
        return eval(last, {}, local_vars)
    except Exception as e:
        return f"❌ Error: {e}"

# === Streamlit Page Setup ===
st.set_page_config(page_title="Multi-Source Data Analyst", layout="wide")

# App Header
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style='font-size: 42px;'>🤖 AI-Powered Data Analyst</h1>
        <p style='font-size:18px; color:gray;'>Choose your data source, upload or connect, and ask anything!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# === Sidebar for Data Source Selection ===
st.sidebar.image("https://verpex.com/assets/uploads/images/blog/How-To-Become-A-Freelance-Data-Analyst.webp?v=1711104899", width=160)
st.sidebar.markdown("### 📂 Data Source")

data_source = st.sidebar.radio(
    "Where is your data?",
    ["📁 Local CSV", "🛢️ SQL Database", "🍃 MongoDB"]
)

df = None  # Placeholder for final DataFrame

# === Local CSV Upload ===
if data_source.startswith("📁"):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.toast("✅ CSV file loaded!")

# === SQL Connection ===
elif data_source.startswith("🛢️"):
    with st.expander("🔐 Enter SQL Credentials", expanded=True):
        host = st.text_input("Host", "localhost")
        port = st.text_input("Port", "3306")
        user = st.text_input("Username", "root")
        password = st.text_input("Password", type="password")
        dbname = st.text_input("Database Name")
        table = st.text_input("Table Name")
        connect = st.button("Connect to SQL")

        if connect:
            try:
                engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}")
                df = pd.read_sql_table(table, con=engine)
                st.toast("✅ SQL connection successful!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ SQL Connection failed: {e}")

# === MongoDB Connection ===
elif data_source.startswith("🍃"):
    with st.expander("🔐 Enter MongoDB Details", expanded=True):
        uri = st.text_input("Mongo URI", "mongodb://localhost:27017")
        dbname = st.text_input("Database Name")
        collection = st.text_input("Collection Name")
        connect = st.button("Connect to MongoDB")

        if connect:
            try:
                client = MongoClient(uri)
                db = client[dbname]
                data = list(db[collection].find())
                df = pd.DataFrame(data)
                st.toast("✅ MongoDB connected!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ MongoDB error: {e}")

# === Main LLM-Powered Analysis Panel ===
if df is not None:
    st.dataframe(df.head(), use_container_width=True)
    st.caption("📌 Columns: " + ", ".join(df.columns))

    with st.expander("🧾 Dataset Summary"):
        st.text(get_df_info(df))
        st.write(df.describe(include='all'))

    head = df.head().to_string(index=False)
    info = get_df_info(df)
    describe = df.describe(include='all').to_string()

    prompt = ChatPromptTemplate.from_template("""
You are a Python data analyst.

CSV Data Preview:
{head}

CSV Info:
{info}

Statistical Summary:
{describe}

User asks: "{user_query}"

Write valid Pandas (or matplotlib) code using this data to answer the question.
Assume the DataFrame is named 'df'.
Only return code. If the question asks for visualization, use matplotlib.
""")

    user_query = st.text_input("💬 Ask a question about the data (e.g., 'plot sales trend', 'top 5 by price')")

    if user_query:
        chain = prompt | llm | StrOutputParser()
        with st.spinner("💡 Thinking..."):
            response = chain.invoke({
                "head": head,
                "info": info,
                "describe": describe,
                "user_query": user_query
            })

        code = clean_code(response)
        st.code(code, language="python")

        st.markdown("### 📊 Result")
        result = execute_pandas_code(code, df)

        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        elif isinstance(result, (int, float, str)):
            st.success(f"✅ Result: {result}")
        elif "plt" in code:
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            st.write(result)
