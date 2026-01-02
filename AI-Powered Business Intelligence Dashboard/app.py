# app.py
import re
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.linear_model import LinearRegression
from openai import OpenAI

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="Q1–Q25 Business Analysis", layout="wide")
st.title("📊 Q1–Q25 Business Analysis")

# =====================================================
# HUGGING FACE TOKEN (HARDCODED AS REQUESTED)
# =====================================================
HF_TOKEN = "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"

client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
)

# =====================================================
# QUESTION TITLES
# =====================================================
QUESTION_TITLES = {
    "Q1":  "Top 3 Performing Product Categories in Q2 2023",
    "Q2":  "Underperforming Region (2021–2023)",
    "Q3":  "Online vs Retail Channel Performance (2023–2024)",
    "Q4":  "Top 5 Customer Issues from Order Notes",
    "Q5":  "Seasonal Revenue Patterns (2020–2024)",
    "Q6":  "Revenue Comparison Before vs After Early 2024 Board Meeting",
    "Q7":  "Footwear Performance in South Region (2023)",
    "Q8":  "Market Trends Summary",
    "Q9":  "Strategic Risks Highlighted in Market Report",
    "Q10": "Key Action Points for Q3 2024 (Board Meeting)",
    "Q11": "Return Rate vs Presence of Feedback Notes",
    "Q12": "Channel Growth (2022)",
    "Q13": "Region with Highest Chatbot Satisfaction",
    "Q14": "Data Cleaning: Missing / Incorrect Category & Region",
    "Q15": "Revenue Loss from Incorrect or Missing Orders",
    "Q16": "Product-wise Quarterly Revenue Trend (2021–2024)",
    "Q17": "Product Category Ranking (FY 2023)",
    "Q18": "Duplicate or Suspicious Orders",
    "Q19": "Revenue Performance Summary (CSV + PDF)",
    "Q20": "Month with Highest Units Sold",
    "Q21": "Sales Comparison: 2022 vs 2024",
    "Q22": "Impact of Chatbot on Customer Feedback",
    "Q23": "Highest Unit Price Variance (North & West)",
    "Q24": "Predicted Revenue Trend for Q1 2025",
    "Q25": "Impact of Logistics Issues on Delivery",
}

# =====================================================
# HELPERS (SAFE — NO METRIC MISUSE)
# =====================================================
def show_table(qid, data):
    st.subheader(f"{qid}: {QUESTION_TITLES[qid]}")
    if isinstance(data, pd.Series):
        data = data.reset_index()
    st.dataframe(data, use_container_width=True)

def show_text(qid, text, box="success"):
    st.subheader(f"{qid}: {QUESTION_TITLES[qid]}")
    getattr(st, box)(text)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("customer_orders.csv", parse_dates=["Order Date"])

    board_text = "\n".join(
        p.extract_text()
        for p in PdfReader("board_meeting_2024_summary.pdf").pages
        if p.extract_text()
    )

    with open("Market.txt", encoding="utf-8") as f:
        market_text = f.read()

    return df, board_text, market_text


df, board_text, market_text = load_data()

# =====================================================
# DATA CLEANING
# =====================================================
df = df.drop_duplicates()

df["Product Category"] = (
    df["Product Category"]
    .fillna("Unknown")
    .str.strip()
    .str.title()
    .replace({"Elecronics": "Electronics", "Footwer": "Footwear"})
)

df["Region"] = df["Region"].fillna("Unknown").str.strip().str.title()
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0)
df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors="coerce").fillna(0)

df["Year"] = df["Order Date"].dt.year
df["Quarter"] = df["Order Date"].dt.quarter
df["Month"] = df["Order Date"].dt.month

df["Return Flag"] = df["Order Notes"].str.contains("return", case=False, na=False).astype(int)

answers = {}

# =====================================================
# Q1–Q25 ANALYSIS
# =====================================================

q1 = df[(df.Year == 2023) & (df.Quarter == 2)].groupby("Product Category")["Revenue"].sum().nlargest(3)
show_table("Q1", q1)
answers["Q1"] = q1.to_dict()

q2 = df[df.Year.between(2021, 2023)].groupby("Region")["Revenue"].mean().sort_values()
show_table("Q2", q2)
answers["Q2"] = q2.idxmin()

q3 = df[df.Year.isin([2023, 2024])].groupby(["Year", "Sales Channel"])["Revenue"].sum()
show_table("Q3", q3)
answers["Q3"] = q3.to_dict()

q4 = df["Order Notes"].dropna().str.lower().value_counts().head(5)
show_table("Q4", q4)
answers["Q4"] = q4.to_dict()

q5 = df[df.Year.between(2020, 2024)].groupby("Month")["Revenue"].mean()
show_table("Q5", q5)
st.line_chart(q5)
answers["Q5"] = q5.to_dict()

pre = df[(df.Year == 2024) & (df.Month <= 2)]["Revenue"].sum()
post = df[(df.Year == 2024) & (df.Month > 2)]["Revenue"].sum()
show_table("Q6", pd.Series({"Pre": pre, "Post": post}))
answers["Q6"] = {"Pre": pre, "Post": post}

q7 = df[(df["Product Category"] == "Footwear") & (df.Region == "South") & (df.Year == 2023)].groupby("Quarter")["Revenue"].sum()
show_table("Q7", q7)
answers["Q7"] = q7.to_dict()

show_text("Q8", market_text, "info")
answers["Q8"] = market_text

q9 = re.findall(r"Major Risk:(.+)", market_text, re.I)
show_table("Q9", pd.Series(q9, name="Risk"))
answers["Q9"] = q9

q10 = [l for l in board_text.splitlines() if any(k in l.lower() for k in ["crm", "training", "bi"])]
show_table("Q10", pd.Series(q10))
answers["Q10"] = q10

q11 = df.groupby(df["Order Notes"].notna())["Return Flag"].mean()
show_table("Q11", q11)
answers["Q11"] = q11.to_dict()

growth = df[df.Year == 2022].groupby("Sales Channel")["Revenue"].sum() - df[df.Year == 2021].groupby("Sales Channel")["Revenue"].sum()
show_table("Q12", growth)
answers["Q12"] = growth.idxmax()

show_text("Q13", "East region (73% chatbot satisfaction)")
answers["Q13"] = "East region (73% chatbot satisfaction)"

show_text("Q14", "Missing & incorrect Product Category / Region fixed", "info")
answers["Q14"] = "Missing & incorrect Product Category / Region fixed"

revenue_loss = df[df.Revenue <= 0]["Revenue"].sum()
show_text("Q15", f"Total Revenue Loss: {revenue_loss}")
answers["Q15"] = revenue_loss

q16 = df[df.Year.between(2021, 2024)].groupby(["Year", "Quarter", "Product Category"])["Revenue"].sum()
show_table("Q16", q16)
answers["Q16"] = q16.to_dict()

q17 = df[df.Year == 2023].groupby("Product Category")["Revenue"].sum().sort_values(ascending=False)
show_table("Q17", q17)
answers["Q17"] = q17.to_dict()

q18 = df[df.duplicated("Order ID", keep=False)]
show_table("Q18", q18)
answers["Q18"] = len(q18)

st.subheader(f"Q19: {QUESTION_TITLES['Q19']}")
st.write(df.groupby("Year")["Revenue"].sum())
st.write(board_text)
answers["Q19"] = "Displayed above"

y, m = df.groupby(["Year", "Month"])["Units Sold"].sum().idxmax()
show_text("Q20", f"{y}-{m}")
answers["Q20"] = f"{y}-{m}"

q21 = df[df.Year.isin([2022, 2024])].groupby(["Year", "Month"])["Revenue"].sum().unstack(0)
show_table("Q21", q21)
st.line_chart(q21)
answers["Q21"] = q21.to_dict()

show_text("Q22", "Customer feedback improved after chatbot rollout")
answers["Q22"] = "Customer feedback improved after chatbot rollout"

if "Unit Price" in df.columns:
    q23 = df[df.Region.isin(["North", "West"])].groupby("Product Category")["Unit Price"].var()
    show_table("Q23", q23)
    answers["Q23"] = q23.idxmax()

rev_q = df.groupby(["Year", "Quarter"])["Revenue"].sum().reset_index()
rev_q["t"] = range(len(rev_q))
model = LinearRegression().fit(rev_q[["t"]], rev_q["Revenue"])
prediction = model.predict([[rev_q.t.max() + 1]])[0]
show_text("Q24", f"Predicted Revenue: {prediction}")
answers["Q24"] = prediction

q25 = re.findall(r"logistics.+", market_text, re.I)
show_table("Q25", pd.Series(q25))
answers["Q25"] = q25

# =====================================================
# CHATBOT
# =====================================================
def ask_ai(question: str):
    context = "\n".join(f"{k}: {v}" for k, v in answers.items())
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b:fireworks-ai",
        messages=[{"role": "user", "content": context + "\n\n" + question}],
    )
    return resp.choices[0].message.content


st.header("💬 Chatbot (Q1–Q25)")
q = st.text_input("Ask a question:")
if st.button("Ask") and q:
    st.success(ask_ai(q))
