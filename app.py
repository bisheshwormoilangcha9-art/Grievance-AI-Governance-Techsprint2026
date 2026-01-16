import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# =======================
# Streamlit Configuration
# =======================
st.set_page_config(
    page_title="GrievanceSense – AI Grievance Intelligence Platform",
    page_icon="🔔",
    layout="wide"
)

# =======================
# Sidebar
# =======================
st.sidebar.title("Portal Selection")
page = st.sidebar.radio(
    "Select Portal",
    ["Citizen Portal", "Official Dashboard"]
)

# =======================
# Load Model
# =======================
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# =======================
# Utility Functions
# =======================
def urgency_score(text):
    keywords = ["days", "accident", "sick", "emergency", "no water", "not working"]
    score = sum(1 for word in keywords if word in text.lower())
    if score >= 3:
        return "Critical"
    elif score == 2:
        return "High"
    else:
        return "Medium"

def credibility_score(text):
    score = 50
    words = text.lower().split()

    if len(words) > 15:
        score += 15
    elif len(words) > 8:
        score += 5

    time_words = ["day", "days", "hour", "hours", "week", "weeks", "month", "months", "year", "years"]
    if any(word in words for word in time_words):
        score += 10

    context_words = ["area", "road", "street", "hospital", "locality", "school"]
    if any(word in words for word in context_words):
        score += 10

    emotion_words = ["worst", "useless", "terrible", "very bad", "hate"]
    if any(word in words for word in emotion_words):
        score -= 10

    return max(30, min(score, 100))

# =======================
# App Title
# =======================
st.title("GrievanceSense – AI Grievance Intelligence Platform")

# =======================
# Citizen Portal
# =======================
if page == "Citizen Portal":

    st.header("🧍 Citizen Grievance Portal")

    complaint = st.text_area("Enter Citizen Complaint")
    area = st.selectbox("Select Area", ["Zone A", "Zone B", "Zone C"])

    # ---------- Analyze ----------
    if st.button("Analyze Complaint"):
        if complaint.strip() == "":
            st.warning("Please enter a complaint.")
        else:
            text_vec = vectorizer.transform([complaint])
            category = model.predict(text_vec)[0]
            urgency = urgency_score(complaint)
            credibility = credibility_score(complaint)

            st.subheader("AI Analysis Result")
            st.write("📌 Category:", category)
            st.write("⚠ Urgency Level:", urgency)
            st.write("✅ Credibility Score:", credibility, "/100")
            st.write("📍 Area:", area)

    # ---------- Submit ----------
    if st.button("Submit Complaint"):
        if complaint.strip() == "":
            st.warning("Please enter a complaint.")
        else:
            text_vec = vectorizer.transform([complaint])
            category = model.predict(text_vec)[0]
            urgency = urgency_score(complaint)
            credibility = credibility_score(complaint)

            new_row = pd.DataFrame([{
                "complaint_text": complaint,
                "category": category,
                "urgency": urgency,
                "credibility": credibility,
                "area": area
            }])

            os.makedirs("data", exist_ok=True)
            file_path = "data/submissions.csv"

            if not os.path.exists(file_path):
                new_row.to_csv(file_path, index=False)
            else:
                new_row.to_csv(file_path, mode="a", header=False, index=False)

            st.success("Complaint submitted successfully!")

# =======================
# Official Dashboard
# =======================
elif page == "Official Dashboard":

    st.header("🧑‍💼 Official Grievance Dashboard")

    file_path = "data/submissions.csv"

    if not os.path.exists(file_path):
        st.warning("No complaints submitted yet.")
        st.stop()

    data = pd.read_csv(file_path)
    required_cols = ["complaint_text", "category", "urgency", "credibility", "area"]
    if not all(col in data.columns for col in required_cols):
        st.error("Invalid CSV structure.")
        st.stop()

    # ---------- Top KPIs ----------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Complaints", len(data))
    col2.metric("Critical", (data["urgency"] == "Critical").sum())
    col3.metric("High", (data["urgency"] == "High").sum())
    col4.metric("Medium", (data["urgency"] == "Medium").sum())

    st.divider()

    # ---------- Charts Section ----------
    st.subheader("📊 Charts Overview")

    # Row 1: Category + Area
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.caption("Complaints by Category")
        category_count = data["category"].dropna().value_counts()
        fig, ax = plt.subplots(figsize=(5,3))
        if category_count.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            category_count.plot(kind="bar", ax=ax)
            ax.set_xlabel("Category")
            ax.set_ylabel("Number of Complaints")
        st.pyplot(fig, use_container_width=True)

    with row1_col2:
        st.caption("Complaints by Area")
        area_count = data["area"].dropna().value_counts()
        fig, ax = plt.subplots(figsize=(5,3))
        if area_count.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            area_count.plot(kind="bar", ax=ax)
            ax.set_xlabel("Area")
            ax.set_ylabel("Number of Complaints")
        st.pyplot(fig, use_container_width=True)

    # Row 2: Urgency + Critical Complaints
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.caption("Urgency Distribution")
        urgency_count = data["urgency"].dropna().value_counts()
        fig, ax = plt.subplots(figsize=(5,3))
        if urgency_count.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            urgency_count.plot(kind="bar", ax=ax)
            ax.set_xlabel("Urgency Level")
            ax.set_ylabel("Number of Complaints")
        st.pyplot(fig, use_container_width=True)

    with row2_col2:
        st.caption("Critical Complaints")
        critical_df = data[data["urgency"] == "Critical"]
        if critical_df.empty:
            st.info("No critical complaints.")
        else:
            st.dataframe(critical_df, use_container_width=True)

    st.divider()

    # ---------- All Submissions ----------
    with st.expander("📋 View All Complaints"):
        st.dataframe(data, use_container_width=True)
