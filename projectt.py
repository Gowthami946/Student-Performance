import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Result Prediction System")
st.write("Machine Learning Web Application using Python and Streamlit")


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv(
    r"C:\Users\Bestha\OneDrive\Desktop\Student_Result_Prediction\student_exam_scores_10000.csv"
)

# Remove missing values
df = df.dropna()

# Add assessment_done column
# Values: 0 to 10
np.random.seed(42)
df["assessment_done"] = np.random.randint(0, 11, len(df))

# Create Result column from exam_score
# 1 = Pass, 0 = Fail
df["Result"] = np.where(df["exam_score"] >= 35, 1, 0)

# For graphs
df["Final_Marks"] = df["exam_score"]


# -------------------------------------------------
# Model Training
# -------------------------------------------------
X = df[[
    "hours_studied",
    "attendance_percent",
    "previous_scores",
    "assessment_done",
    "sleep_hours"
]]

y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


# -------------------------------------------------
# Model Performance Metrics
# -------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)


# -------------------------------------------------
# Create Tabs
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Result",
    "📊 Data Analysis",
    "📈 Model Performance"
])


# -------------------------------------------------
# Tab 1: Predict Result
# -------------------------------------------------
with tab1:
    st.header("Predict Student Result")

    col1, col2 = st.columns(2)

    with col1:
        hours = st.slider(
            "Hours Studied Per Day",
            min_value=1.0,
            max_value=12.0,
            value=6.0,
            step=0.1
        )

        attendance = st.slider(
            "Attendance Percentage",
            min_value=50.0,
            max_value=100.0,
            value=75.0,
            step=0.1
        )

        previous = st.slider(
            "Previous Scores",
            min_value=40.0,
            max_value=95.0,
            value=68.0,
            step=0.1
        )

    with col2:
        assessment = st.slider(
            "Assessments Done",
            min_value=0,
            max_value=10,
            value=5,
            step=1
        )

        sleep = st.slider(
            "Sleep Hours Per Day",
            min_value=4.0,
            max_value=9.0,
            value=7.0,
            step=0.1
        )

    if st.button("Predict Result"):
        input_data = pd.DataFrame({
            "hours_studied": [hours],
            "attendance_percent": [attendance],
            "previous_scores": [previous],
            "assessment_done": [assessment],
            "sleep_hours": [sleep]
        })

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        fail_probability = probability[0] * 100
        pass_probability = probability[1] * 100

        st.subheader("Prediction Output")

        if prediction == 1:
            st.success("✅ Student is likely to PASS")
            st.write(f"Confidence: **{pass_probability:.2f}%**")
        else:
            st.error("❌ Student is likely to FAIL")
            st.write(f"Confidence: **{fail_probability:.2f}%**")

        st.subheader("Input Summary")
        st.dataframe(input_data, use_container_width=True)


# -------------------------------------------------
# Tab 2: Data Analysis
# -------------------------------------------------
with tab2:
    st.header("Dataset Analysis")

    total_students = len(df)
    pass_count = df[df["Result"] == 1].shape[0]
    fail_count = df[df["Result"] == 0].shape[0]
    pass_rate = (pass_count / total_students) * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total_students)
    c2.metric("Pass Count", pass_count)
    c3.metric("Fail Count", fail_count)
    c4.metric("Pass Rate", f"{pass_rate:.2f}%")

    st.subheader("Sample Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.subheader("Hours Studied vs Exam Score")

        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.scatter(df["hours_studied"], df["exam_score"])
        ax1.set_xlabel("Hours Studied")
        ax1.set_ylabel("Exam Score")
        ax1.set_title("Hours Studied vs Exam Score")
        st.pyplot(fig1)

    with graph_col2:
        st.subheader("Attendance vs Exam Score")

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.scatter(df["attendance_percent"], df["exam_score"])
        ax2.set_xlabel("Attendance Percentage")
        ax2.set_ylabel("Exam Score")
        ax2.set_title("Attendance vs Exam Score")
        st.pyplot(fig2)

    graph_col3, graph_col4 = st.columns(2)

    with graph_col3:
        st.subheader("Assessments Done vs Exam Score")

        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.scatter(df["assessment_done"], df["exam_score"])
        ax3.set_xlabel("Assessments Done")
        ax3.set_ylabel("Exam Score")
        ax3.set_title("Assessments Done vs Exam Score")
        st.pyplot(fig3)

    with graph_col4:
        st.subheader("Pass / Fail Distribution")

        result_counts = df["Result"].value_counts().sort_index()
        labels = ["Fail", "Pass"]

        fig4, ax4 = plt.subplots(figsize=(4, 3))
        ax4.pie(result_counts, labels=labels, autopct="%1.1f%%")
        ax4.set_title("Pass / Fail Distribution")
        st.pyplot(fig4)

    graph_col5, graph_col6 = st.columns(2)

    with graph_col5:
        st.subheader("Previous Scores vs Exam Score")

        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5.scatter(df["previous_scores"], df["exam_score"])
        ax5.set_xlabel("Previous Scores")
        ax5.set_ylabel("Exam Score")
        ax5.set_title("Previous Scores vs Exam Score")
        st.pyplot(fig5)

    with graph_col6:
        st.subheader("Sleep Hours vs Exam Score")

        fig6, ax6 = plt.subplots(figsize=(5, 3))
        ax6.scatter(df["sleep_hours"], df["exam_score"])
        ax6.set_xlabel("Sleep Hours")
        ax6.set_ylabel("Exam Score")
        ax6.set_title("Sleep Hours vs Exam Score")
        st.pyplot(fig6)

    st.subheader("Exam Score Distribution")

    fig7, ax7 = plt.subplots(figsize=(6, 3))
    ax7.hist(df["exam_score"], bins=20)
    ax7.set_xlabel("Exam Score")
    ax7.set_ylabel("Number of Students")
    ax7.set_title("Exam Score Distribution")
    st.pyplot(fig7)


# -------------------------------------------------
# Tab 3: Model Performance
# -------------------------------------------------
with tab3:
    st.header("Model Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    m2.metric("Precision", f"{precision * 100:.2f}%")
    m3.metric("Recall", f"{recall * 100:.2f}%")
    m4.metric("F1 Score", f"{f1 * 100:.2f}%")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.subheader("Confusion Matrix")

        fig8, ax8 = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"],
            ax=ax8
        )
        ax8.set_xlabel("Predicted")
        ax8.set_ylabel("Actual")
        ax8.set_title("Confusion Matrix")
        st.pyplot(fig8)

    with perf_col2:
        st.subheader("Feature Importance")

        importance = model.coef_[0]

        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        })

        fig9, ax9 = plt.subplots(figsize=(5, 3))
        ax9.bar(feature_df["Feature"], feature_df["Importance"])
        ax9.set_xlabel("Features")
        ax9.set_ylabel("Importance")
        ax9.set_title("Feature Importance")
        plt.xticks(rotation=30)
        st.pyplot(fig9)

    st.subheader("Metric Explanation")

    st.write("""
    **Accuracy** tells how many total predictions are correct.

    **Precision** tells: when the model says PASS, how often it is correct.

    **Recall** tells: out of all actual PASS students, how many were correctly predicted.

    **F1 Score** is the balance between Precision and Recall.
    """)