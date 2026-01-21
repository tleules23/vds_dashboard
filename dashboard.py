import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Titanic Data Explorer", layout="wide")

@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")

    # Basic cleaning
    train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())

    train["Title"] = train["Name"].str.extract(r',\s*([^.]+)\.')[0]
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Mrs",
        "the Countess": "Mrs", "Dona": "Mrs", "Sir": "Mr", "Jonkheer": "Mr",
        "Don": "Mr", "Col": "Officer", "Major": "Officer", "Capt": "Officer",
        "Rev": "Officer", "Dr": "Officer"
    }
    train["Title"] = train["Title"].replace(title_map)

    train["AgeGroup"] = pd.cut(train["Age"], bins=[0, 12, 18, 35, 50, 80],
                                labels=["Child", "Teen", "Adult", "Middle", "Senior"])

    return train

df = load_data()

# Title
st.title("🚢 Titanic Survival Analysis Dashboard")
st.markdown("### Interactive exploration of passenger survival patterns")

# Sidebar filters
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("*Changes here affect all visualizations*")

# Filter options
classes = st.sidebar.multiselect(
    "Passenger Class",
    options=[1, 2, 3],
    default=[1, 2, 3]
)

sexes = st.sidebar.multiselect(
    "Sex",
    options=["male", "female"],
    default=["male", "female"]
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=0,
    max_value=80,
    value=(0, 80)
)

embark_ports = st.sidebar.multiselect(
    "Embarkation Port",
    options=["C", "Q", "S"],
    default=["C", "Q", "S"],
    format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x]
)

# Apply filters
filtered_df = df[
    (df["Pclass"].isin(classes)) &
    (df["Sex"].isin(sexes)) &
    (df["Age"] >= age_range[0]) &
    (df["Age"] <= age_range[1]) &
    (df["Embarked"].isin(embark_ports))
]

# Show filtered count
st.sidebar.markdown(f"**Showing {len(filtered_df)} of {len(df)} passengers**")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    survival_rate = filtered_df["Survived"].mean() * 100
    st.metric("Survival Rate", f"{survival_rate:.1f}%")
with col2:
    avg_age = filtered_df["Age"].mean()
    st.metric("Average Age", f"{avg_age:.1f}")
with col3:
    avg_fare = filtered_df["Fare"].mean()
    st.metric("Average Fare", f"${avg_fare:.2f}")
with col4:
    total_passengers = len(filtered_df)
    st.metric("Total Passengers", total_passengers)

st.markdown("---")

# Main visualizations (2x2 grid)
col_left, col_right = st.columns(2)

# 1. Survival by Sex and Class
with col_left:
    st.subheader("📊 Survival Rate by Sex and Class")

    survival_data = filtered_df.groupby(["Sex", "Pclass"])["Survived"].mean().reset_index()
    survival_data["Survived"] = survival_data["Survived"] * 100

    fig1 = px.bar(
        survival_data,
        x="Sex",
        y="Survived",
        color="Pclass",
        barmode="group",
        title="Women and 1st class passengers had highest survival",
        labels={"Survived": "Survival Rate (%)", "Pclass": "Class"},
        color_discrete_map={1: "#0173B2", 2: "#DE8F05", 3: "#CC78BC"},
        height=350
    )
    fig1.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig1, use_container_width=True)

# 2. Age Distribution by Survival
with col_right:
    st.subheader("📈 Age Distribution by Survival")

    fig2 = go.Figure()

    # Survived
    fig2.add_trace(go.Histogram(
        x=filtered_df[filtered_df["Survived"] == 1]["Age"],
        name="Survived",
        marker_color="#0173B2",
        opacity=0.7,
        nbinsx=30
    ))

    # Did not survive
    fig2.add_trace(go.Histogram(
        x=filtered_df[filtered_df["Survived"] == 0]["Age"],
        name="Did Not Survive",
        marker_color="#DE8F05",
        opacity=0.7,
        nbinsx=30
    ))

    fig2.update_layout(
        barmode="overlay",
        title="Children had better survival chances",
        xaxis_title="Age",
        yaxis_title="Count",
        height=350,
        legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3. Fare vs Age Scatter
with col_left:
    st.subheader("💰 Fare vs Age (Survival)")

    fig3 = px.scatter(
        filtered_df,
        x="Age",
        y="Fare",
        color="Survived",
        size="Fare",
        hover_data=["Name", "Pclass", "Sex"],
        title="Higher fares correlated with survival",
        labels={"Survived": "Survived"},
        color_discrete_map={0: "#DE8F05", 1: "#0173B2"},
        height=350
    )
    fig3.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig3, use_container_width=True)

# 4. Survival by Title and Class
with col_right:
    st.subheader("👤 Survival by Title")

    title_survival = filtered_df.groupby(["Title", "Survived"]).size().reset_index(name="Count")

    fig4 = px.bar(
        title_survival,
        x="Title",
        y="Count",
        color="Survived",
        barmode="stack",
        title="Title (Mrs, Miss) shows strong survival pattern",
        labels={"Survived": "Survived", "Count": "Number of Passengers"},
        color_discrete_map={0: "#DE8F05", 1: "#0173B2"},
        height=350
    )
    fig4.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig4, use_container_width=True)

# Bottom section - Data table with selection
st.markdown("---")
st.subheader("📋 Filtered Passenger Data")

# Display options
show_survivors = st.checkbox("Show only survivors", value=False)
if show_survivors:
    table_df = filtered_df[filtered_df["Survived"] == 1]
else:
    table_df = filtered_df

# Show table
display_cols = ["Name", "Sex", "Age", "Pclass", "Fare", "Embarked", "Survived", "Title"]
st.dataframe(
    table_df[display_cols].head(50),
    use_container_width=True,
    height=300
)

# Footer
st.markdown("---")
st.markdown("**💡 Insights:** The data clearly shows 'Women and Children First' policy was followed. "
            "Female passengers, especially in 1st and 2nd class, had dramatically higher survival rates.")
