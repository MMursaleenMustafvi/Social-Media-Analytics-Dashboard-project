import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
from streamlit_option_menu import option_menu

# ===== SETUP =====
st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("social_media_data.csv")
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Error: 'social_media_data.csv' not found!")
        return None

df = load_data()

if df is not None:
    # Define columns to analyze
    columns = ['Likes', 'Shares', 'Comments']
    colors = ['#3498db', '#2ecc71', '#9b59b6']  # Blue, Green, Purple
    
    # Style
    sns.set_style("whitegrid")
    plt.style.use('ggplot')
    sns.set_palette(colors)
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("📊 Dashboard Controls")
        
        analysis_type = option_menu(
            menu_title="Analysis Type",
            options=["Overview", "Distributions", "Statistics", "Regression"],
            icons=["speedometer", "bar-chart", "calculator", "graph-up"],
            default_index=0,
            styles={
                "container": {"padding": "5px"},
                "nav-link": {"font-size": "14px"},
            }
        )
        
        st.divider()
        st.subheader("Filter Options")
        
        if 'Date' in df.columns:
            date_range = st.date_input(
                "Select Date Range",
                value=[df['Date'].min(), df['Date'].max()],
                min_value=df['Date'].min(),
                max_value=df['Date'].max()
            )
        
        selected_metrics = st.multiselect(
            "Select Metrics to Analyze",
            options=columns,
            default=columns
        )
        
        st.divider()
        st.caption("This dashboard analyzes social media engagement metrics")

    # ===== MAIN CONTENT =====
    st.title("📈 Social Media Analytics Dashboard")
    
    # Overview Section
    if analysis_type == "Overview":
        st.header("📋 Overview")
        
        # Metrics Cards
        col1, col2, col3 = st.columns(3)
        metrics = [
            ("Total Posts", len(df), "#3498db"),
            ("Avg Likes", df['Likes'].mean(), "#2ecc71"),
            ("Engagement", (df['Likes'].sum() + df['Comments'].sum()) / len(df), "#9b59b6")
        ]
        
        for i, (title, value, color) in enumerate(metrics):
            with [col1, col2, col3][i]:
                st.metric(label=title, value=f"{value:,.0f}" if isinstance(value, (int, float)) else value)
        
        # Time Series
        if 'Date' in df.columns:
            st.header("🕒 Engagement Over Time")
            
            selected_col = st.selectbox("Select Metric", selected_metrics, key='ts_select')
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(
                x='Date', 
                y=selected_col, 
                data=df, 
                marker='o', 
                ax=ax, 
                color=colors[columns.index(selected_col)],
                linewidth=2
            )
            ax.set_title(f"{selected_col} Over Time", fontsize=12)
            plt.xticks(rotation=45)
            st.pyplot(fig, use_container_width=True)

    # Distributions Section
    elif analysis_type == "Distributions":
        st.header("📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Histogram")
            selected_col = st.selectbox("Select Metric", selected_metrics, key='hist_select')
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(
                df[selected_col], 
                bins=20, 
                kde=True, 
                ax=ax, 
                color=colors[columns.index(selected_col)]
            )
            ax.set_title(f"Distribution of {selected_col}", fontsize=12)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.subheader("Box Plot")
            selected_col = st.selectbox("Select Metric", selected_metrics, key='box_select')
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(
                x=df[selected_col], 
                ax=ax, 
                color=colors[columns.index(selected_col)]
            )
            ax.set_title(f"Boxplot of {selected_col}", fontsize=12)
            st.pyplot(fig, use_container_width=True)

    # Statistics Section
    elif analysis_type == "Statistics":
        st.header("🧮 Statistical Analysis")
        
        # Descriptive Statistics
        st.subheader("Descriptive Statistics")
        summary = df[selected_metrics].describe().T
        summary["Variance"] = df[selected_metrics].var()
        summary["Skewness"] = df[selected_metrics].skew()
        summary["Kurtosis"] = df[selected_metrics].kurtosis()
        st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)
        
        # Confidence Intervals
        st.subheader("95% Confidence Intervals")
        conf_intervals = {}
        for col in selected_metrics:
            data = df[col].dropna()
            mean = np.mean(data)
            sem = stats.sem(data)
            ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
            conf_intervals[col] = ci
        
        ci_df = pd.DataFrame(conf_intervals, index=['Lower', 'Upper']).T
        st.dataframe(ci_df.style.format("{:.2f}"), use_container_width=True)
        
        # Distribution Fitting
        st.subheader("Distribution Fitting")
        selected_col = st.selectbox("Select Metric", selected_metrics, key='fit_select')
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            df[selected_col], 
            bins=30, 
            kde=True, 
            stat="density", 
            color=colors[columns.index(selected_col)], 
            ax=ax
        )
        
        # Fit normal distribution
        mu, std = stats.norm.fit(df[selected_col].dropna())
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'r--', linewidth=2, label=f'N({mu:.2f}, {std:.2f}²)')
        ax.set_title(f"{selected_col} with Fitted Normal Distribution", fontsize=12)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # Regression Section
    elif analysis_type == "Regression":
        st.header("📈 Regression Analysis")
        
        # Predict Likes based on Comments and Shares
        st.subheader("Linear Regression Model")
        st.markdown("Predicting **Likes** based on **Comments** and **Shares**")
        
        X = df[['Comments', 'Shares']]
        X = sm.add_constant(X)
        y = df['Likes']
        model = sm.OLS(y, X).fit()
        
        # Display model summary
        st.subheader("Model Summary")
        st.text(model.summary())
        
        # Regression Plots
        st.subheader("Regression Plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.regplot(
                x='Comments', 
                y='Likes', 
                data=df, 
                ax=ax,
                scatter_kws={'alpha':0.6, 'color': colors[0]},
                line_kws={'color': '#e74c3c', 'linewidth': 2}
            )
            ax.set_title("Likes vs Comments", fontsize=12)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.regplot(
                x='Shares', 
                y='Likes', 
                data=df, 
                ax=ax,
                scatter_kws={'alpha':0.6, 'color': colors[1]},
                line_kws={'color': '#e74c3c', 'linewidth': 2}
            )
            ax.set_title("Likes vs Shares", fontsize=12)
            st.pyplot(fig, use_container_width=True)
        
        # Predictions
        df['Predicted_Likes'] = model.predict(X)
        
        st.subheader("Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            x='Likes', 
            y='Predicted_Likes', 
            data=df, 
            ax=ax, 
            color=colors[2],
            alpha=0.7
        )
        ax.plot(
            [df['Likes'].min(), df['Likes'].max()],
            [df['Likes'].min(), df['Likes'].max()], 
            'r--', 
            linewidth=2
        )
        ax.set_title("Actual vs Predicted Likes", fontsize=12)
        ax.set_xlabel("Actual Likes")
        ax.set_ylabel("Predicted Likes")
        st.pyplot(fig, use_container_width=True)

    # Footer
    st.divider()
    st.caption("Social Media Analytics Dashboard .")