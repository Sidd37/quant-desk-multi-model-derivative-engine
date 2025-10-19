"""
Streamlit Dashboard for Global Derivatives Pricing Simulator.

A professional web interface for interactive derivatives pricing and analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.options import OptionPricer
from models.futures import FuturesPricer
from models.swaps import SwapPricer
from utils.stress_tests import volatility_stress_test, spot_price_stress_test
from utils.visualization import plot_payoff_diagram, plot_stress_test
from utils.helpers import load_market_data, create_sample_market_data

# Page configuration
st.set_page_config(
    page_title="Global Derivatives Pricing Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Jane Street style
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f1f5f9;
    }
    
    .stSlider > div > div > div > div {
        background-color: #3b82f6;
    }
    
    .sidebar .sidebar-content {
        background-color: #1e293b;
    }
    
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'market_data' not in st.session_state:
    st.session_state.market_data = None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Global Derivatives Pricing Simulator</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Professional quantitative pricing models for institutional trading**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["Options", "Futures", "Swaps", "Multi-Asset Analysis"]
        )
        
        if model_type == "Options":
            option_type = st.selectbox("Option Type", ["call", "put"])
            model_method = st.selectbox(
                "Pricing Method",
                ["Black-Scholes", "Binomial Tree", "Monte Carlo", "All Models"]
            )
        elif model_type == "Futures":
            futures_type = st.selectbox(
                "Futures Type",
                ["Equity Index", "Commodity", "Currency", "Interest Rate"]
            )
        elif model_type == "Swaps":
            swap_type = st.selectbox(
                "Swap Type",
                ["Interest Rate Swap", "Currency Swap", "Credit Default Swap"]
            )
    
    # Main content area
    if model_type == "Options":
        options_dashboard(option_type, model_method)
    elif model_type == "Futures":
        futures_dashboard(futures_type)
    elif model_type == "Swaps":
        swaps_dashboard(swap_type)
    elif model_type == "Multi-Asset Analysis":
        multi_asset_dashboard()


def options_dashboard(option_type, model_method):
    """Options pricing dashboard."""
    
    st.markdown("## 📈 Options Pricing Analysis")
    
    # Create two columns for inputs and results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Parameters")
        
        # Market parameters
        S = st.slider("Spot Price ($)", 50.0, 200.0, 100.0, 1.0)
        K = st.slider("Strike Price ($)", 50.0, 200.0, 105.0, 1.0)
        T = st.slider("Time to Maturity (years)", 0.01, 2.0, 1.0, 0.01)
        r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100
        sigma = st.slider("Volatility (%)", 5.0, 100.0, 20.0, 1.0) / 100
        
        # Run simulation button
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Calculating option prices..."):
                run_options_simulation(S, K, T, r, sigma, option_type, model_method)
    
    with col2:
        if st.session_state.results:
            display_options_results()
        else:
            st.info("👈 Adjust parameters and click 'Run Simulation' to see results")


def run_options_simulation(S, K, T, r, sigma, option_type, model_method):
    """Run options pricing simulation."""
    
    pricer = OptionPricer()
    results = {}
    
    if model_method == "All Models":
        results = pricer.run_all_models(S, K, T, r, sigma, option_type)
    elif model_method == "Black-Scholes":
        results["black_scholes"] = pricer.price_black_scholes(S, K, T, r, sigma, option_type)
    elif model_method == "Binomial Tree":
        results["binomial"] = pricer.price_binomial(S, K, T, r, sigma, option_type)
    elif model_method == "Monte Carlo":
        results["monte_carlo"] = pricer.price_monte_carlo(S, K, T, r, sigma, option_type)
    
    # Store results in session state
    st.session_state.results = {
        'type': 'options',
        'parameters': {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'option_type': option_type},
        'pricing_results': results,
        'model_method': model_method
    }


def display_options_results():
    """Display options pricing results."""
    
    results = st.session_state.results
    params = results['parameters']
    pricing_results = results['pricing_results']
    
    # Key metrics
    st.markdown("### 📊 Pricing Results")
    
    # Create metrics columns
    cols = st.columns(len(pricing_results))
    
    for i, (model_name, model_result) in enumerate(pricing_results.items()):
        with cols[i]:
            st.metric(
                label=f"{model_name.replace('_', ' ').title()} Price",
                value=f"${model_result['price']:.4f}",
                delta=f"Δ: {model_result['delta']:.3f}"
            )
    
    # Detailed results table
    st.markdown("### 📋 Detailed Analysis")
    
    # Create DataFrame for display
    df_results = pd.DataFrame(pricing_results).T
    df_display = df_results[['price', 'delta', 'gamma', 'theta', 'vega', 'rho']].round(4)
    df_display.columns = ['Price ($)', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    
    st.dataframe(df_display, use_container_width=True)
    
    # Charts section
    st.markdown("### 📈 Visualizations")
    
    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["Payoff Diagram", "Greeks Comparison", "Stress Tests"])
    
    with tab1:
        # Payoff diagram
        fig = plot_payoff_diagram(
            params['S'], params['K'], params['option_type'], 
            params['T'], params['r'], params['sigma']
        )
        st.pyplot(fig)
        plt.close(fig)
    
    with tab2:
        # Greeks comparison
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        greek_values = [pricing_results[list(pricing_results.keys())[0]][greek] for greek in greeks]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(greeks, greek_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_title('Greeks Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Greek Value')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, greek_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        st.pyplot(fig)
        plt.close(fig)
    
    with tab3:
        # Stress tests
        st.markdown("#### Volatility Stress Test")
        
        # Run volatility stress test
        vol_stress_df = volatility_stress_test(
            params['S'], params['K'], params['T'], params['r'], 
            params['sigma'], params['option_type']
        )
        
        # Create interactive plot
        fig = go.Figure()
        
        for col in ['black_scholes_price', 'binomial_price', 'monte_carlo_price']:
            if col in vol_stress_df.columns:
                fig.add_trace(go.Scatter(
                    x=vol_stress_df['volatility'],
                    y=vol_stress_df[col],
                    mode='lines+markers',
                    name=col.replace('_', ' ').title(),
                    line=dict(width=3)
                ))
        
        fig.update_layout(
            title="Volatility Stress Test",
            xaxis_title="Volatility",
            yaxis_title="Option Price ($)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def futures_dashboard(futures_type):
    """Futures pricing dashboard."""
    
    st.markdown("## 🌾 Futures Pricing Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Parameters")
        
        S = st.slider("Spot Price", 50.0, 200.0, 100.0, 1.0)
        T = st.slider("Time to Maturity (years)", 0.01, 2.0, 0.25, 0.01)
        r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100
        
        if futures_type == "Commodity":
            q = st.slider("Convenience Yield (%)", 0.0, 5.0, 2.0, 0.1) / 100
            storage_cost = st.slider("Storage Cost (%)", 0.0, 3.0, 1.0, 0.1) / 100
        elif futures_type == "Currency":
            q = st.slider("Foreign Interest Rate (%)", 0.0, 10.0, 3.0, 0.1) / 100
            storage_cost = 0.0
        else:
            q = st.slider("Dividend Yield (%)", 0.0, 5.0, 2.0, 0.1) / 100
            storage_cost = 0.0
        
        if st.button("🚀 Calculate Futures Price", type="primary"):
            with st.spinner("Calculating futures price..."):
                run_futures_simulation(S, T, r, q, storage_cost, futures_type)
    
    with col2:
        if st.session_state.results and st.session_state.results.get('type') == 'futures':
            display_futures_results()
        else:
            st.info("👈 Adjust parameters and click 'Calculate Futures Price' to see results")


def run_futures_simulation(S, T, r, q, storage_cost, futures_type):
    """Run futures pricing simulation."""
    
    pricer = FuturesPricer()
    
    if futures_type == "Commodity":
        result = pricer.price_commodity_futures(S, T, r, q, storage_cost)
    elif futures_type == "Currency":
        result = pricer.price_currency_futures(S, T, r, q)
    else:
        result = pricer.price_cost_of_carry(S, T, r, q, storage_cost)
    
    st.session_state.results = {
        'type': 'futures',
        'parameters': {'S': S, 'T': T, 'r': r, 'q': q, 'storage_cost': storage_cost},
        'pricing_results': result,
        'futures_type': futures_type
    }


def display_futures_results():
    """Display futures pricing results."""
    
    results = st.session_state.results
    pricing_results = results['pricing_results']
    
    st.markdown("### 📊 Futures Pricing Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Futures Price", f"${pricing_results['futures_price']:.4f}")
    
    with col2:
        st.metric("Fair Value", f"${pricing_results['fair_value']:.4f}")
    
    with col3:
        st.metric("Implied Yield", f"{pricing_results['implied_yield']:.4f}")
    
    with col4:
        st.metric("Basis", f"${pricing_results['basis']:.4f}")
    
    # Cost of carry analysis
    st.markdown("### 📈 Cost of Carry Analysis")
    
    params = results['parameters']
    
    # Create cost of carry breakdown
    carry_components = {
        'Interest Cost': params['r'] * params['T'],
        'Storage Cost': params['storage_cost'] * params['T'],
        'Convenience Yield': -params['q'] * params['T'],
        'Net Cost of Carry': pricing_results['cost_of_carry'] * params['T']
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    components = list(carry_components.keys())
    values = list(carry_components.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(components, values, color=colors, alpha=0.8)
    ax.set_title('Cost of Carry Components', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rate')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
               f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    st.pyplot(fig)
    plt.close(fig)


def swaps_dashboard(swap_type):
    """Interest rate swap dashboard."""
    
    st.markdown("## 🔄 Interest Rate Swap Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Parameters")
        
        notional = st.slider("Notional Amount ($)", 100000, 10000000, 1000000, 100000)
        fixed_rate = st.slider("Fixed Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100
        maturity = st.slider("Maturity (years)", 0.5, 10.0, 5.0, 0.5)
        
        st.markdown("#### Floating Rate Schedule")
        n_periods = int(maturity * 4)  # Quarterly payments
        floating_rates = []
        
        for i in range(n_periods):
            rate = st.slider(f"Period {i+1} Rate (%)", 0.0, 10.0, 4.5 + i*0.1, 0.1) / 100
            floating_rates.append(rate)
        
        if st.button("🚀 Calculate Swap Value", type="primary"):
            with st.spinner("Calculating swap value..."):
                run_swaps_simulation(notional, fixed_rate, floating_rates, maturity)
    
    with col2:
        if st.session_state.results and st.session_state.results.get('type') == 'swaps':
            display_swaps_results()
        else:
            st.info("👈 Adjust parameters and click 'Calculate Swap Value' to see results")


def run_swaps_simulation(notional, fixed_rate, floating_rates, maturity):
    """Run interest rate swap simulation."""
    
    pricer = SwapPricer()
    result = pricer.price_interest_rate_swap(notional, fixed_rate, floating_rates, maturity)
    
    st.session_state.results = {
        'type': 'swaps',
        'parameters': {'notional': notional, 'fixed_rate': fixed_rate, 'floating_rates': floating_rates, 'maturity': maturity},
        'pricing_results': result
    }


def display_swaps_results():
    """Display interest rate swap results."""
    
    results = st.session_state.results
    pricing_results = results['pricing_results']
    
    st.markdown("### 📊 Swap Valuation Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Swap Value", f"${pricing_results['swap_value']:,.2f}")
    
    with col2:
        st.metric("Fixed Leg PV", f"${pricing_results['fixed_leg_pv']:,.2f}")
    
    with col3:
        st.metric("Floating Leg PV", f"${pricing_results['floating_leg_pv']:,.2f}")
    
    with col4:
        st.metric("Duration", f"{pricing_results['duration']:.4f}")
    
    # Cash flow analysis
    st.markdown("### 📈 Cash Flow Analysis")
    
    params = results['parameters']
    n_periods = len(params['floating_rates'])
    periods = list(range(1, n_periods + 1))
    
    # Calculate cash flows
    fixed_coupon = params['notional'] * params['fixed_rate'] / 4  # Quarterly
    floating_coupons = [params['notional'] * rate / 4 for rate in params['floating_rates']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fixed leg cash flows
    ax1.bar(periods, [fixed_coupon] * n_periods, alpha=0.7, color='blue', label='Fixed Leg')
    ax1.set_title('Fixed Leg Cash Flows', fontweight='bold')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Cash Flow ($)')
    ax1.grid(True, alpha=0.3)
    
    # Floating leg cash flows
    ax2.bar(periods, floating_coupons, alpha=0.7, color='red', label='Floating Leg')
    ax2.set_title('Floating Leg Cash Flows', fontweight='bold')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Cash Flow ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def multi_asset_dashboard():
    """Multi-asset analysis dashboard."""
    
    st.markdown("## 🌍 Multi-Asset Analysis")
    
    # Load or create market data
    if st.session_state.market_data is None:
        with st.spinner("Loading market data..."):
            market_data = load_market_data('data/sample_market_data.csv')
            if market_data.empty:
                market_data = create_sample_market_data('data/sample_market_data.csv')
            st.session_state.market_data = market_data
    
    market_data = st.session_state.market_data
    
    st.markdown(f"### 📊 Market Data Overview ({len(market_data)} assets)")
    
    # Display market data
    st.dataframe(market_data, use_container_width=True)
    
    if st.button("🚀 Run Multi-Asset Pricing", type="primary"):
        with st.spinner("Running multi-asset pricing analysis..."):
            run_multi_asset_simulation(market_data)


def run_multi_asset_simulation(market_data):
    """Run multi-asset pricing simulation."""
    
    pricer = OptionPricer()
    results = []
    
    progress_bar = st.progress(0)
    
    for idx, row in market_data.iterrows():
        try:
            result = pricer.price_black_scholes(
                row['spot_price'], 
                row['strike_price'], 
                row['time_to_maturity'], 
                row['risk_free_rate'], 
                row['volatility'], 
                row['option_type']
            )
            
            result['symbol'] = row['symbol']
            result['spot_price'] = row['spot_price']
            result['strike_price'] = row['strike_price']
            result['volatility'] = row['volatility']
            result['time_to_maturity'] = row['time_to_maturity']
            result['option_type'] = row['option_type']
            
            results.append(result)
            
        except Exception as e:
            st.error(f"Error processing {row['symbol']}: {e}")
        
        progress_bar.progress((idx + 1) / len(market_data))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results
    st.markdown("### 📈 Multi-Asset Pricing Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Price", f"${results_df['price'].mean():.4f}")
    
    with col2:
        st.metric("Price Range", f"${results_df['price'].min():.2f} - ${results_df['price'].max():.2f}")
    
    with col3:
        st.metric("Average Delta", f"{results_df['delta'].mean():.3f}")
    
    with col4:
        st.metric("Average Vega", f"{results_df['vega'].mean():.3f}")
    
    # Results table
    st.dataframe(results_df, use_container_width=True)
    
    # Volatility surface
    st.markdown("### 📊 Volatility Surface")
    
    # Create volatility surface plot
    fig = go.Figure(data=go.Scatter3d(
        x=results_df['strike_price'],
        y=results_df['time_to_maturity'],
        z=results_df['volatility'],
        mode='markers',
        marker=dict(
            size=8,
            color=results_df['price'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Option Price")
        ),
        text=results_df['symbol'],
        hovertemplate='<b>%{text}</b><br>' +
                     'Strike: $%{x:.2f}<br>' +
                     'Maturity: %{y:.2f} years<br>' +
                     'Volatility: %{z:.1%}<br>' +
                     'Price: $%{marker.color:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="3D Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price ($)",
            yaxis_title="Time to Maturity (years)",
            zaxis_title="Volatility"
        ),
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()


