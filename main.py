"""
Main orchestration script for the Global Derivatives Pricing Simulator.

This script demonstrates the complete functionality of the pricing simulator
by running all models, stress tests, and generating comprehensive results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.options import OptionPricer, validate_put_call_parity
from models.futures import FuturesPricer, validate_futures_pricing
from models.swaps import SwapPricer, validate_swap_pricing
from utils.stress_tests import (
    volatility_stress_test, 
    spot_price_stress_test, 
    time_decay_stress_test,
    run_comprehensive_stress_tests,
    calculate_stress_metrics
)
from utils.visualization import (
    plot_payoff_diagram, 
    plot_stress_test, 
    plot_model_comparison,
    plot_greeks_comparison
)
from utils.helpers import (
    print_pricing_summary, 
    save_results_to_csv, 
    load_market_data,
    create_sample_market_data,
    format_results_table,
    validate_inputs
)


def run_all_models(S: float = 100.0, K: float = 105.0, T: float = 1.0, 
                  r: float = 0.05, sigma: float = 0.2, option_type: str = "call") -> dict:
    """
    Run all pricing models and return comprehensive results.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free interest rate
        sigma: Volatility
        option_type: Option type ("call" or "put")
        
    Returns:
        Dictionary containing all results
    """
    print("="*60)
    print("GLOBAL DERIVATIVES PRICING SIMULATOR")
    print("="*60)
    print(f"Parameters: S=${S:.2f}, K=${K:.2f}, T={T:.2f} years, r={r:.1%}, σ={sigma:.1%}")
    print(f"Option Type: {option_type.upper()}")
    print("="*60)
    
    # Validate inputs
    validation = validate_inputs(S, K, T, r, sigma, option_type)
    if not validation['valid']:
        print("ERROR: Invalid inputs detected:")
        for error in validation['errors']:
            print(f"  - {error}")
        return {}
    
    if validation['warnings']:
        print("WARNINGS:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
        print()
    
    results = {}
    
    # 1. OPTIONS PRICING
    print("1. OPTIONS PRICING MODELS")
    print("-" * 30)
    
    option_pricer = OptionPricer()
    option_results = option_pricer.run_all_models(S, K, T, r, sigma, option_type)
    results['model_comparison'] = option_results
    
    # Print individual model results
    for model_name, model_result in option_results.items():
        print(f"{model_name.replace('_', ' ').title()}:")
        print(f"  Price: ${model_result['price']:.4f}")
        print(f"  Delta: {model_result['delta']:.4f}")
        print(f"  Gamma: {model_result['gamma']:.4f}")
        print(f"  Theta: {model_result['theta']:.4f}")
        print(f"  Vega:  {model_result['vega']:.4f}")
        print(f"  Rho:   {model_result['rho']:.4f}")
        print()
    
    # 2. PUT-CALL PARITY VALIDATION
    print("2. PUT-CALL PARITY VALIDATION")
    print("-" * 30)
    
    parity_result = validate_put_call_parity(S, K, T, r, sigma)
    results['put_call_parity'] = parity_result
    
    print(f"Call Price: ${parity_result['call_price']:.4f}")
    print(f"Put Price:  ${parity_result['put_price']:.4f}")
    print(f"LHS (C-P):  ${parity_result['lhs']:.4f}")
    print(f"RHS (S-Ke^(-rT)): ${parity_result['rhs']:.4f}")
    print(f"Difference: ${parity_result['difference']:.6f}")
    print(f"Parity Holds: {parity_result['parity_holds']}")
    print()
    
    # 3. FUTURES PRICING
    print("3. FUTURES PRICING")
    print("-" * 30)
    
    futures_pricer = FuturesPricer()
    futures_result = futures_pricer.price_cost_of_carry(S, T, r, q=0.02)
    results['futures_pricing'] = futures_result
    
    print(f"Spot Price: ${S:.2f}")
    print(f"Futures Price: ${futures_result['futures_price']:.4f}")
    print(f"Fair Value: ${futures_result['fair_value']:.4f}")
    print(f"Implied Yield: {futures_result['implied_yield']:.4f}")
    print(f"Cost of Carry: {futures_result['cost_of_carry']:.4f}")
    print(f"Basis: ${futures_result['basis']:.4f}")
    print()
    
    # 4. SWAP PRICING
    print("4. INTEREST RATE SWAP PRICING")
    print("-" * 30)
    
    swap_pricer = SwapPricer()
    notional = 1000000  # $1M notional
    fixed_rate = 0.05
    floating_rates = [0.045, 0.047, 0.049, 0.051]  # Quarterly rates
    swap_maturity = 1.0
    
    swap_result = swap_pricer.price_interest_rate_swap(
        notional, fixed_rate, floating_rates, swap_maturity
    )
    results['swap_pricing'] = swap_result
    
    print(f"Notional: ${notional:,.0f}")
    print(f"Fixed Rate: {fixed_rate:.1%}")
    print(f"Swap Value: ${swap_result['swap_value']:,.2f}")
    print(f"Fixed Leg PV: ${swap_result['fixed_leg_pv']:,.2f}")
    print(f"Floating Leg PV: ${swap_result['floating_leg_pv']:,.2f}")
    print(f"Duration: {swap_result['duration']:.4f}")
    print(f"Convexity: {swap_result['convexity']:.4f}")
    print()
    
    # 5. STRESS TESTING
    print("5. STRESS TESTING")
    print("-" * 30)
    
    # Volatility stress test
    print("Running volatility stress test...")
    vol_stress_df = volatility_stress_test(S, K, T, r, sigma, option_type)
    results['volatility_stress'] = vol_stress_df
    
    # Spot price stress test
    print("Running spot price stress test...")
    spot_stress_df = spot_price_stress_test(K, T, r, sigma, S, option_type)
    results['spot_price_stress'] = spot_stress_df
    
    # Time decay stress test
    print("Running time decay stress test...")
    time_stress_df = time_decay_stress_test(S, K, r, sigma, option_type)
    results['time_decay_stress'] = time_stress_df
    
    # Calculate stress test metrics
    stress_metrics = calculate_stress_metrics(vol_stress_df, 'black_scholes_price')
    results['stress_test_metrics'] = stress_metrics
    
    print("Stress test metrics:")
    for metric, value in stress_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    # 6. VISUALIZATION
    print("6. GENERATING VISUALIZATIONS")
    print("-" * 30)
    
    # Create output directory for plots
    os.makedirs('screenshots', exist_ok=True)
    
    # Payoff diagram
    print("Creating payoff diagram...")
    payoff_fig = plot_payoff_diagram(S, K, option_type, T, r, sigma)
    payoff_fig.savefig('screenshots/payoff_diagram.png', dpi=300, bbox_inches='tight')
    plt.close(payoff_fig)
    
    # Model comparison chart
    print("Creating model comparison chart...")
    comparison_fig = plot_model_comparison(option_results)
    comparison_fig.savefig('screenshots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(comparison_fig)
    
    # Greeks comparison
    print("Creating Greeks comparison chart...")
    greeks_fig = plot_greeks_comparison(option_results, 'delta')
    greeks_fig.savefig('screenshots/greeks_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(greeks_fig)
    
    # Volatility stress test plot
    print("Creating volatility stress test plot...")
    vol_stress_fig = plot_stress_test(
        vol_stress_df, 
        'volatility', 
        ['black_scholes_price', 'binomial_price', 'monte_carlo_price'],
        'Volatility Stress Test'
    )
    vol_stress_fig.savefig('screenshots/volatility_stress_test.png', dpi=300, bbox_inches='tight')
    plt.close(vol_stress_fig)
    
    # Spot price stress test plot
    print("Creating spot price stress test plot...")
    spot_stress_fig = plot_stress_test(
        spot_stress_df,
        'spot_price',
        ['black_scholes_price', 'binomial_price', 'monte_carlo_price'],
        'Spot Price Stress Test'
    )
    spot_stress_fig.savefig('screenshots/spot_price_stress_test.png', dpi=300, bbox_inches='tight')
    plt.close(spot_stress_fig)
    
    print("All visualizations saved to screenshots/ directory")
    print()
    
    # 7. SAVE RESULTS
    print("7. SAVING RESULTS")
    print("-" * 30)
    
    # Save comprehensive results to CSV
    results_df = pd.DataFrame(option_results).T
    results_df.to_csv('data/results.csv')
    print("Results saved to data/results.csv")
    
    # Save stress test results
    vol_stress_df.to_csv('data/volatility_stress_test.csv', index=False)
    spot_stress_df.to_csv('data/spot_price_stress_test.csv', index=False)
    time_stress_df.to_csv('data/time_decay_stress_test.csv', index=False)
    print("Stress test results saved to data/ directory")
    
    # 8. SUMMARY
    print("8. EXECUTION SUMMARY")
    print("-" * 30)
    print(f"Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models tested: {len(option_results)}")
    print(f"Stress test scenarios: {len(vol_stress_df)}")
    print(f"Visualizations created: 5")
    print(f"Output files: 4 CSV files + 5 PNG files")
    print()
    
    return results


def run_multi_asset_analysis():
    """
    Run analysis on multiple assets from sample market data.
    """
    print("="*60)
    print("MULTI-ASSET ANALYSIS")
    print("="*60)
    
    # Load sample market data
    market_data = load_market_data('data/sample_market_data.csv')
    
    if market_data.empty:
        print("Creating sample market data...")
        market_data = create_sample_market_data('data/sample_market_data.csv')
    
    print(f"Loaded {len(market_data)} assets for analysis")
    print()
    
    # Initialize pricer
    option_pricer = OptionPricer()
    
    # Results storage
    all_results = []
    
    # Process each asset
    for idx, row in market_data.iterrows():
        print(f"Processing {row['symbol']}...")
        
        try:
            # Run Black-Scholes pricing
            result = option_pricer.price_black_scholes(
                row['spot_price'], 
                row['strike_price'], 
                row['time_to_maturity'], 
                row['risk_free_rate'], 
                row['volatility'], 
                row['option_type']
            )
            
            # Add asset information
            result['symbol'] = row['symbol']
            result['spot_price'] = row['spot_price']
            result['strike_price'] = row['strike_price']
            result['volatility'] = row['volatility']
            result['time_to_maturity'] = row['time_to_maturity']
            result['option_type'] = row['option_type']
            
            all_results.append(result)
            
        except Exception as e:
            print(f"  Error processing {row['symbol']}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save multi-asset results
    results_df.to_csv('data/multi_asset_results.csv', index=False)
    print(f"\nMulti-asset analysis completed. Results saved to data/multi_asset_results.csv")
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"Average option price: ${results_df['price'].mean():.4f}")
    print(f"Price range: ${results_df['price'].min():.4f} - ${results_df['price'].max():.4f}")
    print(f"Average delta: {results_df['delta'].mean():.4f}")
    print(f"Average gamma: {results_df['gamma'].mean():.4f}")
    print(f"Average theta: {results_df['theta'].mean():.4f}")
    print(f"Average vega: {results_df['vega'].mean():.4f}")
    
    return results_df


def main():
    """
    Main execution function.
    """
    try:
        # Run single asset analysis
        results = run_all_models()
        
        # Run multi-asset analysis
        multi_asset_results = run_multi_asset_analysis()
        
        # Print final summary
        print_pricing_summary(results)
        
        print("="*60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Check the following directories for outputs:")
        print("  - data/: CSV files with results")
        print("  - screenshots/: PNG files with visualizations")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


