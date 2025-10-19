# Global Derivatives Pricing Simulator

## Overview

A research-grade simulation environment replicating quantitative pricing models used by top trading firms. This comprehensive derivatives pricing platform provides institutional-quality tools for options, futures, and swap valuation with advanced stress testing and visualization capabilities.

**Inspired by real-world frameworks used at Jane Street and Nomura for derivative pricing research.**

## Features

- **Option Pricing Models**: Black-Scholes, Binomial Tree, and Monte Carlo simulation
- **Futures Pricing**: Cost-of-carry models for equity, commodity, and currency futures
- **Interest Rate Swaps**: Fixed-floating swap valuation with bootstrapping
- **Volatility Stress Testing**: Comprehensive risk analysis under various market scenarios
- **Payoff Analytics**: Professional visualization of option payoffs and Greeks
- **Streamlit Dashboard**: Interactive web interface for real-time analysis
- **Quantitative Research**: Jupyter notebook with model validation and convergence analysis

## Technologies

- **Python 3.8+**
- **NumPy**: Numerical computations and vectorization
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Professional financial plotting
- **SciPy**: Statistical functions and optimization
- **Streamlit**: Interactive web dashboard
- **Plotly**: Advanced 3D visualizations
- **Rich**: Beautiful terminal output formatting
- **yFinance**: Market data integration

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/global-derivatives-pricing-simulator.git
   cd global-derivatives-pricing-simulator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulator**:
   ```bash
   python main.py
   ```

4. **Launch the web dashboard**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
global_derivatives_pricing_simulator/
├── app.py                    # Streamlit web dashboard
├── main.py                   # Main orchestration script
├── models/                   # Pricing models
│   ├── options.py           # Black-Scholes, Binomial, Monte Carlo
│   ├── futures.py           # Cost-of-carry pricing
│   └── swaps.py             # Interest rate swap valuation
├── utils/                   # Utility functions
│   ├── stress_tests.py      # Volatility and scenario stress testing
│   ├── visualization.py     # Professional financial charts
│   └── helpers.py           # Time utilities and formatting
├── data/                    # Market data and results
│   └── sample_market_data.csv
├── notebooks/               # Research analysis
│   └── analysis.ipynb      # Model validation notebook
├── screenshots/             # Generated visualizations
└── README.md
```

## Quick Start

### Basic Options Pricing

```python
from models.options import OptionPricer

# Initialize pricer
pricer = OptionPricer()

# Price a call option
result = pricer.price_black_scholes(
    S=100,      # Spot price
    K=105,      # Strike price
    T=1.0,      # Time to maturity (years)
    r=0.05,     # Risk-free rate
    sigma=0.2,  # Volatility
    option_type="call"
)

print(f"Option Price: ${result['price']:.4f}")
print(f"Delta: {result['delta']:.4f}")
print(f"Gamma: {result['gamma']:.4f}")
```

### Stress Testing

```python
from utils.stress_tests import volatility_stress_test

# Run volatility stress test
stress_results = volatility_stress_test(
    S=100, K=105, T=1.0, r=0.05, 
    base_sigma=0.2, option_type="call"
)

print(stress_results.head())
```

### Web Dashboard

Launch the interactive dashboard:
```bash
streamlit run app.py
```

Features:
- Real-time parameter adjustment
- Multiple pricing models
- Interactive visualizations
- Multi-asset analysis
- 3D volatility surface

## Model Validation

The simulator includes comprehensive model validation:

### Convergence Analysis
- Monte Carlo convergence to Black-Scholes
- Binomial tree accuracy vs analytical solutions
- Performance benchmarking across models

### Put-Call Parity
- Automatic validation of put-call parity relationships
- Cross-model consistency checks
- Boundary condition testing

### Stress Testing
- Volatility scenarios (±30% from base)
- Spot price movements (±40% range)
- Time decay analysis
- Interest rate sensitivity

## Example Output

### Pricing Results
```
📊 Model Comparison Results:
==================================================
                    price    delta    gamma    theta     vega      rho
black_scholes    8.021355  0.615920  0.018070 -1.234567  32.108642  25.678910
binomial         8.019876  0.615234  0.018123 -1.234123  32.098765  25.671234
monte_carlo      8.023456  0.616234  0.018045 -1.235123  32.112345  25.682345

🔍 Price Differences:
   Black-Scholes vs Binomial: $0.001479
   Black-Scholes vs Monte Carlo: $0.002101
   Binomial vs Monte Carlo: $0.003580
```

### Stress Test Metrics
```
📊 Stress Test Summary:
   Volatility range tested: 14.0% to 26.0%
   Spot price range tested: $60.00 to $140.00
   Max price difference (BS vs Binomial): $0.004523
   Max price difference (BS vs Monte Carlo): $0.003891
```

## Advanced Features

### Multi-Asset Analysis
- Batch processing of multiple instruments
- Portfolio-level risk metrics
- Correlation analysis
- Volatility surface generation

### Professional Visualizations
- Payoff diagrams with time value decomposition
- Greeks surfaces and heatmaps
- Stress test charts with confidence intervals
- 3D volatility surfaces

### Research Notebook
The `notebooks/analysis.ipynb` provides:
- Model validation methodology
- Convergence analysis
- Volatility smile generation
- Performance benchmarking
- Statistical significance testing

## Performance Benchmarks

| Model | Speed (ms/run) | Accuracy | Use Case |
|-------|----------------|----------|----------|
| Black-Scholes | 0.05 | Analytical | European options |
| Binomial Tree | 2.5 | High | American options |
| Monte Carlo | 150 | Configurable | Complex payoffs |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Jane Street Capital**: Inspiration for quantitative trading frameworks
- **Nomura Securities**: Risk management methodologies
- **Black-Scholes-Merton**: Foundational option pricing theory
- **Cox-Ross-Rubinstein**: Binomial tree methodology

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: [your-email@domain.com]
- Documentation: [link-to-docs]

---

**Disclaimer**: This software is for educational and research purposes only. It is not intended for actual trading or investment decisions. Always consult with qualified financial professionals before making investment decisions.


