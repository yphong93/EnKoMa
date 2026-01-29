# EnKoMa Model Comparison Framework

A comprehensive framework for comparing EnKoMa (Enhanced Koopman via Mamba) model with baseline models on synthetic and real-world time series datasets.

## 📁 Directory Structure

```
model_20260124/
├── models/                    # Model implementations
│   ├── enkoma.py             # EnKoMa model
│   ├── deep_koopman.py       # Deep Koopman baseline
│   ├── pure_mamba.py         # Pure Mamba model
│   ├── lstm.py               # LSTM model
│   ├── transformer.py       # Transformer model
│   ├── gru.py                # GRU model
│   ├── linear.py             # Linear baseline
│   └── components.py         # Shared components
├── system/                    # System implementations
│   ├── synthetic.py          # Synthetic systems (Lorenz, Van der Pol, Duffing, etc.)
│   └── real_data.py          # Real data loaders (ETT, SST, AirQuality, NASA Bearing, EnergyConsumption)
├── configs/                   # Configuration files
│   ├── config_Lorenz.json
│   ├── config_SST_improved.json
│   ├── config_AirQuality.json
│   └── ...                   # Other config files
├── analysis/                  # Analysis tools
│   ├── visualization.py      # Plotting functions
│   ├── eigenvalue.py         # Eigenvalue analysis
│   └── robustness.py         # Robustness testing
├── utils/                     # Utility functions
│   ├── metrics.py            # Evaluation metrics
│   └── lyapunov.py           # Lyapunov time calculation
├── data/                      # Data directory
│   └── real_data/            # Real-world datasets
├── compare_experiment.py      # Main comparison experiment script
├── ablation_test_loss.py      # Ablation test script
├── config.py                  # Configuration class
└── compare_result/            # Experiment results directory
```

## 🚀 Quick Start

### 1. Comparison Experiment

Run comparison experiments with multiple models:

```bash
# Synthetic systems
python compare_experiment.py configs/config_Lorenz.json

# Real-world datasets
python compare_experiment.py configs/config_AirQuality.json
```

### 2. Ablation Test

Test different loss component combinations:

```bash
python ablation_test_loss.py --config configs/config_AirQuality.json
```

## 📊 Supported Datasets

### Synthetic Systems
- **Lorenz System**: Chaotic system
- **Van der Pol Oscillator**: Nonlinear oscillator
- **Duffing Oscillator**: Forced oscillation system
- **Burgers Equation**: Fluid dynamics
- **Kuramoto-Sivashinsky**: Spatiotemporal chaos

### Real-World Datasets
- **ETT**: Electricity Transformer Temperature (ETTh1, ETTh2, ETTm1, ETTm2)
- **SST**: NOAA Sea Surface Temperature
- **AirQuality**: Air quality monitoring data
- **NASA Bearing**: Bearing degradation dataset
- **EnergyConsumption**: Energy consumption dataset

## 🔧 Configuration Files

Configuration files are located in `configs/` directory. Each config file specifies:
- Model hyperparameters (latent_dim, seq_len, pred_len, etc.)
- Training settings (epochs, lr, batch_size, etc.)
- Loss weights (alpha_rec, alpha_pred, alpha_spectral, etc.)
- Data preprocessing options (detrend_method, normalization_method, etc.)

## 📈 Evaluation Metrics

The framework evaluates models using:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R² Score** (Coefficient of Determination)

Metrics are computed for:
- **Short-term prediction**: Configurable steps (e.g., 6, 12, 18)
- **Long-term prediction**: Configurable steps (e.g., 36, 48, 60)

## 📁 Results Structure

Results are saved in `compare_result/{system_name}/{system_name}_{timestamp}/`:

```
compare_result/SST_SST/SST_SST_20260127_160817/
├── config.json                    # Used configuration
├── experiment.log                  # Full experiment log
├── results_summary.json           # Results summary (JSON)
├── metrics_summary.csv           # Overall metrics (CSV)
├── stepwise_metrics.csv          # Step-wise metrics (CSV)
├── metrics_comparison.png        # Metrics comparison chart
├── stepwise_*.png                # Step-wise comparison charts
└── {model_name}/                 # Per-model results
    ├── pred_vs_gt_sample_0.png
    ├── phase_space_*.png
    ├── {model_name}_predictions_sample_0.csv
    └── eigenvalues/              # Eigenvalue analysis (EnKoMa only)
        ├── eigenvalue_analysis.png
        └── global_jacobian_stitching.png
```

## 🔬 Analysis Features

### Phase Space Visualization
- 2D and 3D phase space reconstruction
- Trajectory comparison between models

### Eigenvalue Analysis (EnKoMa only)
- Global Jacobian stitching
- Complex plane distribution
- Mode frequency analysis
- Spectral radius tracking

### Robustness Testing
- Noise injection at various levels
- Performance degradation analysis

## 📝 Data Preparation

### Real-World Data

Place your data files in `data/real_data/`:

- **ETT**: `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- **SST**: `sst.csv`
- **AirQuality**: `air_quality.csv`
- **NASA Bearing**: `bearing_1.csv`, `bearing_2.csv`, etc.
- **EnergyConsumption**: `Energy_consumption_dataset.csv`

### Data Preprocessing

The framework automatically applies:
- **Detrending**: Removes trends (linear, polynomial, or seasonal)
- **Normalization**: StandardScaler, RobustScaler, or MinMaxScaler
- **Outlier handling**: IQR method for outlier capping
- **Smoothing**: Optional Gaussian filtering for noisy data

## 🛠️ Dependencies

- PyTorch
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- pandas

## 📚 Additional Documentation

- `ABLATION_TEST_README.md`: Ablation test usage guide
- `SST_PREPROCESSING_VERIFICATION.md`: SST preprocessing details
- `EIGENVALUE_ANALYSIS_GUIDE.md`: Eigenvalue analysis guide
- Various analysis markdown files for specific experiments

## 💡 Tips

1. **GPU Selection**: Use `CUDA_VISIBLE_DEVICES` to specify GPU
2. **Config Customization**: Modify config files to adjust hyperparameters
3. **Result Analysis**: Check `experiment.log` for detailed training logs
4. **Model Comparison**: Review `metrics_summary.csv` for quick comparison

## 📧 Support

For issues or questions, refer to the analysis markdown files in the root directory or check the experiment logs.
