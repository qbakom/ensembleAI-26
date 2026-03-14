"""
Task 3 v2: Improved Heat Pump Load Forecasting
- Use per-device seasonal patterns from training data
- Use external temperature correlation for better extrapolation
- Compute device-specific monthly trends
"""

import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Target: predict avg x2 per device per month (May-Oct 2025)
# Training: Oct 2024 - Apr 2025
# Validation: May-Jun 2025 (x2 withheld)
# Test: Jul-Oct 2025 (x2 withheld)

DATA_PATH = 'data/data.csv'
DEVICES_PATH = 'data/devices.csv'
OUTPUT_PATH = 'data/out/load_submission_v2.csv'


def main():
    print("Task 3 v2: Heat Pump Load Forecasting")

    devices = pd.read_csv(DEVICES_PATH)
    device_ids = devices['deviceId'].unique()
    print(f"Devices: {len(device_ids)}")

    # Read data in chunks - collect per-device per-month stats
    print("Reading data (chunked)...")

    # Collect: {(deviceId, year, month): {x2_values, t1_values}}
    train_data = {}  # (dev, month) -> list of x2
    train_t1 = {}    # (dev, month) -> list of t1
    val_test_t1 = {} # (dev, year, month) -> list of t1

    chunk_size = 1000000
    for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=chunk_size,
                                           usecols=['deviceId', 'timedate', 'period', 't1', 'x2'])):
        if (i+1) % 10 == 0:
            print(f"  Chunk {i+1}...")

        chunk['dt'] = pd.to_datetime(chunk['timedate'])
        chunk['month'] = chunk['dt'].dt.month
        chunk['year'] = chunk['dt'].dt.year

        # Training data
        train_mask = chunk['period'] == 'train'
        train_chunk = chunk[train_mask].dropna(subset=['x2'])
        for (dev, month), grp in train_chunk.groupby(['deviceId', 'month']):
            key = (dev, month)
            if key not in train_data:
                train_data[key] = []
                train_t1[key] = []
            train_data[key].extend(grp['x2'].tolist())
            train_t1[key].extend(grp['t1'].dropna().tolist())

        # Validation/test - collect temperature data
        vt_mask = chunk['period'].isin(['valid', 'test'])
        vt_chunk = chunk[vt_mask]
        for (dev, year, month), grp in vt_chunk.groupby(['deviceId', 'year', 'month']):
            key = (dev, year, month)
            if key not in val_test_t1:
                val_test_t1[key] = []
            vals = grp['t1'].dropna().tolist()
            if vals:
                val_test_t1[key].extend(vals)

    print(f"  Training entries: {len(train_data)}")

    # Compute per-device per-month averages
    dev_month_x2 = {k: np.mean(v) for k, v in train_data.items()}
    dev_month_t1 = {k: np.mean(v) for k, v in train_t1.items() if v}

    # Compute per-device overall
    dev_x2_all = {}
    for (dev, month), vals in train_data.items():
        if dev not in dev_x2_all:
            dev_x2_all[dev] = []
        dev_x2_all[dev].extend(vals)
    dev_overall_x2 = {dev: np.mean(v) for dev, v in dev_x2_all.items()}

    # Global monthly pattern
    month_x2_all = {}
    month_t1_all = {}
    for (dev, month), vals in train_data.items():
        if month not in month_x2_all:
            month_x2_all[month] = []
        month_x2_all[month].extend(vals)
    for (dev, month), vals in train_t1.items():
        if month not in month_t1_all:
            month_t1_all[month] = []
        month_t1_all[month].extend(vals)

    global_month_x2 = {m: np.mean(v) for m, v in month_x2_all.items()}
    global_month_t1 = {m: np.mean(v) for m, v in month_t1_all.items()}
    global_x2 = np.mean([v for vals in month_x2_all.values() for v in vals])

    print(f"\nGlobal monthly x2: {dict(sorted(global_month_x2.items()))}")
    print(f"Global monthly t1: {dict(sorted(global_month_t1.items()))}")

    # Build simple temperature -> x2 model per device
    # x2 tends to be higher when t1 is lower (winter heating)
    # For prediction months, use the validation/test temperature to estimate x2

    # Simple approach: per-device linear relationship between t1 and x2
    from sklearn.linear_model import LinearRegression

    dev_models = {}
    for dev in device_ids:
        months_data = [(m, dev_month_x2.get((dev, m)), dev_month_t1.get((dev, m)))
                       for m in range(1, 13)
                       if (dev, m) in dev_month_x2 and (dev, m) in dev_month_t1]
        if len(months_data) >= 3:
            t1_vals = np.array([d[2] for d in months_data]).reshape(-1, 1)
            x2_vals = np.array([d[1] for d in months_data])
            model = LinearRegression()
            model.fit(t1_vals, x2_vals)
            dev_models[dev] = model

    print(f"Devices with temperature models: {len(dev_models)}")

    # Compute val/test temperatures per device per month
    vt_month_t1 = {}
    for (dev, year, month), vals in val_test_t1.items():
        vt_month_t1[(dev, month)] = np.mean(vals)

    # Make predictions
    predictions = []
    target_months = [5, 6, 7, 8, 9, 10]

    for dev_id in device_ids:
        for month in target_months:
            pred = None

            # Method 1: Use temperature model if available
            if dev_id in dev_models and (dev_id, month) in vt_month_t1:
                t1_val = vt_month_t1[(dev_id, month)]
                pred = dev_models[dev_id].predict([[t1_val]])[0]
                # Clip to reasonable range
                pred = max(0.0, pred)

            # Method 2: Use device's monthly pattern with seasonal scaling
            elif (dev_id, month) in dev_month_x2:
                pred = dev_month_x2[(dev_id, month)]

            elif dev_id in dev_overall_x2:
                dev_avg = dev_overall_x2[dev_id]
                # Scale by global seasonal pattern
                if global_x2 > 0 and month in global_month_x2:
                    pred = dev_avg * (global_month_x2[month] / global_x2)
                else:
                    # Extrapolate: summer months typically have lower load
                    # Training shows Oct~0.08, Nov~0.17, Dec~0.20, Jan~0.21, Feb~0.23, Mar~0.13, Apr~0.08
                    # Summer (May-Oct) should be similar to Apr/Oct level or lower
                    summer_factor = {5: 0.4, 6: 0.3, 7: 0.25, 8: 0.25, 9: 0.3, 10: 0.5}
                    pred = dev_avg * summer_factor.get(month, 0.3)
            else:
                # Use global pattern with summer extrapolation
                summer_avg = {5: 0.06, 6: 0.04, 7: 0.03, 8: 0.03, 9: 0.04, 10: 0.06}
                pred = summer_avg.get(month, 0.04)

            predictions.append((dev_id, 2025, month, max(0.0, pred)))

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['deviceId', 'year', 'month', 'prediction'])
        writer.writerows(predictions)

    # Stats
    df = pd.DataFrame(predictions, columns=['deviceId', 'year', 'month', 'prediction'])
    print(f"\nPrediction stats per month:")
    print(df.groupby('month')['prediction'].describe())
    print(f"\nSubmission saved: {OUTPUT_PATH} ({len(predictions)} rows)")


if __name__ == "__main__":
    main()
