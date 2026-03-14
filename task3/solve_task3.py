"""
Task 3: Heat Pump Load Forecasting
- Predict average x2 per device per month (May-Oct 2025)
- Metric: MAE
- Strategy: Streaming aggregation of training data, seasonal extrapolation
"""
import os, csv, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


def main():
    t0 = time.time()
    print("Task 3: Heat Pump Load Forecasting")

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
    devices_path = os.path.join(os.path.dirname(__file__), 'data', 'devices.csv')

    devices = pd.read_csv(devices_path)
    device_ids = devices['deviceId'].unique()
    print(f"Devices: {len(device_ids)}")

    # Streaming aggregation: sum and count per (device, month)
    dev_month_sum = {}
    dev_month_cnt = {}

    print("Reading training data (streaming)...")
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=2_000_000,
                                           usecols=['deviceId', 'timedate', 'period', 'x2'])):
        train = chunk[chunk['period'] == 'train'].dropna(subset=['x2'])
        if len(train) == 0:
            continue
        train = train.copy()
        train['month'] = pd.to_datetime(train['timedate']).dt.month

        agg = train.groupby(['deviceId', 'month'])['x2'].agg(['sum', 'count'])
        for (dev, month), row in agg.iterrows():
            key = (dev, month)
            dev_month_sum[key] = dev_month_sum.get(key, 0.0) + row['sum']
            dev_month_cnt[key] = dev_month_cnt.get(key, 0) + row['count']

        if (i + 1) % 5 == 0:
            print(f"  {(i+1)*2}M rows processed ({time.time()-t0:.0f}s)", flush=True)

    print(f"Reading done in {time.time()-t0:.0f}s")

    # Compute averages
    dev_month_avg = {k: dev_month_sum[k] / dev_month_cnt[k] for k in dev_month_sum}

    # Per-device overall average
    dev_sum = {}
    dev_cnt = {}
    for (dev, month), s in dev_month_sum.items():
        dev_sum[dev] = dev_sum.get(dev, 0.0) + s
        dev_cnt[dev] = dev_cnt.get(dev, 0) + dev_month_cnt[(dev, month)]
    dev_overall = {d: dev_sum[d] / dev_cnt[d] for d in dev_sum}

    # Global monthly averages
    month_sum = {}
    month_cnt = {}
    for (dev, month), s in dev_month_sum.items():
        month_sum[month] = month_sum.get(month, 0.0) + s
        month_cnt[month] = month_cnt.get(month, 0) + dev_month_cnt[(dev, month)]
    global_month_avg = {m: month_sum[m] / month_cnt[m] for m in month_sum}
    global_overall = sum(month_sum.values()) / sum(month_cnt.values())

    print(f"\nGlobal avg x2: {global_overall:.6f}")
    print("Monthly averages:", {m: round(v, 5) for m, v in sorted(global_month_avg.items())})

    # Training has months 10,11,12,1,2,3,4
    # Need to predict 5,6,7,8,9,10
    # Month 10 exists in training! Can use it directly for Oct prediction

    # For months 5-9, extrapolate from the seasonal trend
    # Heat pumps: winter high, summer low (for heating)
    # But x2 could also reflect cooling load in summer

    # Build prediction
    predictions = []
    for dev_id in device_ids:
        for month in range(5, 11):
            if (dev_id, month) in dev_month_avg:
                # Direct data (e.g. Oct from training)
                pred = dev_month_avg[(dev_id, month)]
            elif dev_id in dev_overall:
                dev_avg = dev_overall[dev_id]
                if month in global_month_avg and global_overall > 0:
                    factor = global_month_avg[month] / global_overall
                    pred = dev_avg * factor
                else:
                    # For unseen months, use April as proxy for warm season
                    apr_global = global_month_avg.get(4, global_overall)
                    if global_overall > 0:
                        pred = dev_avg * (apr_global / global_overall) * 0.6
                    else:
                        pred = dev_avg * 0.3
            else:
                pred = global_month_avg.get(month, global_overall * 0.3)

            predictions.append((dev_id, 2025, month, max(0.0, pred)))

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'out', 'load_submission.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['deviceId', 'year', 'month', 'prediction'])
        writer.writerows(predictions)

    print(f"\nSaved: {output_path}")
    print(f"Predictions: {len(predictions)} rows")
    preds_arr = np.array([p[3] for p in predictions])
    print(f"Stats: mean={preds_arr.mean():.5f}, std={preds_arr.std():.5f}, "
          f"min={preds_arr.min():.5f}, max={preds_arr.max():.5f}")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
