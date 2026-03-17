import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder='../../templates')

# ── Load processed data once at startup ──
_df = None

def get_data():
    global _df
    if _df is None:
        from src.data.loader import load_raw_data
        from src.data.preprocessor import preprocess
        from src.features.engineer import engineer_features
        df = load_raw_data('data/raw/Metro_Interstate_Traffic_Volume.csv')
        df = preprocess(df)
        df = engineer_features(df)
        _df = df
    return _df


def compute_risk_score(hour, is_weekend, bad_weather, is_holiday):
    """
    Congestion Risk Score 0–100.
    Higher = more likely to be congested.
    """
    score = 0

    # Time component (max 50)
    if not is_weekend and not is_holiday:
        if hour in [7, 8]:
            score += 45
        elif hour in [9, 16, 17, 18]:
            score += 40
        elif hour in [6, 19]:
            score += 25
        elif hour in [10, 11, 12, 13, 14, 15]:
            score += 20
        else:
            score += 5
    else:
        if 10 <= hour <= 18:
            score += 20
        else:
            score += 5

    # Weather component (max 30)
    if bad_weather:
        score += 30
    else:
        score += 5

    # Holiday component
    if is_holiday:
        score = max(0, score - 20)

    return min(100, score)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/hourly')
def api_hourly():
    df = get_data()
    wday = df[df['is_weekend']==0].groupby('hour')['traffic_volume'].mean().round(0)
    wend = df[df['is_weekend']==1].groupby('hour')['traffic_volume'].mean().round(0)
    return jsonify({
        'hours':   list(range(24)),
        'weekday': wday.reindex(range(24), fill_value=0).tolist(),
        'weekend': wend.reindex(range(24), fill_value=0).tolist(),
    })


@app.route('/api/heatmap')
def api_heatmap():
    df  = get_data()
    pivot = df.groupby(['day_of_week','hour'])['traffic_volume'].mean().unstack(fill_value=0)
    return jsonify({
        'days':  ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
        'hours': list(range(24)),
        'data':  pivot.reindex(range(7)).fillna(0).values.tolist(),
    })


@app.route('/api/weather')
def api_weather():
    df  = get_data()
    avg = df.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=False)
    return jsonify({
        'weather': avg.index.tolist(),
        'volume':  avg.round(0).tolist(),
        'overall': round(df['traffic_volume'].mean(), 0),
    })


@app.route('/api/congestion_dist')
def api_congestion_dist():
    df   = get_data()
    dist = df['congestion_level'].value_counts(normalize=True) * 100
    levels = ['Low','Moderate','High','Critical']
    return jsonify({
        'levels': levels,
        'pct':    [round(dist.get(l, 0), 1) for l in levels],
    })


@app.route('/api/risk_scores')
def api_risk_scores():
    rows = []
    for hour in range(24):
        for is_weekend in [0, 1]:
            score = compute_risk_score(hour, bool(is_weekend), False, False)
            rows.append({
                'hour':       hour,
                'is_weekend': bool(is_weekend),
                'risk_score': score,
                'risk_label': (
                    'Critical' if score >= 70 else
                    'High'     if score >= 45 else
                    'Moderate' if score >= 25 else
                    'Low'
                )
            })
    return jsonify(rows)


@app.route('/api/monthly')
def api_monthly():
    df = get_data()
    monthly = df.groupby(['year','month'])['traffic_volume'].mean().reset_index()
    result  = {}
    for yr in sorted(monthly['year'].unique()):
        subset = monthly[monthly['year']==yr]
        result[int(yr)] = subset.set_index('month')['traffic_volume'].reindex(
            range(1,13), fill_value=0
        ).round(0).tolist()
    return jsonify(result)


@app.route('/api/summary')
def api_summary():
    df = get_data()
    return jsonify({
        'total_records':  len(df),
        'avg_volume':     round(df['traffic_volume'].mean(), 0),
        'peak_hour':      int(df.groupby('hour')['traffic_volume'].mean().idxmax()),
        'peak_day':       ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][
                              int(df.groupby('day_of_week')['traffic_volume'].mean().idxmax())],
        'worst_weather':  df.groupby('weather_main')['traffic_volume'].mean().idxmin(),
        'date_range':     f"{df['date_time'].min().year}–{df['date_time'].max().year}",
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)