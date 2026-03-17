import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

OUTPUT_DIR = 'outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    'bg':     '#0D1117',
    'panel':  '#161B22',
    'border': '#30363D',
    'text':   '#E6EDF3',
    'sub':    '#8B949E',
    'green':  '#3FB950',
    'blue':   '#58A6FF',
    'orange': '#F78166',
    'yellow': '#E3B341',
    'purple': '#BC8CFF',
    'red':    '#FF6E6E',
    'teal':   '#39D353',
}

CONG_COLORS = ['#3FB950', '#E3B341', '#F78166', '#FF6E6E']


def style_ax(ax, title):
    """Apply consistent dark theme styling to an axis."""
    ax.set_facecolor(PALETTE['panel'])
    ax.set_title(title, color=PALETTE['text'], fontsize=11,
                 fontweight='bold', pad=8)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE['border'])
    ax.tick_params(colors=PALETTE['sub'])
    return ax


def plot_traffic_overview(df, save=True):
    """Full overview dashboard — hourly, daily, monthly patterns."""
    fig = plt.figure(figsize=(22, 14), facecolor=PALETTE['bg'])
    fig.suptitle('Traffic Volume — Overview Dashboard',
                 fontsize=20, fontweight='bold', color=PALETTE['text'], y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                           top=0.93, bottom=0.07, left=0.06, right=0.97)

    # ── 1. Hourly pattern weekday vs weekend ──
    ax1 = style_ax(fig.add_subplot(gs[0, :2]),
                   '⏰  Avg Traffic Volume by Hour')
    wday = df[df['is_weekend']==0].groupby('hour')['traffic_volume'].mean()
    wend = df[df['is_weekend']==1].groupby('hour')['traffic_volume'].mean()
    ax1.plot(range(24), wday, color=PALETTE['blue'], lw=2.5,
             marker='o', ms=5, label='Weekday')
    ax1.plot(range(24), wend, color=PALETTE['purple'], lw=2.5,
             marker='s', ms=5, ls='--', label='Weekend')
    ax1.fill_between(range(24), wday, alpha=0.15, color=PALETTE['blue'])
    ax1.fill_between(range(24), wend, alpha=0.10, color=PALETTE['purple'])
    for ph in [7, 8, 17, 18]:
        ax1.axvline(ph, color=PALETTE['orange'], alpha=0.4, lw=1.2, ls=':')
    ax1.set_xlabel('Hour of Day', color=PALETTE['sub'])
    ax1.set_ylabel('Avg Volume (vehicles/hr)', color=PALETTE['sub'])
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)],
                        rotation=35, fontsize=8)
    ax1.legend(fontsize=9)

    # ── 2. Congestion donut ──
    ax2 = style_ax(fig.add_subplot(gs[0, 2]), '🚦  Congestion Distribution')
    ax2.set_aspect('equal')
    dist   = df['congestion_level'].value_counts(normalize=True) * 100
    levels = ['Low', 'Moderate', 'High', 'Critical']
    sizes  = [dist.get(l, 0) for l in levels]
    wedges, _ = ax2.pie(sizes, colors=CONG_COLORS, startangle=90,
                        wedgeprops=dict(width=0.55, edgecolor=PALETTE['bg'], lw=2))
    labels = [f'{l}\n{s:.1f}%' for l, s in zip(levels, sizes)]
    ax2.legend(wedges, labels, loc='center', fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), ncol=2, framealpha=0)

    # ── 3. Day × Hour heatmap ──
    ax3 = style_ax(fig.add_subplot(gs[1, :2]),
                   '📅  Traffic Heatmap — Hour × Day of Week')
    pivot = df.groupby(['day_of_week', 'hour'])['traffic_volume'].mean().unstack()
    cmap  = LinearSegmentedColormap.from_list('cong', CONG_COLORS, N=256)
    im    = ax3.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=0, vmax=7000)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax3.set_yticks(range(7))
    ax3.set_yticklabels(day_names, fontsize=9)
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xticklabels([f'{h:02d}h' for h in range(0, 24, 2)], fontsize=8)
    plt.colorbar(im, ax=ax3, label='Avg Volume', shrink=0.85, pad=0.02)

    # ── 4. Monthly trend ──
    ax4 = style_ax(fig.add_subplot(gs[1, 2]), '📆  Monthly Traffic Trend')
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    monthly = df.groupby('month')['traffic_volume'].mean()
    vals    = [monthly.get(m, 0) for m in range(1, 13)]
    bar_c   = [CONG_COLORS[min(3, int(v/2000))] for v in vals]
    bars    = ax4.bar(range(1, 13), vals, color=bar_c,
                      edgecolor=PALETTE['border'], lw=0.8, width=0.7)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(month_names, rotation=45, fontsize=8)
    ax4.set_ylabel('Avg Volume', color=PALETTE['sub'])
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                 f'{val:.0f}', ha='center', fontsize=7, color=PALETTE['sub'])

    plt.savefig(f'{OUTPUT_DIR}/traffic_overview.png',
                dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    print("✅ Saved: traffic_overview.png")
    plt.close()


def plot_weather_impact(df, save=True):
    """Weather conditions vs traffic volume."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Weather Impact on Traffic Volume', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    # ── Bar chart — avg volume by weather ──
    ax = style_ax(axes[0], '🌦️  Avg Volume by Weather Type')
    weather_avg = df.groupby('weather_main')['traffic_volume'].mean().sort_values()
    overall_avg = df['traffic_volume'].mean()
    bar_c = [PALETTE['red'] if v < overall_avg*0.85 else
             PALETTE['yellow'] if v < overall_avg else
             PALETTE['green'] for v in weather_avg.values]
    ax.barh(weather_avg.index, weather_avg.values,
            color=bar_c, edgecolor=PALETTE['border'])
    ax.axvline(overall_avg, color=PALETTE['blue'], ls='--',
               lw=1.5, label=f'Overall avg: {overall_avg:.0f}')
    ax.set_xlabel('Avg Traffic Volume', color=PALETTE['sub'])
    ax.legend(fontsize=9)

    # ── Box plot — volume distribution by weather ──
    ax2 = style_ax(axes[1], '📊  Volume Distribution by Weather')
    weather_order = df.groupby('weather_main')['traffic_volume'].median().sort_values().index
    data_by_weather = [df[df['weather_main']==w]['traffic_volume'].values
                       for w in weather_order]
    bp = ax2.boxplot(data_by_weather, patch_artist=True, vert=True)
    for patch, color in zip(bp['boxes'], [PALETTE['blue']]*len(weather_order)):
        patch.set_facecolor(PALETTE['blue'])
        patch.set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians', 'fliers']:
        plt.setp(bp[element], color=PALETTE['orange'])
    ax2.set_xticklabels(weather_order, rotation=40, ha='right', fontsize=8)
    ax2.set_ylabel('Traffic Volume', color=PALETTE['sub'])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/weather_impact.png',
                dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    print("✅ Saved: weather_impact.png")
    plt.close()


def plot_congestion_risk(df, save=True):
    """High-risk congestion windows and bottleneck analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Congestion Risk & Bottleneck Analysis', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    # ── Stacked bar — congestion level % by hour ──
    ax = style_ax(axes[0], '⚠️  Congestion Risk by Hour (Weekday)')
    wd = df[df['is_weekend']==0]
    cong_hour = wd.groupby(['hour', 'congestion_level']).size().unstack(fill_value=0)
    cong_pct  = cong_hour.div(cong_hour.sum(axis=1), axis=0) * 100
    bottom    = np.zeros(24)
    for lvl, col in zip(['Low','Moderate','High','Critical'], CONG_COLORS):
        if lvl in cong_pct.columns:
            vals = cong_pct[lvl].reindex(range(24), fill_value=0).values
            ax.bar(range(24), vals, bottom=bottom, color=col, label=lvl, width=0.85)
            bottom += vals
    ax.set_xlabel('Hour of Day', color=PALETTE['sub'])
    ax.set_ylabel('% of Records', color=PALETTE['sub'])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h:02d}h' for h in range(0, 24, 2)], fontsize=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, 100)

    # ── Risk heatmap by hour+day ──
    ax2 = style_ax(axes[1], '🔴  High/Critical Risk — Hour × Day')
    risk_pivot = df.groupby(['day_of_week', 'hour']).apply(
        lambda g: (g['congestion_level'].isin(['High','Critical'])).mean() * 100
    ).unstack()
    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    cmap2 = LinearSegmentedColormap.from_list('risk', ['#0D1117','#E3B341','#FF6E6E'], N=256)
    im2   = ax2.imshow(risk_pivot.values, aspect='auto', cmap=cmap2, vmin=0, vmax=100)
    ax2.set_yticks(range(7))
    ax2.set_yticklabels(day_names, fontsize=9)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f'{h:02d}h' for h in range(0, 24, 2)], fontsize=8)
    plt.colorbar(im2, ax=ax2, label='% High/Critical', shrink=0.85)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/congestion_risk.png',
                dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    print("✅ Saved: congestion_risk.png")
    plt.close()


def plot_temperature_analysis(df, save=True):
    """Temperature vs traffic volume analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE['bg'])
    fig.suptitle('Temperature & Seasonal Analysis', fontsize=16,
                 fontweight='bold', color=PALETTE['text'])

    # ── Temp vs volume scatter ──
    ax = style_ax(axes[0], '🌡️  Temperature vs Traffic Volume')
    sample = df.sample(min(5000, len(df)), random_state=42)
    sc = ax.scatter(sample['temp_c'], sample['traffic_volume'],
                    alpha=0.2, s=5, c=sample['hour'],
                    cmap='plasma', rasterized=True)
    plt.colorbar(sc, ax=ax, label='Hour of Day')
    ax.set_xlabel('Temperature (°C)', color=PALETTE['sub'])
    ax.set_ylabel('Traffic Volume', color=PALETTE['sub'])

    # ── Year over year ──
    ax2 = style_ax(axes[1], '📉  Year-over-Year Monthly Trend')
    year_colors = [PALETTE['blue'], PALETTE['purple'], PALETTE['teal'],
                   PALETTE['orange'], PALETTE['yellow'], PALETTE['green']]
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    for i, yr in enumerate(sorted(df['year'].unique())):
        subset = df[df['year']==yr].groupby('month')['traffic_volume'].mean()
        ax2.plot(subset.index, subset.values, marker='o', ms=4, lw=2,
                 color=year_colors[i % len(year_colors)], label=str(yr))
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names, rotation=40, fontsize=8)
    ax2.set_ylabel('Avg Traffic Volume', color=PALETTE['sub'])
    ax2.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/temperature_analysis.png',
                dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    print("✅ Saved: temperature_analysis.png")
    plt.close()


def run_all_plots(df):
    """Generate all EDA visualizations."""
    print("\n── Generating All Visualizations ─────")
    plot_traffic_overview(df)
    plot_weather_impact(df)
    plot_congestion_risk(df)
    plot_temperature_analysis(df)
    print("\n✅ All visualizations saved to outputs/figures/")