#!/usr/bin/env python3
"""
Compute per-class statistics from a Gwyddion (.gwy) topography and a
corresponding mask saved as a NumPy array (.npy).

Usage examples:
  python stats_from_mask.py data/file.gwy masks/file.npy --out results.csv

Batch mode (process all .gwy files in a folder and match masks by base name):
  python stats_from_mask.py --batch data_dir --mask-dir masks_dir --out all_stats.csv

Outputs:
 - Per-file per-label table saved as CSV (and optional pickle)
 - Aggregated table by (temperature, percentage, label) saved as CSV
 - Interactive Plotly plots by temperature/percentage saved as HTML files (one per label)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

LABEL_NAMES = {
    1: 'Grain interior',
    2: 'Grain boundary',
}
# Monkeypatch numpy.fromstring BEFORE importing gwyfile
# (gwyfile uses deprecated np.fromstring which was removed in newer NumPy versions)
if not hasattr(np, 'fromstring'):
    # Store the old numpy module reference
    import importlib
    
    # Create a wrapper that mimics the old fromstring behavior for binary mode
    def _fromstring_compat(data, dtype=None, sep='', **kwargs):
        """Compatibility wrapper for removed np.fromstring"""
        if isinstance(data, str) and sep:
            # Text mode with separator - convert string to bytes then parse
            return np.fromstring(data.encode() if isinstance(data, str) else data, 
                                dtype=dtype, sep=sep, **kwargs)
        else:
            # Binary mode - use frombuffer instead
            return np.frombuffer(data, dtype=dtype, **kwargs)
    
    np.fromstring = _fromstring_compat

try:
    import gwyfile
except Exception:  # pragma: no cover - informative error when not installed
    gwyfile = None


def load_gwy_topography_like_notebook(path: str, channel_name: Optional[str] = None, scale: float = 1e4) -> np.ndarray:
    """Load a .gwy file and return a 2D numpy array (topography)."""
    if gwyfile is None:
        raise RuntimeError("gwyfile is required to load .gwy files (install gwyfile).")

    obj = gwyfile.load(path)
    channels = gwyfile.util.get_datafields(obj)
    if not channels:
        raise ValueError(f"No datafields found in GWY file: {path}")

    if channel_name is None:
        channel_name = list(channels.keys())[0]

    channel = channels[channel_name]
    img = channel.data * scale
    # reshape if flattened data object
    arr = np.asarray(img)
    if arr.ndim == 1 and hasattr(channel, 'yres') and hasattr(channel, 'xres'):
        arr = arr.reshape(channel.yres, channel.xres)

    return arr.astype('float32')


def compute_stats_for_mask(image: np.ndarray, mask: np.ndarray) -> Sequence[tuple]:
    """Return list of tuples with statistics for each label > 0 found in mask.

    Each tuple: (label, count, mean, std, median, p25, p75, min, max)
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Shape mismatch: image {image.shape} vs mask {mask.shape}")

    labels = np.unique(mask)
    out = []
    for lab in labels:
        lab = int(lab)
        if lab == 0:
            continue
        m = mask == lab
        vals = image[m]
        if vals.size == 0:
            cnt = 0
            mean = std = median = p25 = p75 = minv = maxv = float('nan')
        else:
            cnt = int(vals.size)
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            median = float(np.median(vals))
            p25 = float(np.percentile(vals, 25))
            p75 = float(np.percentile(vals, 75))
            minv = float(np.min(vals))
            maxv = float(np.max(vals))
        out.append((lab, cnt, mean, std, median, p25, p75, minv, maxv))

    out.sort(key=lambda x: x[0])
    return out
def plot_histogram_for_mask(image: np.ndarray,
                            mask: np.ndarray,
                            out_path: str,
                            bins: int = 50,
                            title: Optional[str] = None):
    """
    Create an interactive Plotly histogram where each label (>0) in the mask
    is shown as an overlaid histogram (one trace per label), with a Gaussian fit
    overlaid for each label. The legend shows μ and σ for each fit.
    """
    labels = np.unique(mask)
    labels = [int(lab) for lab in labels if lab != 0]

    if not labels:
        print(f'No labels > 0 found for histogram {out_path}, skipping')
        return

    # Collect all values for global x-range
    all_vals = []
    per_label_vals = {}
    for lab in labels:
        m = mask == lab
        vals = image[m].ravel()
        if vals.size == 0:
            continue
        per_label_vals[lab] = vals
        all_vals.append(vals)

    if not all_vals:
        print(f'All labels are empty for histogram {out_path}, skipping')
        return

    all_vals_concat = np.concatenate(all_vals)
    x_min = float(all_vals_concat.min())
    x_max = float(all_vals_concat.max())
    if x_min == x_max:
        # Avoid degenerate range
        x_min -= 0.5
        x_max += 0.5

    fig = go.Figure()

    for lab, vals in per_label_vals.items():
        label_name = LABEL_NAMES.get(lab, f'Label {lab}')

        # --- Histogram trace ---
        hist = go.Histogram(
            x=vals,
            name=f'{label_name} (data)',
            opacity=0.5,  
            nbinsx=bins
        )
        fig.add_trace(hist)

        # Extract the auto-assigned color for this histogram
        # (Plotly assigns colors after the Figure is displayed, so we must
        #  manually specify colors using a colorscale to ensure consistency)
        # Here we just set our own matching colors manually.
        # Color sequence similar to Plotly defaults:
        color_seq = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
            "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
            "#FF97FF", "#FECB52"
        ]
        color = color_seq[(lab - 1) % len(color_seq)]

        # --- Gaussian fit ---
        if vals.size > 1:
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
        else:
            mu = float(vals[0])
            sigma = 0.0

        if sigma > 0:
            xs = np.linspace(x_min, x_max, 300)
            pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

            # Scale Gaussian height to histogram counts
            counts, edges = np.histogram(vals, bins=bins, range=(x_min, x_max))
            max_count = counts.max() if counts.size > 0 else 1
            scale_factor = max_count / pdf.max()

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=pdf * scale_factor,
                    mode='lines',
                    name=f'{label_name} fit (μ={mu:.3g}, σ={sigma:.3g})',
                    line=dict(color=color, width=3)   # SAME COLOR AS HISTOGRAM
                )
            )


    fig.update_layout(
        barmode='overlay',  # overlay histograms
        title=title or 'Per-label histogram',
        xaxis_title='CPD (Volts)',
        yaxis_title='Count',
        width=900,
        height=600,
        template='plotly_white'
    )

    fig.write_html(out_path)
    print(f'Saved histogram to {out_path}')


def find_matching_mask(gwy_path: str, mask_dir: Optional[str] = None) -> Optional[str]:
    """Given a .gwy path, look for a matching .npy mask with same base name."""
    base = os.path.splitext(os.path.basename(gwy_path))[0]
    search_dir = mask_dir if mask_dir is not None else os.path.dirname(gwy_path)
    cand = os.path.join(search_dir, base + '.npy')
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(search_dir, base + '_mask.npy')
    if os.path.exists(cand2):
        return cand2
    return None


def _parse_temp_and_percentage(filename: str) -> tuple:
    """Extract temperature (digits before 'C') and percentage (digits before 'per') from filename."""
    base = os.path.basename(filename)
    t_match = re.search(r"(\d+)(?=\s*[Cc])", base)
    temp = int(t_match.group(1)) if t_match else None
    p_match = re.search(r"(\d+)(?=\s*per)", base, flags=re.IGNORECASE)
    pct = int(p_match.group(1)) if p_match else None
    return temp, pct


def load_metadata_log(log_csv: str) -> dict:
    """Load metadata log CSV and return a dict mapping file basename to metadata (including light column).
    
    Expected columns in log: afm_image_id, Light (or similar).
    Returns dict: {filename_stem -> {'Light': value, ...}}
    """
    if not os.path.exists(log_csv):
        print(f"Warning: metadata log {log_csv} not found, light column will be empty")
        return {}
    
    try:
        log_df = pd.read_csv(log_csv)
    except Exception as e:
        print(f"Warning: could not load metadata log {log_csv}: {e}")
        return {}
    
    # Try to find the image ID column (common names: afm_image_id, image_id, File)
    id_col = None
    for col in ['afm_image_id', 'image_id', 'File', 'file']:
        if col in log_df.columns:
            id_col = col
            break
    
    if id_col is None:
        print(f"Warning: could not find image ID column in {log_csv}")
        return {}
    
    # Try to find the light column
    light_col = None
    for col in ['Light', 'light', 'Light_condition']:
        if col in log_df.columns:
            light_col = col
            break
    
    if light_col is None:
        print(f"Warning: could not find Light column in {log_csv}")
        return {}
    
    # Build dict: {file_stem -> {metadata columns}}
    metadata_dict = {}
    for _, row in log_df.iterrows():
        file_id = str(row[id_col]).strip()
        # Store the light value and any other relevant metadata
        metadata_dict[file_id] = {
            'light_condition': str(row[light_col]).strip()
        }
    
    return metadata_dict


def process_pair(
    gwy_path: str,
    mask_path: str,
    channel: Optional[str] = None,
    scale: float = 1e6,
    metadata_dict: Optional[dict] = None,
    hist_dir: Optional[str] = None,
    hist_bins: int = 50,
):
    """Load the image and mask, compute stats and return list of row dicts.

    Each row dict keys:
      file,label,temperature,percentage,count,mean,std,median,p25,p75,min,max,light_condition
    """
    if gwyfile is None:
        raise RuntimeError("gwyfile is required to load .gwy files (install gwyfile).")

    img = load_gwy_topography_like_notebook(gwy_path, channel_name=channel, scale=scale)
    mask = np.load(mask_path)
    base = os.path.basename(gwy_path)
    base_stem = os.path.splitext(base)[0]  # Remove extension for lookup
    # --- NEW: per-pair histogram plotting ---
    if hist_dir is not None:
        os.makedirs(hist_dir, exist_ok=True)

        hist_path = os.path.join(hist_dir, f'hist_{base_stem}.html')
        plot_histogram_for_mask(
            img,
            mask,
            out_path=hist_path,
            bins=hist_bins,
            title=f'Histogram per label – {base_stem}'
        )
    stats = compute_stats_for_mask(img, mask)

    temp, pct = _parse_temp_and_percentage(base)
    
    # Look up metadata (light condition) from log
    light = None
    if metadata_dict and base_stem in metadata_dict:
        light = metadata_dict[base_stem].get('light_condition')

    rows = []
    for (lab, cnt, mean, std, median, p25, p75, minv, maxv) in stats:
        row = {
            'file': base,
            'label': int(lab),
            'temperature': temp,
            'percentage': pct,
            'count': int(cnt),
            'mean': mean,
            'std': std,
            'median': median,
            'p25': p25,
            'p75': p75,
            'min': minv,
            'max': maxv,
            'light_condition': light
        }
        rows.append(row)
    return rows


def aggregate_by_temp_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-file per-label DataFrame by (temperature, percentage, label)."""
    # drop rows missing temperature or percentage for aggregation
    agg_df = (
        df.dropna(subset=['temperature', 'percentage'])
        .groupby(['temperature', 'percentage', 'label'])
        .agg(
            n_files=('file', 'nunique'),
            n_observations=('count', 'sum'),
            mean_of_mean=('mean', 'mean'),
            std_of_mean=('mean', 'std'),
            mean_of_std=('std', 'mean')
        )
        .reset_index()
    )
    return agg_df


def plot_aggregated(agg_df: pd.DataFrame, out_dir: str):
    """For each label, create an interactive Plotly plot of mean_of_mean vs temperature with a line per percentage."""
    os.makedirs(out_dir, exist_ok=True)
    labels = sorted(agg_df['label'].unique())
    for lab in labels:
        sub = agg_df[agg_df['label'] == lab].copy()
        if sub.empty:
            continue
        # Ensure temperature is numeric
        sub = sub.dropna(subset=['temperature', 'percentage'])
        sub['temperature'] = sub['temperature'].astype(float)
        sub['percentage'] = sub['percentage'].astype(float)
        
        # Create interactive Plotly figure
        fig = go.Figure()
        
        # Add a line for each percentage
        for pct in sorted(sub['percentage'].unique()):
            pct_data = sub[sub['percentage'] == pct].sort_values('temperature')
            fig.add_trace(go.Scatter(
                x=pct_data['temperature'],
                y=pct_data['mean_of_mean'],
                mode='lines+markers',
                name=f'{int(pct)}%',
                hovertemplate='<b>Temp: %{x}°C</b><br>Mean: %{y:.3f}<extra></extra>'
            ))
        label_name = LABEL_NAMES.get(int(lab), f'Label {lab}')
        fig.update_layout(
            title=f'Label {lab}: mean(topography) vs Temperature',
            xaxis_title='Temperature (°C)',
            yaxis_title='Mean of mean',
            hovermode='x unified',
            template='plotly_white',
            width=900,
            height=600
        )
        
        fname = os.path.join(out_dir, f'agg_label_{lab}.html')
        fig.write_html(fname)
        print(f'Saved interactive plot to {fname}')


def plot_combined_dashboard(agg_df: pd.DataFrame, out_dir: str):
    """Create a single combined dashboard with all labels as stacked subplots."""
    os.makedirs(out_dir, exist_ok=True)
    
    labels = sorted(agg_df['label'].unique())
    if not labels:
        print('No data to plot')
        return
    
    # Create subplots using Plotly's make_subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=len(labels), cols=1,
        subplot_titles=[LABEL_NAMES.get(int(lab), f'Label {lab}') for lab in labels],
        specs=[[{'secondary_y': False}] for _ in labels]
    )
    
    # Add traces for each label
    for row_idx, lab in enumerate(labels, start=1):
        sub = agg_df[agg_df['label'] == lab].copy()
        if sub.empty:
            continue
        
        sub = sub.dropna(subset=['temperature', 'percentage'])
        sub['temperature'] = sub['temperature'].astype(float)
        sub['percentage'] = sub['percentage'].astype(float)
        
        # Add a line for each percentage
        for pct in sorted(sub['percentage'].unique()):
            pct_data = sub[sub['percentage'] == pct].sort_values('temperature')
            fig.add_trace(
                go.Scatter(
                    x=pct_data['temperature'],
                    y=pct_data['mean_of_mean'],
                    mode='lines+markers',
                    name=f'{int(pct)}%',
                    legendgroup=f'{int(pct)}',  # Group by percentage for legend
                    showlegend=(row_idx == 1),  # Show legend only on first subplot
                    hovertemplate='<b>Temp: %{x}°C</b><br>Mean: %{y:.3f}<extra></extra>'
                ),
                row=row_idx, col=1
            )
        
        # Update y-axis label
        fig.update_yaxes(title_text='Mean of mean', row=row_idx, col=1)
    
    # Update x-axis label
    fig.update_xaxes(title_text='Temperature (°C)', row=len(labels), col=1)
    
    # Update overall layout
    fig.update_layout(
        title_text='Combined Dashboard: All Labels',
        hovermode='x unified',
        template='plotly_white',
        height=300 * len(labels),
        width=1000
    )
    
    fname = os.path.join(out_dir, 'combined_dashboard.html')
    fig.write_html(fname)
    print(f'Saved combined dashboard to {fname}')


def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Compute per-class stats from GWY + mask pairs")
    p.add_argument('gwy', nargs='?', help='Path to .gwy file (or directory with --batch)')
    p.add_argument('mask', nargs='?', help='Path to .npy mask file')
    p.add_argument('--out', '-o', help='Output CSV path for raw rows. If omitted and --batch used, default stats_from_mask_results.csv')
    p.add_argument('--channel', help='Channel name inside GWY to use (optional)')
    p.add_argument('--scale', type=float, default=1e10, help='Scale factor applied to GWY data (default 1e6)')
    p.add_argument('--batch', action='store_true', help='Treat first arg as a directory and process all .gwy files')
    p.add_argument('--mask-dir', help='Directory where masks live (when using --batch)')
    p.add_argument('--metadata-log', help='Path to CSV log with metadata (e.g. Light conditions)')
    p.add_argument('--save-pickle', action='store_true', help='Also save pandas DataFrame as pickle (.pkl)')
    p.add_argument('--plot-dir', help='Directory to save aggregated plots (if omitted no plots saved)')
    p.add_argument('--hist-dir', help='Directory to save per-file histograms (overlaid per label)')
    p.add_argument('--hist-bins', type=int, default=50,help='Number of bins for histograms (default 50)')
    args = p.parse_args(argv)

    # Load metadata log if provided
    metadata_dict = {}
    if args.metadata_log:
        metadata_dict = load_metadata_log(args.metadata_log)

    rows = []
    if args.batch:
        if not args.gwy or not os.path.isdir(args.gwy):
            raise SystemExit('When using --batch you must pass a directory as the first argument')
        for fn in sorted(os.listdir(args.gwy)):
            if not fn.lower().endswith('.gwy'):
                continue
            gwy_path = os.path.join(args.gwy, fn)
            mask_path = find_matching_mask(gwy_path, mask_dir=args.mask_dir)
            if mask_path is None:
                print(f'Warning: no mask found for {gwy_path}, skipping')
                continue
            try:
                rows.extend(
                    process_pair(
                        gwy_path,
                        mask_path,
                        channel=args.channel,
                        scale=args.scale,
                        metadata_dict=metadata_dict,
                        hist_dir=args.hist_dir,
                        hist_bins=args.hist_bins,
                    )
                )

            except Exception as e:
                print(f'Error processing {gwy_path} + {mask_path}: {e}')
    else:
        if not args.gwy or not args.mask:
            raise SystemExit('Provide both a .gwy path and a .npy mask path, or use --batch')
        rows = process_pair(
            args.gwy,
            args.mask,
            channel=args.channel,
            scale=args.scale,
            metadata_dict=metadata_dict,
            hist_dir=args.hist_dir,
            hist_bins=args.hist_bins,
        )

    # Build DataFrame
    df = pd.DataFrame(rows, columns=[
        'file', 'label', 'temperature', 'percentage',
        'count', 'mean', 'std', 'median', 'p25', 'p75', 'min', 'max', 'light_condition'
    ])

    # If running in batch mode and no --out provided, choose a sensible default filename
    default_out = args.out or ('stats_from_mask_results.csv' if args.batch else None)

    if default_out:
        df.to_csv(default_out, index=False)
        print(f'Wrote {len(df)} rows to {default_out}')

    if args.save_pickle:
        pkl_name = os.path.splitext(default_out)[0] + '.pkl' if default_out else 'stats_from_mask_results.pkl'
        df.to_pickle(pkl_name)
        print(f'Saved pandas DataFrame to {pkl_name}')

    # Aggregate and save aggregated table
    agg_df = aggregate_by_temp_pct(df)
    
    # Auto-create plots folder in batch mode if not specified
    plot_dir = args.plot_dir
    if args.batch and not plot_dir:
        plot_dir = 'plots'
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    
    agg_out = None
    if args.out:
        base_dir = os.path.dirname(args.out) or '.'
        agg_out = os.path.join(base_dir, 'aggregated_' + os.path.basename(args.out))
    else:
        # if default_out used (batch without --out), put aggregated next to it
        if args.batch:
            agg_out = 'aggregated_stats_from_mask_results.csv'
    if agg_out:
        agg_df.to_csv(agg_out, index=False)
        print(f'Wrote aggregated table to {agg_out}')

    # Plot aggregated results
    if plot_dir and not agg_df.empty:
        plot_aggregated(agg_df, plot_dir)
        plot_combined_dashboard(agg_df, plot_dir)
        print(f'Saved aggregated plots to {plot_dir}')

    # If no outputs specified, print to stdout
    if not default_out and not args.save_pickle:
        print(df.to_csv(index=False))


if __name__ == '__main__':
    main()