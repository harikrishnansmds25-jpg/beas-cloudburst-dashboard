"""Shared code for the forecast model: fetch ECMWF IFS 0.25 deg forecasts (via the Open-Meteo API) for the
31 Beas IMD cells and turn them into the same daily predictors the ERA5 training data uses.

Daily predictors per IMD day D use the 06, 12, 18 UTC (D-1) and 00 UTC (D) values, exactly like step9.
Forecast-derived quantities:
  tcwv  = total_column_integrated_water_vapour (mm)
  cape  = CAPE (J/kg)
  kx    = K-index from 850/700/500 hPa temperature and humidity: (T850 - T500) + Td850 - (T700 - Td700)
  ivt   = vertically integrated water-vapour flux, (1/g) * integral of q*u and q*v from the surface
          pressure up to 300 hPa over 1000-300 hPa levels (ERA5 uses its full-column flux; residual
          differences are removed by the quantile mapping in step19)
  t2m, rh2m (Magnus from 2 m dewpoint), sp (hPa)
Not available from the forecast and therefore not used by the forecast model: cloud liquid/ice water, convergence.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "outputs/model/imd_cells_terrain.csv"
UA = {"User-Agent": "beas-cloudburst-forecast/1.0 (academic research)"}
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300]
HOURLY = (["total_column_integrated_water_vapour", "cape", "temperature_2m", "dew_point_2m", "surface_pressure"]
          + [f"{v}_{p}hPa" for p in LEVELS for v in ("temperature", "relative_humidity", "wind_speed", "wind_direction")])
FC_ATMOS = ["tcwv_max", "tcwv_mean", "cape_max", "cape_mean", "kx_max", "kx_mean", "ivt_mean", "ivt_max", "ivt_u", "ivt_v",
            "t2m_mean", "t2m_range", "rh2m_mean", "rh2m_max", "sp_mean"]
STATIC = ["elevation_m", "slope_deg", "relief_std_m", "doy_sin", "doy_cos"]
FEATS = FC_ATMOS + STATIC
G = 9.80665


def cells():
    return pd.read_csv(CELLS)


def _get(url, p, tries=6):
    err = None
    for k in range(tries):
        try:
            r = requests.get(url, params=p, headers=UA, timeout=180)
            if r.status_code == 200:
                return r.json()
            err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            err = e
        time.sleep(4 * (k + 1))
    raise err


def fetch(url, params, batch=8):
    """Open-Meteo hourly data for all 31 cells (requested in small batches); long DataFrame (time, lat, lon, vars)."""
    c = cells()
    frames = []
    for s in range(0, len(c), batch):
        sub = c.iloc[s:s + batch]
        p = {"latitude": ",".join(f"{v:.2f}" for v in sub.lat), "longitude": ",".join(f"{v:.2f}" for v in sub.lon),
             "hourly": ",".join(HOURLY), "timezone": "UTC", "wind_speed_unit": "ms", "cell_selection": "nearest", **params}
        js = _get(url, p)
        js = js if isinstance(js, list) else [js]
        for (lat, lon), j in zip(zip(sub.lat, sub.lon), js):
            h = pd.DataFrame(j["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h["lat"], h["lon"] = lat, lon
            frames.append(h)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True)


def magnus_es(t_c):
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def dewpoint(t_c, rh):
    g = np.log(np.clip(rh, 1, 100) / 100) + 17.67 * t_c / (t_c + 243.5)
    return 243.5 * g / (17.67 - g)


def daily_features(h):
    """Hourly forecast rows -> daily predictors per IMD day (needs all 4 synoptic times)."""
    h = h[h.time.dt.hour.isin([0, 6, 12, 18])].copy()
    h["date"] = (h.time + pd.Timedelta(hours=21)).dt.floor("D")
    sp = h.surface_pressure.values
    t = {p: h[f"temperature_{p}hPa"].values for p in LEVELS}
    rh = {p: h[f"relative_humidity_{p}hPa"].values for p in LEVELS}
    td = {p: dewpoint(t[p], rh[p]) for p in (850, 700)}
    h["kx"] = (t[850] - t[500]) + td[850] - (t[700] - td[700])
    # integrated vapour flux from the surface (or 1000 hPa) to 300 hPa, levels below ground skipped
    qu, qv, pp = [], [], []
    for p in LEVELS:
        e = rh[p] / 100 * magnus_es(t[p])
        q = 0.622 * e / (p - 0.378 * e)
        ws, wd = h[f"wind_speed_{p}hPa"].values, np.radians(h[f"wind_direction_{p}hPa"].values)
        qu.append(q * -ws * np.sin(wd)); qv.append(q * -ws * np.cos(wd)); pp.append(np.full(len(h), float(p)))
    qu, qv, pp = np.array(qu), np.array(qv), np.array(pp)
    above = pp <= sp[None, :]
    iu, iv = np.zeros(len(h)), np.zeros(len(h))
    for k in range(len(LEVELS) - 1):  # trapezoid between consecutive levels that are both above ground
        ok = above[k] & above[k + 1]
        dp = (pp[k] - pp[k + 1]) * 100
        iu += np.where(ok, 0.5 * (qu[k] + qu[k + 1]) * dp, 0); iv += np.where(ok, 0.5 * (qv[k] + qv[k + 1]) * dp, 0)
    first = np.argmax(above, axis=0)  # lowest above-ground level: extend from the surface to it
    idx = np.arange(len(h))
    dps = np.clip(sp - pp[first, idx], 0, None) * 100
    iu += qu[first, idx] * dps; iv += qv[first, idx] * dps
    h["ivt_u"], h["ivt_v"] = iu / G, iv / G
    h["ivt"] = np.hypot(h.ivt_u, h.ivt_v)
    h["tcwv"] = h.total_column_integrated_water_vapour
    h["rh2m"] = np.clip(100 * magnus_es(h.dew_point_2m) / magnus_es(h.temperature_2m), 0, 100)
    agg = h.groupby(["date", "lat", "lon"]).agg(
        tcwv_mean=("tcwv", "mean"), tcwv_max=("tcwv", "max"), cape_max=("cape", "max"), cape_mean=("cape", "mean"),
        kx_max=("kx", "max"), kx_mean=("kx", "mean"), ivt_mean=("ivt", "mean"), ivt_max=("ivt", "max"),
        ivt_u=("ivt_u", "mean"), ivt_v=("ivt_v", "mean"), t2m_mean=("temperature_2m", "mean"),
        t2m_range=("temperature_2m", lambda x: x.max() - x.min()), rh2m_mean=("rh2m", "mean"), rh2m_max=("rh2m", "max"),
        sp_mean=("surface_pressure", "mean"), n=("tcwv", "count")).reset_index()
    agg = agg[agg.n == 4].drop(columns="n")
    agg["ivt_toward_deg"] = (np.degrees(np.arctan2(agg.ivt_u, agg.ivt_v)) + 360) % 360
    return agg


def add_static(df):
    df = df.merge(cells()[["lat", "lon", "elevation_m", "slope_deg", "relief_std_m", "hill_cell"]], on=["lat", "lon"], how="left")
    doy = pd.to_datetime(df.date).dt.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return df


def fit_qmap(fc, era, feats=FC_ATMOS):
    """Quantile mapping forecast -> ERA5 climate, pooled over cells (1st..99th percentiles)."""
    qs = np.linspace(0.005, 0.995, 100)
    return {f: {"fc": np.nanquantile(fc[f], qs).tolist(), "era": np.nanquantile(era[f], qs).tolist()} for f in feats}


def apply_qmap(df, qmap):
    df = df.copy()
    for f, m in qmap.items():
        xf, xe = np.array(m["fc"]), np.array(m["era"])
        x = df[f].values.astype(float)
        y = np.interp(x, xf, xe)
        lo, hi = x < xf[0], x > xf[-1]  # keep the tails' offsets beyond the fitted range
        y[lo] = x[lo] + (xe[0] - xf[0]); y[hi] = x[hi] + (xe[-1] - xf[-1])
        df[f] = y
    return df


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")
