"""Live forecast for the Beas basin: ECMWF IFS 0.25 deg (Open-Meteo) -> predictors -> trained LightGBM models.
Writes ../forecast.json. Run daily (GitHub Actions) or by hand: python forecast/run_forecast.py"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import forecast_common as F

HERE = Path(__file__).resolve().parent
F.CELLS = HERE / "cells.csv"
URL = "https://api.open-meteo.com/v1/forecast"


def main(out=HERE.parent / "forecast.json", forecast_days=8):
    meta = json.loads((HERE / "meta.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    h = F.fetch(URL, {"models": "ecmwf_ifs025", "past_days": 1, "forecast_days": forecast_days})
    d = F.daily_features(h)
    today_ist = (now + timedelta(hours=5, minutes=30)).date()
    d = d[d.date.dt.date > today_ist]  # IMD day D ends 08:30 IST on D; keep days not yet finished
    d = F.add_static(F.apply_qmap(d, meta["qmap"]))
    cells = pd.read_csv(F.CELLS)
    for key, m in meta["models"].items():
        booster = lgb.Booster(model_file=str(HERE / m["file"]))
        d["p_" + key] = booster.predict(d[meta["features"]].values)
    days = []
    for date, g in d.groupby("date"):
        g = g.set_index(["lat", "lon"]).reindex(list(zip(cells.lat, cells.lon)))
        days.append({"date": date.strftime("%Y-%m-%d"), "lead_days": (date.date() - today_ist).days,
                     "in_season": date.month in (6, 7, 8, 9),
                     "p_extreme": np.round(g.p_extreme.values, 4).tolist(), "p_cloudburst": np.round(g.p_cloudburst.values, 4).tolist(),
                     "ivt": np.round(g.ivt_max.values, 1).tolist(), "ivt_dir": np.round(g.ivt_toward_deg.values, 0).tolist(),
                     "tcwv": np.round(g.tcwv_max.values, 1).tolist()})
    res = {"issued_utc": now.strftime("%Y-%m-%dT%H:%MZ"), "source": "ECMWF IFS 0.25 deg open-data forecast via Open-Meteo",
           "alarm": {k: m["threshold"] for k, m in meta["models"].items()}, "models": {k: m["name"] for k, m in meta["models"].items()},
           "cells": [[float(a), float(b)] for a, b in zip(cells.lat, cells.lon)], "days": days}
    Path(out).write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    print(f"forecast.json: issued {res['issued_utc']}, {len(days)} IMD days {days[0]['date'] if days else ''}..{days[-1]['date'] if days else ''}")
    return res


if __name__ == "__main__":
    main()
