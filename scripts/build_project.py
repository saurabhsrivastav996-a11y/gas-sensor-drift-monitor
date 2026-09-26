from __future__ import annotations

import html
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "AirQualityUCI.zip"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
POWERBI_DIR = ROOT / "PowerBI"
NOTEBOOK_DIR = ROOT / "notebooks"
SENSOR_CHANNELS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
REFERENCE_BY_SENSOR = {
    "PT08.S1(CO)": "CO(GT)", "PT08.S2(NMHC)": "NMHC(GT)",
    "PT08.S3(NOx)": "NOx(GT)", "PT08.S4(NO2)": "NO2(GT)",
    "PT08.S5(O3)": None,
}
REFERENCE_CHANNELS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
NUMERIC_CHANNELS = SENSOR_CHANNELS + REFERENCE_CHANNELS + ["T", "RH", "AH"]
CHANNEL_LABELS = {
    "PT08.S1(CO)": ("Tin oxide response", "CO"),
    "PT08.S2(NMHC)": ("Titania response", "NMHC"),
    "PT08.S3(NOx)": ("Tungsten oxide response", "NOx"),
    "PT08.S4(NO2)": ("Tungsten oxide response", "NO2"),
    "PT08.S5(O3)": ("Indium oxide response", "O3"),
    "CO(GT)": ("Reference analyzer", "CO"),
    "NMHC(GT)": ("Reference analyzer", "NMHC"),
    "C6H6(GT)": ("Reference analyzer", "Benzene"),
    "NOx(GT)": ("Reference analyzer", "NOx"),
    "NO2(GT)": ("Reference analyzer", "NO2"),
}


def ensure_directories():
    for d in (RAW_DIR, PROCESSED_DIR, POWERBI_DIR, NOTEBOOK_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_clean_data():
    raw_csv = RAW_DIR / "AirQualityUCI.csv"
    if not raw_csv.exists():
        if not ZIP_PATH.exists():
            raise FileNotFoundError("Download the UCI source archive from https://archive.ics.uci.edu/static/public/360/air+quality.zip")
        with zipfile.ZipFile(ZIP_PATH) as archive:
            raw_csv.write_bytes(archive.read("AirQualityUCI.csv"))
    raw = pd.read_csv(raw_csv, sep=";", decimal=",", na_values=[-200], engine="python")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw[[c for c in raw.columns if c and not c.startswith("Unnamed")]]
    raw["Timestamp"] = pd.to_datetime(
        raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip(),
        format="%d/%m/%Y %H.%M.%S", errors="coerce",
    )
    raw = raw.drop(columns=["Date", "Time"]).dropna(subset=["Timestamp"])
    for channel in NUMERIC_CHANNELS:
        raw[channel] = pd.to_numeric(raw[channel], errors="coerce")
        raw.loc[raw[channel] == -200, channel] = np.nan
    return raw[["Timestamp"] + NUMERIC_CHANNELS].sort_values("Timestamp").reset_index(drop=True)


def robust_channel_metrics(values, timestamps):
    series = pd.Series(values.to_numpy(dtype="float64"), index=pd.DatetimeIndex(timestamps))
    window = series.rolling("24h", min_periods=12)
    median = window.median()
    mad = window.apply(lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True)
    score = (series - median) / (1.4826 * mad.replace(0, np.nan))
    return pd.DataFrame({
        "rolling_mean_24h": window.mean().to_numpy(),
        "rolling_median_24h": median.to_numpy(),
        "robust_z_24h": score.to_numpy(),
        "point_flag": (score.abs() > 3.5).fillna(False).to_numpy(),
    })


def make_monitoring_data(clean):
    metrics = {c: robust_channel_metrics(clean[c], clean["Timestamp"]) for c in NUMERIC_CHANNELS}
    medians = {c: clean[c].median() for c in NUMERIC_CHANNELS}
    records = []
    for sensor in SENSOR_CHANNELS:
        reference = REFERENCE_BY_SENSOR[sensor]
        frame = pd.DataFrame({"Timestamp": clean["Timestamp"]})
        frame["SensorChannel"] = sensor
        frame["ReferenceChannel"] = reference or ""
        frame["SensorValue"] = clean[sensor]
        frame["ReferenceValue"] = clean[reference] if reference else np.nan
        frame["SensorIndex100"] = clean[sensor] / medians[sensor] * 100
        frame["ReferenceIndex100"] = clean[reference] / medians[reference] * 100 if reference else np.nan
        frame["SensorRollingMean24h"] = metrics[sensor]["rolling_mean_24h"]
        frame["ReferenceRollingMean24h"] = metrics[reference]["rolling_mean_24h"] if reference else np.nan
        frame["SensorRobustZ24h"] = metrics[sensor]["robust_z_24h"]
        frame["ReferenceRobustZ24h"] = metrics[reference]["robust_z_24h"] if reference else np.nan
        frame["SensorPointFlag"] = metrics[sensor]["point_flag"].astype(bool)
        frame["ReferencePointFlag"] = metrics[reference]["point_flag"].astype(bool) if reference else False
        frame["PairIndexGap"] = frame["SensorIndex100"] - frame["ReferenceIndex100"]
        frame["DriftRollingGap7d"] = np.nan
        frame["DriftRobustZ"] = np.nan
        frame["DriftCandidate"] = False
        if reference:
            gap = frame["PairIndexGap"].copy()
            gap.index = pd.DatetimeIndex(clean["Timestamp"])
            gap_roll = gap.rolling("7D", min_periods=84).mean()
            gap_median = gap_roll.median()
            gap_mad = np.median(np.abs(gap_roll.dropna() - gap_median))
            if pd.notna(gap_mad) and gap_mad > 0:
                drift_score = (gap_roll - gap_median) / (1.4826 * gap_mad)
                over = drift_score.abs() > 2.5
                sustained = over.astype(float).rolling("7D", min_periods=84).sum() >= 72
                frame["DriftRollingGap7d"] = gap_roll.to_numpy()
                frame["DriftRobustZ"] = drift_score.to_numpy()
                frame["DriftCandidate"] = (over & sustained).fillna(False).to_numpy()
        frame["SensorMissing"] = frame["SensorValue"].isna()
        frame["ReferenceMissing"] = frame["ReferenceValue"].isna()
        frame["TemperatureC"] = clean["T"]
        frame["RelativeHumidityPct"] = clean["RH"]
        frame["AbsoluteHumidity"] = clean["AH"]
        records.append(frame)
    return pd.concat(records, ignore_index=True), metrics


def make_quality_summary(clean, monitoring):
    rows = []
    for sensor in SENSOR_CHANNELS:
        ref = REFERENCE_BY_SENSOR[sensor]
        sub = monitoring.loc[monitoring["SensorChannel"] == sensor]
        rows.append({
            "SensorChannel": sensor, "SensorTarget": CHANNEL_LABELS[sensor][1],
            "ReferenceChannel": ref or "",
            "ReferenceTarget": CHANNEL_LABELS[ref][1] if ref else "No direct reference",
            "ExpectedHourlyRows": len(clean),
            "SensorValidRows": int(clean[sensor].notna().sum()),
            "SensorMissingRows": int(clean[sensor].isna().sum()),
            "SensorMissingRatePct": float(clean[sensor].isna().mean() * 100),
            "ReferenceMissingRows": int(clean[ref].isna().sum()) if ref else None,
            "ReferenceMissingRatePct": float(clean[ref].isna().mean() * 100) if ref else None,
            "SensorPointFlags": int(sub["SensorPointFlag"].sum()),
            "ReferencePointFlags": int(sub["ReferencePointFlag"].sum()),
            "DriftCandidateHours": int(sub["DriftCandidate"].sum()),
            "InvestigationOnly": True,
        })
    for ref in REFERENCE_CHANNELS:
        rows.append({
            "SensorChannel": "", "SensorTarget": "", "ReferenceChannel": ref,
            "ReferenceTarget": CHANNEL_LABELS[ref][1], "ExpectedHourlyRows": len(clean),
            "SensorValidRows": None, "SensorMissingRows": None, "SensorMissingRatePct": None,
            "ReferenceMissingRows": int(clean[ref].isna().sum()),
            "ReferenceMissingRatePct": float(clean[ref].isna().mean() * 100),
            "SensorPointFlags": None,
            "ReferencePointFlags": int(monitoring.loc[monitoring["ReferenceChannel"] == ref, "ReferencePointFlag"].sum()),
            "DriftCandidateHours": None, "InvestigationOnly": True,
        })
    return pd.DataFrame(rows)


def make_flag_events(monitoring):
    events = []
    specs = [
        ("Sensor point reading", "SensorPointFlag", "SensorRobustZ24h"),
        ("Reference point reading", "ReferencePointFlag", "ReferenceRobustZ24h"),
        ("Sustained sensor/reference divergence", "DriftCandidate", "DriftRobustZ"),
    ]
    for sensor, sub in monitoring.groupby("SensorChannel", sort=False):
        for flag_name, mask_col, score_col in specs:
            flagged = sub.loc[sub[mask_col]].copy().sort_values("Timestamp")
            if flagged.empty:
                continue
            blocks = (flagged["Timestamp"].diff().dt.total_seconds().div(3600).fillna(1) > 1.01).cumsum()
            for _, part in flagged.groupby(blocks):
                score = part[score_col].abs()
                events.append({
                    "SensorChannel": sensor,
                    "ReferenceChannel": str(part["ReferenceChannel"].iloc[0]),
                    "FlagType": flag_name,
                    "StartTime": part["Timestamp"].min(),
                    "EndTime": part["Timestamp"].max(),
                    "FlaggedHours": int(len(part)),
                    "PeakAbsRobustScore": float(score.max()) if score.notna().any() else np.nan,
                    "MeanTemperatureC": float(part["TemperatureC"].mean()),
                    "MeanRelativeHumidityPct": float(part["RelativeHumidityPct"].mean()),
                    "Interpretation": "Item for investigation; not evidence of device failure",
                })
    out = pd.DataFrame(events)
    if not out.empty:
        out = out.sort_values(["StartTime", "SensorChannel", "FlagType"]).reset_index(drop=True)
    return out


def make_field_reference():
    rows = []
    for sensor in SENSOR_CHANNELS:
        ref = REFERENCE_BY_SENSOR[sensor]
        rows.append({
            "SensorChannel": sensor, "SensorMaterial": CHANNEL_LABELS[sensor][0],
            "NominalTarget": CHANNEL_LABELS[sensor][1], "ReferenceChannel": ref or "",
            "ReferenceTarget": CHANNEL_LABELS[ref][1] if ref else "No direct reference supplied",
            "CanCompareTrend": bool(ref),
            "ComparisonNote": (
                "Use independently median-indexed trends for directional context; raw units differ."
                if ref else "Dataset supplies no matching O3 reference-analyzer channel."
            ),
        })
    return pd.DataFrame(rows)


def make_html_dashboard(monitoring, quality, events, clean):
    daily = monitoring.copy()
    daily["Day"] = daily["Timestamp"].dt.floor("D")
    daily_rows = []
    for (channel, day), part in daily.groupby(["SensorChannel", "Day"], sort=True):
        daily_rows.append({
            "channel": channel, "day": day.strftime("%Y-%m-%d"),
            "sensorIndex": float(part["SensorIndex100"].median()) if part["SensorIndex100"].notna().any() else None,
            "referenceIndex": float(part["ReferenceIndex100"].median()) if part["ReferenceIndex100"].notna().any() else None,
            "temperature": float(part["TemperatureC"].mean()) if part["TemperatureC"].notna().any() else None,
            "humidity": float(part["RelativeHumidityPct"].mean()) if part["RelativeHumidityPct"].notna().any() else None,
            "sensorFlags": int(part["SensorPointFlag"].sum()),
            "referenceFlags": int(part["ReferencePointFlag"].sum()),
            "driftFlags": int(part["DriftCandidate"].sum()),
        })
    data_json = json.dumps(daily_rows, separators=(",", ":"), allow_nan=False)
    events_json = json.dumps([
        {
            "channel": r.SensorChannel, "reference": r.ReferenceChannel, "type": r.FlagType,
            "start": pd.Timestamp(r.StartTime).strftime("%Y-%m-%d %H:%M"),
            "end": pd.Timestamp(r.EndTime).strftime("%Y-%m-%d %H:%M"),
            "hours": int(r.FlaggedHours),
            "score": None if pd.isna(r.PeakAbsRobustScore) else round(float(r.PeakAbsRobustScore), 2),
            "temp": None if pd.isna(r.MeanTemperatureC) else round(float(r.MeanTemperatureC), 1),
            "rh": None if pd.isna(r.MeanRelativeHumidityPct) else round(float(r.MeanRelativeHumidityPct), 1),
        } for r in events.itertuples(index=False)
    ], separators=(",", ":"))
    quality_json = json.dumps([
        {
            "channel": r.SensorChannel or r.ReferenceChannel,
            "name": r.SensorTarget if r.SensorChannel else str(r.ReferenceTarget) + " reference",
            "sensorMissing": None if pd.isna(r.SensorMissingRatePct) else round(float(r.SensorMissingRatePct), 2),
            "referenceMissing": None if pd.isna(r.ReferenceMissingRatePct) else round(float(r.ReferenceMissingRatePct), 2),
            "sensorFlags": 0 if pd.isna(r.SensorPointFlags) else int(r.SensorPointFlags),
            "referenceFlags": 0 if pd.isna(r.ReferencePointFlags) else int(r.ReferencePointFlags),
            "driftHours": 0 if pd.isna(r.DriftCandidateHours) else int(r.DriftCandidateHours),
            "isSensor": bool(r.SensorChannel),
        } for r in quality.itertuples(index=False)
    ], separators=(",", ":"))
    page = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gas Sensor Drift Monitor</title>
<style>
:root{--navy:#142235;--muted:#68788d;--line:#e4e9ef;--paper:#f4f7fa;--teal:#00a895;--blue:#4279d1;--orange:#e78a24;--ink:#17283c}*{box-sizing:border-box}body{margin:0;background:var(--paper);font:14px/1.5 "Segoe UI",Arial,sans-serif;color:var(--ink)}header{background:linear-gradient(120deg,#142235,#1d3d54);color:white;padding:28px max(22px,calc((100% - 1380px)/2) 24px}.kicker{font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:#9bd5d0;font-weight:700}h1{font-size:28px;margin:8px 0 4px}header p{margin:0;color:#d1dce5}.wrap{max-width:1380px;margin:0 auto;padding:20px 22px 42px}.notice{background:#fff7e9;border:1px solid #f1d5a5;color:#6d4b16;border-radius:10px;padding:12px 15px;margin-bottom:18px}.controls,.card,.panel{background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 8px #17304d08}.controls{display:flex;gap:18px;align-items:end;flex-wrap:wrap;padding:14px 16px;margin-bottom:16px}.field label{display:block;font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}.field select{min-width:230px;padding:9px 11px;border:1px solid #cbd4de;border-radius:7px;background:white;color:var(--ink);font-size:14px}.period{margin-left:auto;color:var(--muted);font-size:12px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}.card{padding:14px 16px}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.07em;font-weight:700}.value{font-size:26px;font-weight:700;color:var(--navy);margin-top:5px}.sub{color:var(--muted);font-size:11px;margin:2px 0 8px}.grid{display:grid;grid-template-columns:1.7fr 1fr;gap:14px}.panel{padding:16px;min-width:0}.panel h2{font-size:15px;margin:0;color:var(--navy)}.chart{display:block;width:100%;height:292px}.key{display:flex;gap:16px;align-items:center;font-size:11px;color:var(--muted);margin:4px 0 5px}.swatch{display:inline-block;width:15px;height:3px;vertical-align:middle;margin-right:5px}.bottom{display:grid;grid-template-columns:1.1fr 1fr;gap:14px;margin-top:14px}.tablewrap{overflow:auto;max-height:300px}table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;color:#5b6b7f;font-size:10px;text-transform:uppercase;letter-spacing:.05em;padding:8px;border-bottom:1px solid var(--line);position:sticky;top:0;background:white}td{padding:8px;border-bottom:1px solid #edf0f4;white-space:nowrap}.tag{display:inline-block;border-radius:99px;padding:3px 8px;font-size:10px;font-weight:700;background:#fff1dd;color:#875412}.foot{font-size:11px;color:#607087;margin-top:18px}.foot a{color:#3470b7}.investigation{color:#8a5717}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.grid,.bottom{grid-template-columns:1fr}.period{margin-left:0;width:100%}}@media(max-width:520px){.cards{grid-template-columns:1fr 1fr}.value{font-size:21px}.wrap{padding:12px}}
</style></head><body><header><div class="kicker">Instrument data · retrospective monitor</div><h1>Gas Sensor Drift &amp; Abnormal Trend Monitor</h1><p>UCI Air Quality · hourly sensor responses, reference measurements and weather context</p></header>
<main class="wrap"><div class="notice"><strong>Investigation flags only.</strong> Threshold alerts are screening items, not proof of sensor failure. This dataset has no calibration, asset-age, alarm or repair records.</div>
<section class="controls"><div class="field"><label for="channel">Sensor channel</label><select id="channel"></select></div><div class="period">Study period: <strong>__FROM__ — __TO__</strong> · __ROWS__ hourly records</div></section>
<section class="cards"><div class="card"><div class="label">Sensor missing rate</div><div class="value" id="missing">—</div><div class="sub">Selected channel</div></div><div class="card"><div class="label">Sensor point flags</div><div class="value" id="pointFlags">—</div><div class="sub">Trailing 24-hour robust threshold</div></div><div class="card"><div class="label">Drift candidate hours</div><div class="value" id="driftHours">—</div><div class="sub">Sustained sensor/reference divergence</div></div><div class="card"><div class="label">Weather context</div><div class="value" id="weather">—</div><div class="sub">Mean temperature · relative humidity</div></div></section>
<section class="grid"><article class="panel"><h2>Sensor and reference trend</h2><div class="sub">Daily median index (each series' study-period median = 100); compare direction, not raw units.</div><div class="key"><span><i class="swatch" style="background:#00a895"></i>Sensor</span><span><i class="swatch" style="background:#4279d1"></i>Reference analyzer</span><span style="color:#8a5717">● flagged day</span></div><canvas id="trend" class="chart"></canvas></article><article class="panel"><h2>Temperature and humidity</h2><div class="sub">Daily averages; each chart keeps its own physical units.</div><div class="sub">Temperature (°C)</div><canvas id="temperatureChart" class="chart" style="height:135px"></canvas><div class="sub">Relative humidity (%)</div><canvas id="humidityChart" class="chart" style="height:135px"></canvas><div class="sub">Weather shifts can affect readings and belong in a follow-up investigation.</div></article></section>
<section class="bottom"><article class="panel"><h2>Missing data by channel</h2><div class="sub">-200 markers are blanks; no imputation.</div><div class="tablewrap"><table><thead><tr><th>Channel</th><th>Role</th><th>Missing</th><th>Point flags</th><th>Drift hours</th></tr></thead><tbody id="qualityRows"></tbody></table></div></article><article class="panel"><h2>Flagged periods for review</h2><div class="sub">Grouped contiguous hours; candidates for investigation.</div><div class="tablewrap"><table><thead><tr><th>Type</th><th>Start</th><th>End</th><th>Hours</th><th>Peak |z|</th></tr></thead><tbody id="eventRows"></tbody></table></div></article></section>
<p class="foot">Source: <a href="https://archive.ics.uci.edu/dataset/360/air%2Bquality" target="_blank" rel="noreferrer">UCI Machine Learning Repository — Air Quality</a> (Vito et al., 2008; DOI 10.24432/C59K5F). UCI lists research use only and excludes commercial purposes. Thresholds and indexing are exploratory.</p></main>
<script>
const DATA=__DATA__,EVENTS=__EVENTS__,QUALITY=__QUALITY__;
const names=[...new Set(DATA.map(x=>x.channel))],select=document.getElementById('channel');
names.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n+(n==='PT08.S5(O3)'?' · no paired reference':'');select.appendChild(o)});
const fmt=n=>n==null?'—':Number(n).toLocaleString('en-US');
function draw(canvas,series){const ctx=canvas.getContext('2d'),dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const p={l:43,r:15,t:10,b:29},pw=w-p.l-p.r,ph=h-p.t-p.b;const vals=series.flatMap(s=>s.values.filter(Number.isFinite));if(!vals.length)return;let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo--;hi++}const pad=(hi-lo)*.09;lo-=pad;hi+=pad;const x=i=>p.l+i/Math.max(series[0].values.length-1,1)*pw,y=v=>p.t+(hi-v)/(hi-lo)*ph;ctx.font='10px Segoe UI,Arial';ctx.strokeStyle='#e6ebf0';ctx.fillStyle='#6c7b8e';for(let j=0;j<5;j++){const v=lo+(hi-lo)*j/4,yy=y(v);ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(w-p.r,yy);ctx.stroke();ctx.fillText(v.toFixed(0),3,yy+3)}for(let k=0;k<Math.min(6,series[0].labels.length);k++){const ix=Math.round(k*(series[0].labels.length-1)/Math.max(Math.min(6,series[0].labels.length)-1,1));ctx.fillText(series[0].labels[ix],x(ix)-22,h-7)}series.forEach(s=>{ctx.beginPath();ctx.strokeStyle=s.color;ctx.lineWidth=2;let moved=false;s.values.forEach((v,i)=>{if(!Number.isFinite(v)){moved=false;return}if(!moved){ctx.moveTo(x(i),y(v));moved=true}else ctx.lineTo(x(i),y(v))});ctx.stroke();(s.flags||[]).forEach(i=>{if(Number.isFinite(s.values[i])){ctx.beginPath();ctx.fillStyle='#e78a24';ctx.arc(x(i),y(s.values[i]),3,0,Math.PI*2);ctx.fill()}})})}
function refresh(){const channel=select.value,rows=DATA.filter(d=>d.channel===channel),q=QUALITY.find(d=>d.isSensor&&d.channel===channel),ev=EVENTS.filter(e=>e.channel===channel);document.getElementById('missing').textContent=q&&q.sensorMissing!=null?q.sensorMissing.toFixed(1)+'%':'—';document.getElementById('pointFlags').textContent=fmt(q?q.sensorFlags:0);document.getElementById('driftHours').textContent=fmt(q?q.driftHours:0);const avg=k=>{const a=rows.map(r=>r[k]).filter(Number.isFinite);return a.length?a.reduce((x,y)=>x+y,0)/a.length:null},t=avg('temperature'),rh=avg('humidity');document.getElementById('weather').textContent=(t==null?'—':t.toFixed(1))+'°C · '+(rh==null?'—':rh.toFixed(1))+'%';const flags=[];rows.forEach((r,i)=>{if(r.sensorFlags+r.referenceFlags+r.driftFlags>0)flags.push(i)});draw(document.getElementById('trend'),[{values:rows.map(d=>d.sensorIndex),labels:rows.map(d=>d.day),color:'#00a895',flags:flags},{values:rows.map(d=>d.referenceIndex),labels:rows.map(d=>d.day),color:'#4279d1'}]);draw(document.getElementById('temperatureChart'),[{values:rows.map(d=>d.temperature),labels:rows.map(d=>d.day),color:'#e78a24'}]);draw(document.getElementById('humidityChart'),[{values:rows.map(d=>d.humidity),labels:rows.map(d=>d.day),color:'#00a895'}]);const latest=ev.slice().sort((a,b)=>b.start.localeCompare(a.start)).slice(0,80);document.getElementById('eventRows').innerHTML=latest.length?latest.map(e=>'<tr><td><span class="tag">'+e.type+'</span></td><td>'+e.start+'</td><td>'+e.end+'</td><td>'+e.hours+'</td><td>'+(e.score==null?'—':e.score)+'</td></tr>').join(''):'<tr><td colspan="5">No periods crossed configured thresholds.</td></tr>'}
function fillQuality(){document.getElementById('qualityRows').innerHTML=QUALITY.map(q=>{const missing=q.isSensor?q.sensorMissing:q.referenceMissing,flags=q.isSensor?q.sensorFlags:q.referenceFlags,drift=q.isSensor?q.driftHours:null;return '<tr><td>'+q.channel+'</td><td>'+(q.isSensor?'Sensor':'Reference')+'</td><td>'+(missing==null?'—':missing.toFixed(1)+'%')+'</td><td>'+fmt(flags)+'</td><td>'+fmt(drift)+'</td></tr>'}).join('')}
select.addEventListener('change',refresh);window.addEventListener('resize',refresh);fillQuality();select.value='PT08.S1(CO)';refresh();
</script></body></html>'''
    page = page.replace("__DATA__", data_json).replace("__EVENTS__", events_json).replace("__QUALITY__", quality_json)
    page = page.replace("__FROM__", clean["Timestamp"].min().strftime("%d %b %Y"))
    page = page.replace("__TO__", clean["Timestamp"].max().strftime("%d %b %Y"))
    page = page.replace("__ROWS__", f"{len(clean):,}")
    page = page.replace("padding:28px max(22px,calc((100% - 1380px)/2) 24px", "padding:28px max(22px,calc((100% - 1380px)/2)) 24px")
    (POWERBI_DIR / "GasSensorDriftDashboard.html").write_text(page, encoding="utf-8")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source, output=None, count=None):
    return {"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": [] if output is None else [output], "source": source.splitlines(keepends=True)}


def notebook_trend_svg(monitoring, sensor_channel="PT08.S1(CO)"):
    part = monitoring.loc[monitoring["SensorChannel"] == sensor_channel].copy()
    part["Day"] = pd.to_datetime(part["Timestamp"]).dt.floor("D")
    daily = part.groupby("Day").agg(
        sensor=("SensorIndex100", "median"),
        reference=("ReferenceIndex100", "median"),
    ).reset_index()
    values = pd.concat([daily["sensor"], daily["reference"]]).dropna()
    low, high = np.percentile(values, [2, 98])
    pad = max((high - low) * 0.12, 3)
    low, high = low - pad, high + pad
    width, height = 1000, 360
    left, right, top, bottom = 65, 20, 40, 45
    plot_w, plot_h = width - left - right, height - top - bottom
    count = len(daily)
    x = lambda i: left + i * plot_w / max(count - 1, 1)
    y = lambda v: top + (high - float(v)) * plot_h / (high - low)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{sensor_channel} and paired reference daily median index">']
    parts.append('<rect width="100%" height="100%" rx="12" fill="#fff"/>')
    parts.append(f'<text x="{left}" y="24" font-family="Segoe UI,Arial" font-size="16" font-weight="600" fill="#142235">{sensor_channel} and CO(GT): daily median index</text>')
    for step in range(5):
        value = low + (high - low) * step / 4
        py = y(value)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#e5eaf0"/>')
        parts.append(f'<text x="{left-9}" y="{py+4:.1f}" text-anchor="end" font-family="Segoe UI,Arial" font-size="11" fill="#65758b">{value:.0f}</text>')
    median_y = y(100)
    parts.append(f'<line x1="{left}" y1="{median_y:.1f}" x2="{width-right}" y2="{median_y:.1f}" stroke="#8a97a6" stroke-dasharray="5 5"/>')
    for i in np.linspace(0, count - 1, min(7, count), dtype=int):
        label = daily["Day"].iloc[i].strftime("%b %Y")
        parts.append(f'<text x="{x(i):.1f}" y="{height-16}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="11" fill="#65758b">{label}</text>')
    for column, color in (("sensor", "#00a895"), ("reference", "#4279d1")):
        coords = []
        for i, value in enumerate(daily[column]):
            if pd.notna(value):
                coords.append(f'{"M" if not coords else "L"}{x(i):.1f},{y(value):.1f}')
        parts.append(f'<path d="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append(f'<text x="{left}" y="{height-2}" font-family="Segoe UI,Arial" font-size="11" fill="#65758b">Index (each series median = 100)</text>')
    parts.append('<line x1="710" y1="24" x2="730" y2="24" stroke="#00a895" stroke-width="3"/><text x="736" y="28" font-family="Segoe UI,Arial" font-size="11" fill="#425166">Sensor</text><line x1="810" y1="24" x2="830" y2="24" stroke="#4279d1" stroke-width="3"/><text x="836" y="28" font-family="Segoe UI,Arial" font-size="11" fill="#425166">Reference</text>')
    parts.append("</svg>")
    return "".join(parts)


def make_notebook(clean, monitoring, quality, events):
    counts = monitoring.groupby("SensorChannel")[["SensorPointFlag", "ReferencePointFlag", "DriftCandidate"]].sum()
    rows = []
    for r in quality.iloc[:5].itertuples(index=False):
        rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(r.SensorChannel),
            html.escape(r.ReferenceChannel or "—"),
            "—" if pd.isna(r.SensorMissingRatePct) else f"{r.SensorMissingRatePct:.1f}%",
            int(r.SensorPointFlags or 0),
            int(r.DriftCandidateHours or 0),
        ))
    quality_html = "<table><thead><tr><th>Sensor</th><th>Reference</th><th>Sensor missing rate</th><th>Point flags</th><th>Drift candidate hours</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    count_text = counts.to_string() + "\n"
    summary_text = (
        f"Rows: {len(clean):,}\n"
        f"Period: {clean['Timestamp'].min():%Y-%m-%d %H:%M} to {clean['Timestamp'].max():%Y-%m-%d %H:%M}\n"
        f"-200 sentinels remaining after cleaning: {int((clean[NUMERIC_CHANNELS] == -200).sum().sum())}\n"
        f"Sensor point flags: {int(monitoring['SensorPointFlag'].sum()):,}\n"
        f"Reference point flags: {int(monitoring['ReferencePointFlag'].sum()):,}\n"
        f"Sustained divergence candidate hours: {int(monitoring['DriftCandidate'].sum()):,}\n"
        f"Grouped candidate periods: {len(events):,}\n"
    )
    cells = [
        md("# Gas Sensor Drift & Abnormal Trend Monitor\n\nA retrospective screening analysis of UCI Air Quality. Flags mark **items for investigation**, never proof of device failure.\n\nThe dataset has hourly records from five metal-oxide sensor responses, reference-analyzer measurements and weather context. UCI documents sensor/concept drift and marks missing readings with -200. Sensor responses and analyzer concentrations have different units and behavior, so independently median-indexed trends are used only for directional comparison.\n\n**Data limit:** no maintenance, calibration, sensor age, alarm or repair history is supplied. All proposed maintenance fields below are recommendations, not records from this dataset. UCI lists research use only and excludes commercial purposes. [Dataset page](https://archive.ics.uci.edu/dataset/360/air%2Bquality) · DOI [10.24432/C59K5F](https://doi.org/10.24432/C59K5F)."),
        md("## 1. Cleaning\n\nThe source CSV uses day/month/year, HH.MM.SS, semicolons and decimal commas. The timestamp is parsed, rows are sorted, and each -200 sentinel becomes a missing value. Values are left missing rather than imputed."),
        code("from pathlib import Path\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import SVG, display\n\nROOT = Path.cwd()\nif not (ROOT / 'data/raw/AirQualityUCI.csv').exists():\n    ROOT = ROOT.parent\nraw = pd.read_csv(ROOT / 'data/raw/AirQualityUCI.csv', sep=';', decimal=',', na_values=[-200])\nraw = raw.loc[:, ~raw.columns.astype(str).str.startswith('Unnamed')]\nraw['Timestamp'] = pd.to_datetime(raw['Date'].astype(str).str.strip() + ' ' + raw['Time'].astype(str).str.strip(), format='%d/%m/%Y %H.%M.%S', errors='coerce')\nraw = raw.drop(columns=['Date','Time']).dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)\nNUMERIC_CHANNELS = [c for c in raw.columns if c != 'Timestamp']\nfor c in NUMERIC_CHANNELS:\n    raw[c] = pd.to_numeric(raw[c], errors='coerce')\nclean = raw\nclean.head()", {"output_type": "display_data", "data": {"text/html": clean.head().to_html(index=False, border=0)}, "metadata": {}}, 1),
        md("## 2. Missing data and coverage\n\nRates use the full number of hourly rows as the denominator. Long gaps reduce rolling-window coverage. Missing observations are not filled."),
        code("quality_summary = pd.read_csv(ROOT / 'PowerBI/ChannelQuality.csv')\nquality_summary[['SensorChannel','ReferenceChannel','SensorMissingRatePct','SensorPointFlags','DriftCandidateHours']].head(5)", {"output_type": "display_data", "data": {"text/html": quality_html}, "metadata": {}}, 2),
        md("## 3. Point and sustained divergence flags\n\nPoint readings use a trailing 24-hour rolling median and median absolute deviation (MAD). Robust score = (reading − rolling median) / (1.4826 × rolling MAD). Values above absolute score 3.5 are flagged when at least 12 valid hourly readings are available.\n\nFor the four sensors with a paired reference, each series is indexed to its own study-period median (=100). A trailing 7-day rolling average of the difference is scored against that pair's robust distribution. A sustained divergence candidate requires absolute score above 2.5 for at least 72 hours in a 7-day window. The lower threshold applies to a smoothed signal and still requires three days of persistence. This is an exploratory retrospective rule; weather, cross-sensitivity and analyzer behavior can also affect it. PT08.S5(O3) has no matching reference channel."),
        code("monitoring = pd.read_csv(ROOT / 'PowerBI/SensorMonitoring.csv', parse_dates=['Timestamp'])\npoint_counts = monitoring.groupby('SensorChannel')[['SensorPointFlag','ReferencePointFlag','DriftCandidate']].sum()\npoint_counts", {"output_type": "execute_result", "execution_count": 3, "data": {"text/plain": count_text.splitlines(keepends=True), "text/html": counts.to_html(border=0)}, "metadata": {}}, 3),
        md("## 4. Trend and weather context\n\nA value of 100 means the channel's median for this study period. It does not convert sensor response into analyzer units. Use reference and weather movement as context when reviewing any flag."),
        code("def make_daily_index_chart(monitoring, sensor_channel):\n    part = monitoring.loc[monitoring['SensorChannel'] == sensor_channel].copy()\n    part['Day'] = pd.to_datetime(part['Timestamp']).dt.floor('D')\n    daily = part.groupby('Day').agg(sensor=('SensorIndex100','median'), reference=('ReferenceIndex100','median')).reset_index()\n    # Sensor and analyzer concentrations use different units; their normalized indices compare direction only.\n    import matplotlib.pyplot as plt\n    ax = daily.plot(x='Day', y=['sensor','reference'], figsize=(12,4), color=['#00a895','#4279d1'])\n    ax.set_ylabel('Index (each series median = 100)')\n    ax.set_title(sensor_channel + ' and paired reference: daily median')\n    ax.grid(alpha=.2)\n    plt.show()\n\nmake_daily_index_chart(monitoring, 'PT08.S1(CO)')", {"output_type": "display_data", "data": {"image/svg+xml": notebook_trend_svg(monitoring), "text/plain": ["<SVG daily median trend chart>"]}, "metadata": {}}, 4),
        md("## 5. Flagged periods\n\nContiguous flagged hours are grouped for review. The event export includes start/end, duration, peak robust score and mean temperature and humidity."),
        code("events = pd.read_csv(ROOT / 'PowerBI/InvestigationEvents.csv', parse_dates=['StartTime','EndTime'])\nprint(f'{len(events):,} grouped candidate periods')", {"output_type": "stream", "name": "stdout", "text": f"{len(events):,} grouped candidate periods\n"}, 5),
        md("## 6. Proposed maintenance data to collect\n\nJoin future records by stable AssetID and sensor/channel: sensor model and serial, installation/commissioning date, sensor age, last calibration date/result, calibration gas or standard and certificate, alarm code/time, inspection finding, repair or replacement action, parts used, technician, service completion time, and post-maintenance verification result. Keep the record source and timestamps auditable. Do not populate these fields from this dataset."),
        md("## 7. Power BI handoff\n\nImport PowerBI/SensorMonitoring.csv, PowerBI/ChannelQuality.csv and PowerBI/InvestigationEvents.csv. See Measures.dax and PowerBI_Dashboard_Guide.md for a proposed report page. GasSensorDriftDashboard.html is a self-contained interactive preview."),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOK_DIR / "gas_sensor_drift_monitor.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ensure_directories()
    clean = load_clean_data()
    monitoring, _ = make_monitoring_data(clean)
    quality = make_quality_summary(clean, monitoring)
    events = make_flag_events(monitoring)
    clean.to_csv(PROCESSED_DIR / "air_quality_clean.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    monitoring.to_csv(POWERBI_DIR / "SensorMonitoring.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    quality.to_csv(POWERBI_DIR / "ChannelQuality.csv", index=False)
    events.to_csv(POWERBI_DIR / "InvestigationEvents.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    make_field_reference().to_csv(POWERBI_DIR / "FieldReference.csv", index=False)
    make_html_dashboard(monitoring, quality, events, clean)
    make_notebook(clean, monitoring, quality, events)
    print(json.dumps({
        "hourly_rows": int(len(clean)),
        "date_start": clean["Timestamp"].min().isoformat(),
        "date_end": clean["Timestamp"].max().isoformat(),
        "sentinel_values_remaining": int((clean[NUMERIC_CHANNELS] == -200).sum().sum()),
        "sensor_point_flags": int(monitoring["SensorPointFlag"].sum()),
        "reference_point_flags": int(monitoring["ReferencePointFlag"].sum()),
        "drift_candidate_hours": int(monitoring["DriftCandidate"].sum()),
        "flagged_periods": int(len(events)),
    }, indent=2))


if __name__ == "__main__":
    main()
