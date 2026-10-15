import datetime as dt
import re
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- Settings (must match the notebook) ----------
SPIKE = 1500                      # grains/m³: NAB "Extreme" level for trees
LAT, LON = 33.749, -84.388        # Atlanta
WEATHER_VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                "relative_humidity_2m_mean", "wind_speed_10m_max",
                "wind_speed_10m_mean", "precipitation_sum"]
FEATURES = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "relative_humidity_2m_mean", "wind_speed_10m_mean", "precipitation_sum",
            "wind_speed_10m_max", "pollen_lag1", "sin_day", "cos_day"]

st.set_page_config(page_title="Atlanta Pollen Spike Predictor", page_icon="🌳")

# ---------- Data functions (same as the notebook) ----------
@st.cache_resource
def load_model():
    return joblib.load("pollen_model.joblib")

@st.cache_data(ttl=3600)          # re-download at most once an hour
def get_pollen_count(date):
    url = f"https://www.atlantaallergy.com/pollen_counts/index/{date:%Y/%m}"
    html = requests.get(url, timeout=30).text
    match = re.search(rf'pollen_counts/index/{date:%Y/%m/%d}">(\d+)<', html)
    return int(match.group(1)) if match else None

@st.cache_data(ttl=3600)
def get_forecast():
    params = {"latitude": LAT, "longitude": LON, "daily": WEATHER_VARS,
              "timezone": "America/New_York", "forecast_days": 3}
    data = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30).json()
    forecast = pd.DataFrame(data["daily"])
    forecast["date"] = pd.to_datetime(forecast["time"]).dt.date
    return forecast.drop(columns=["time"]).set_index("date")

model = load_model()

# ---------- Page ----------
st.title("🌳 Atlanta Pollen Spike Predictor")
st.write("Predicts whether pollen will reach the **Extreme** level (1,500+ grains/m³) "
         "so you can plan ahead.")

tab1, tab2, tab3 = st.tabs(["Tomorrow", "Replay a past day", "Season calendar"])

# ----- Tab 1: live prediction -----
with tab1:
    today = dt.datetime.now(ZoneInfo("America/New_York")).date()
    try:
        last_date, last_count = today, get_pollen_count(today)
        if last_count is None:
            last_date = today - dt.timedelta(days=1)
            last_count = get_pollen_count(last_date)

        if last_count is None:
            st.info("No pollen count was posted today or yesterday (the station doesn't "
                    "count most weekends). Check back after the next count.")
        else:
            target = last_date + dt.timedelta(days=1)
            day = pd.Timestamp(target).day_of_year
            row = get_forecast().loc[target].to_dict()
            row["pollen_lag1"] = last_count
            row["sin_day"] = np.sin(2 * np.pi * day / 365)
            row["cos_day"] = np.cos(2 * np.pi * day / 365)
            prediction = model.predict(pd.DataFrame([row])[FEATURES])[0]

            st.metric(f"Predicted pollen for {target:%A, %B %d}", f"{prediction:,.0f} grains/m³")
            if prediction >= SPIKE:
                st.error("⚠️ Spike likely. If you take allergy medication, consider asking "
                         "your doctor about starting or stepping it up.")
            else:
                st.success("✅ No spike expected.")
            st.caption(f"Based on the latest count ({last_date:%b %d}: {last_count:,} grains/m³) "
                       "and the weather forecast.")
            if target.month not in [2, 3, 4, 5]:
                st.caption("It's outside tree pollen season, so spikes are unlikely right now. "
                           "Try the Replay tab to see the model on a spring day.")
    except Exception as e:
        st.error("Couldn't load live data right now. Please try again later.")
        st.caption(f"Details: {e}")

# ----- Tab 2: replay test-period days -----
with tab2:
    replay = pd.read_csv("replay.csv", parse_dates=["date"])
    st.write("Days the model **never saw during training**. Pick one to compare the "
             "prediction with what actually happened.")
    biggest = int(replay["total_pollen"].idxmax())
    pick = st.selectbox("Date", replay.index, index=biggest,
                        format_func=lambda i: replay.loc[i, "date"].strftime("%b %d, %Y"))
    actual, predicted = replay.loc[pick, "total_pollen"], replay.loc[pick, "predicted"]
    col1, col2 = st.columns(2)
    col1.metric("Predicted", f"{predicted:,.0f}")
    col2.metric("Actual", f"{actual:,.0f}")
    st.line_chart(replay.set_index("date")[["total_pollen", "predicted"]])

# ----- Tab 3: season calendar (YOUR TURN) -----
with tab3:
    calendar = pd.DataFrame({
        "Pollen": ["All trees", "Oak", "Pine", "Juniper", "Birch", "Grass", "Ragweed"],
        "Usually starts": ["Jan 16", "Mar 3", "Feb 23", "Jan 24", "Mar 5", "Mar 29", "Aug 21"],
        "Peaks": ["Apr 4", "Apr 4", "Apr 4", "Feb 25", "Apr 4", "May 10", "Sep 17"],
        "Ends": ["Jun 23", "May 18", "Jun 5", "Apr 4", "Apr 28", "Jun 18", "Nov 8"],
    })
    # TODO: show the table, a source line (Manangan et al., 2021, 1992–2018 averages),
    # and a note that seasons have been trending earlier.

st.divider()
st.caption("Pollen data: Atlanta Allergy & Asthma (NAB-certified station). "
           "Weather: Open-Meteo. General information only, not medical advice.")
