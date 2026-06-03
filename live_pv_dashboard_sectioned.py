import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(page_title="Live PV Health Monitoring Dashboard", layout="wide")

st.title("Live Solar PV Health Monitoring Dashboard")
st.caption(
    "Live PV monitoring dashboard for anomaly detection, Health Index, "
    "Damage Index, and maintenance-oriented RUL estimation"
)

# ==========================
# FILE PATHS
# ==========================
LIVE_JSON = Path("data/live_stream_data.json")
LIVE_CSV = Path("data/live_stream_data.csv")
HI_FILE = Path("data/health_index_results_live.csv")
DRIFT_FILE = Path("data/drift_results_live.csv")

VALIDATION_DATASETS = {
    "Normal": Path("data/health_index_results_Normal.csv"),
    "Mild": Path("data/health_index_results_Mild.csv"),
    "Moderate": Path("data/health_index_results_Moderate.csv"),
    "Severe": Path("data/health_index_results_Severe.csv"),
}

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚡ FYP Dashboard")

section = st.sidebar.radio(
    "Choose section",
    [
        "Overview",
        "Live Sensor Monitoring",
        "Health Index Analysis",
        "Drift & Anomaly Detection",
        "Maintenance Forecast",
        "Event Log",
        "Validation Results",
        "Data Table",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Dashboard Settings")
refresh_seconds = st.sidebar.slider(
    "Auto-refresh interval (seconds)", min_value=2, max_value=30, value=5, step=1
)

if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_seconds * 1000, key="live_refresh")
    st.sidebar.success("Auto-refresh enabled")
else:
    st.sidebar.warning("Auto-refresh package not installed. Run: pip install streamlit-autorefresh")

st.sidebar.write("Live data source:")
st.sidebar.code(str(LIVE_JSON))
st.sidebar.write("Health result source:")
st.sidebar.code(str(HI_FILE))

# ==========================
# LOADERS
# ==========================
def load_live_data():
    if LIVE_JSON.exists():
        try:
            df = pd.read_json(LIVE_JSON)
        except ValueError:
            st.warning("Live JSON is updating. Waiting for next refresh...")
            st.stop()
    elif LIVE_CSV.exists():
        df = pd.read_csv(LIVE_CSV)
    else:
        return None

    if len(df) == 0:
        return None

    df["rtc_time"] = pd.to_datetime(df["rtc_time"], errors="coerce")
    df = df.dropna(subset=["rtc_time"]).sort_values("rtc_time").reset_index(drop=True)

    for col in ["voltage", "current", "power", "pf", "degradation_factor", "day_counter"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_health_data():
    if not HI_FILE.exists() or not DRIFT_FILE.exists():
        return None, None
    try:
        hi_df = pd.read_csv(HI_FILE)
        drift_df = pd.read_csv(DRIFT_FILE)
    except Exception:
        return None, None
    if len(hi_df) == 0 or len(drift_df) == 0:
        return None, None

    hi_df["window_mid"] = pd.to_datetime(hi_df["window_mid"], errors="coerce")
    drift_df["window_mid"] = pd.to_datetime(drift_df["window_mid"], errors="coerce")
    hi_df = hi_df.dropna(subset=["window_mid"]).sort_values("window_mid").reset_index(drop=True)
    drift_df = drift_df.dropna(subset=["window_mid"]).sort_values("window_mid").reset_index(drop=True)
    return hi_df, drift_df


def get_hi_col(hi_df):
    if hi_df is None:
        return None
    if "HI_used" in hi_df.columns:
        return "HI_used"
    if "HI_cal" in hi_df.columns:
        return "HI_cal"
    if "HI_smooth" in hi_df.columns:
        return "HI_smooth"
    return None


def status_message(condition_state, maintenance_required):
    if maintenance_required:
        return "Maintenance inspection is recommended. Critical damage level has been reached.", "error"
    if condition_state == "Low Performance":
        return "Low performance detected. Continue monitoring for persistence.", "warning"
    if condition_state == "Moderate Condition":
        return "Moderate condition detected. System performance is degrading.", "warning"
    if condition_state == "Good Condition":
        return "System is operating within healthy monitoring range.", "success"
    return "System condition is currently unavailable.", "info"


def show_box(message, level):
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "success":
        st.success(message)
    else:
        st.info(message)

# ==========================
# REUSABLE SECTIONS
# ==========================
def show_live_sensor_cards(live_df):
    latest = live_df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Time", str(latest["rtc_time"]))
    c2.metric("Power", f"{latest['power']:.2f} W")
    c3.metric("Voltage", f"{latest['voltage']:.2f} V")
    c4.metric("Current", f"{latest['current']:.3f} A")
    if "day_counter" in live_df.columns:
        c5.metric("Monitoring Day", int(latest["day_counter"]))
    else:
        c5.metric("Total Samples", len(live_df))

    if "degradation_factor" in live_df.columns:
        st.progress(float(latest["degradation_factor"]))
        st.caption(
            f"Synthetic degradation factor: {latest['degradation_factor']:.4f} "
            "(lower value means higher simulated degradation)"
        )


def show_system_health_summary(hi_df, drift_df):
    if hi_df is None or drift_df is None or len(hi_df) == 0:
        st.info("Health results are not available yet. Run live_pipeline_processor.py first.")
        return

    latest = hi_df.iloc[-1]
    hi_col = get_hi_col(hi_df)
    h1, h2, h3, h4, h5 = st.columns(5)

    h1.metric("Health Index", f"{latest[hi_col]:.3f}" if hi_col else "N/A")

    condition = latest["condition_state"] if "condition_state" in hi_df.columns else "Unknown"
    h2.metric("Condition State", condition)

    h3.metric(
        "Drift Magnitude",
        f"{latest['drift_magnitude']:.3f}" if "drift_magnitude" in hi_df.columns else "N/A",
    )
    h4.metric(
        "Damage Index",
        f"{latest['damage_index']:.3f}" if "damage_index" in hi_df.columns else "N/A",
    )

    if "RUL_to_critical_days" in hi_df.columns and pd.notna(latest["RUL_to_critical_days"]):
        h5.metric("Days Before Inspection", f"{latest['RUL_to_critical_days']:.1f} days")
    else:
        h5.metric("Days Before Inspection", "Not Available")

    maintenance_required = bool(latest.get("maintenance_required", False))
    msg, level = status_message(condition, maintenance_required)
    show_box(msg, level)
    st.caption(
        "Critical condition means maintenance inspection is recommended. "
        "It does not mean the PV panel is physically broken."
    )


def show_health_index_chart(hi_df):
    hi_col = get_hi_col(hi_df)
    if hi_df is None or hi_col is None:
        st.info("Health Index result is not available yet.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hi_df["window_mid"], hi_df[hi_col], label="Health Index", linewidth=2)
    if "good_th" in hi_df.columns:
        ax.axhline(hi_df["good_th"].iloc[-1], linestyle="--", label="Good Threshold")
    if "low_th" in hi_df.columns:
        ax.axhline(hi_df["low_th"].iloc[-1], linestyle="--", label="Low Threshold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Health Index")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)

    if "condition_state" in hi_df.columns:
        st.subheader("Condition State Distribution")
        st.bar_chart(hi_df["condition_state"].value_counts())


def show_drift_chart(drift_df):
    if drift_df is None or "drift_magnitude" not in drift_df.columns:
        st.info("Drift result is not available yet.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(drift_df["window_mid"], drift_df["drift_magnitude"], label="Drift Magnitude", linewidth=2)
    if "warning_tau" in drift_df.columns and pd.notna(drift_df["warning_tau"].iloc[-1]):
        ax.axhline(drift_df["warning_tau"].iloc[-1], linestyle="--", label="Warning Threshold")
    if "critical_tau" in drift_df.columns and pd.notna(drift_df["critical_tau"].iloc[-1]):
        ax.axhline(drift_df["critical_tau"].iloc[-1], linestyle="--", label="Critical Threshold")
    if "persistent_alarm" in drift_df.columns:
        alarm_points = drift_df[drift_df["persistent_alarm"] == True]
        if len(alarm_points) > 0:
            ax.scatter(alarm_points["window_mid"], alarm_points["drift_magnitude"], label="Persistent Alarm")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drift Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)


def show_damage_chart(hi_df):
    if hi_df is None or "damage_index" not in hi_df.columns:
        st.info("Damage Index result is not available yet.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hi_df["window_mid"], hi_df["damage_index"], label="Damage Index", linewidth=2)
    if "damage_trend" in hi_df.columns:
        ax.plot(hi_df["window_mid"], hi_df["damage_trend"], linestyle="--", label="Damage Trend")
    ax.axhline(0.30, linestyle="--", label="Warning Damage")
    ax.axhline(0.60, linestyle="--", label="Critical Damage")
    ax.set_xlabel("Time")
    ax.set_ylabel("Damage Index")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)


def show_maintenance_forecast(hi_df):
    if hi_df is None or len(hi_df) == 0:
        st.info("Maintenance forecast is not available yet.")
        return

    latest = hi_df.iloc[-1]
    f1, f2, f3 = st.columns(3)
    if "rul_reliable" in hi_df.columns:
        f1.metric("Forecast Reliability", "Reliable" if bool(latest["rul_reliable"]) else "Not Reliable Yet")
    else:
        f1.metric("Forecast Reliability", "N/A")
    if "RUL_to_warning_days" in hi_df.columns and pd.notna(latest["RUL_to_warning_days"]):
        f2.metric("Days Before Warning Level", f"{latest['RUL_to_warning_days']:.1f} days")
    else:
        f2.metric("Days Before Warning Level", "N/A")
    if "RUL_to_critical_days" in hi_df.columns and pd.notna(latest["RUL_to_critical_days"]):
        f3.metric("Days Before Inspection", f"{latest['RUL_to_critical_days']:.1f} days")
    else:
        f3.metric("Days Before Inspection", "N/A")
    if "rul_reason" in hi_df.columns:
        st.caption(f"RUL reason/status: {latest['rul_reason']}")
    st.caption(
        "Maintenance forecast is based on persistent low-HI behaviour and accumulated damage index. "
        "It estimates inspection timing, not exact physical failure time."
    )


def build_event_log(hi_df):
    if hi_df is None or len(hi_df) == 0:
        return pd.DataFrame()
    events = []
    hi_col = get_hi_col(hi_df)
    if hi_col is None:
        return pd.DataFrame()

    if "condition_state" in hi_df.columns:
        for _, row in hi_df[hi_df["condition_state"] == "Moderate Condition"].tail(3).iterrows():
            events.append({"Time": row["window_mid"], "Event": "Moderate condition detected", "HI": round(row[hi_col], 3)})
        for _, row in hi_df[hi_df["condition_state"] == "Low Performance"].tail(3).iterrows():
            events.append({"Time": row["window_mid"], "Event": "Low performance detected", "HI": round(row[hi_col], 3)})
    if "persistent_alarm" in hi_df.columns:
        for _, row in hi_df[hi_df["persistent_alarm"] == True].tail(3).iterrows():
            events.append({"Time": row["window_mid"], "Event": "Persistent anomaly alarm", "HI": round(row[hi_col], 3)})
    if "maintenance_required" in hi_df.columns:
        for _, row in hi_df[hi_df["maintenance_required"] == True].tail(3).iterrows():
            events.append({"Time": row["window_mid"], "Event": "Maintenance inspection recommended", "HI": round(row[hi_col], 3)})
    if len(events) == 0:
        return pd.DataFrame()
    return pd.DataFrame(events).drop_duplicates().sort_values("Time").tail(10)


def show_sensor_trends(live_df):
    display_points = st.slider(
        "Number of latest points to display",
        min_value=50,
        max_value=max(50, min(2000, len(live_df))),
        value=min(300, len(live_df)),
        step=50,
    )
    plot_df = live_df.tail(display_points)

    fig_power, ax_power = plt.subplots(figsize=(12, 4))
    ax_power.plot(plot_df["rtc_time"], plot_df["power"], label="Power")
    ax_power.set_title("Live Power Trend")
    ax_power.set_xlabel("Time")
    ax_power.set_ylabel("Power (W)")
    ax_power.grid(True, alpha=0.3)
    ax_power.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig_power)

    sensor_col1, sensor_col2 = st.columns(2)
    with sensor_col1:
        fig_voltage, ax_voltage = plt.subplots(figsize=(8, 4))
        ax_voltage.plot(plot_df["rtc_time"], plot_df["voltage"], label="Voltage")
        ax_voltage.set_title("Live Voltage Trend")
        ax_voltage.set_xlabel("Time")
        ax_voltage.set_ylabel("Voltage (V)")
        ax_voltage.grid(True, alpha=0.3)
        ax_voltage.legend()
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig_voltage)
    with sensor_col2:
        fig_current, ax_current = plt.subplots(figsize=(8, 4))
        ax_current.plot(plot_df["rtc_time"], plot_df["current"], label="Current")
        ax_current.set_title("Live Current Trend")
        ax_current.set_xlabel("Time")
        ax_current.set_ylabel("Current (A)")
        ax_current.grid(True, alpha=0.3)
        ax_current.legend()
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig_current)

    if "degradation_factor" in live_df.columns:
        fig_deg, ax_deg = plt.subplots(figsize=(12, 4))
        ax_deg.plot(plot_df["rtc_time"], plot_df["degradation_factor"], label="Synthetic Degradation Factor")
        ax_deg.set_title("Synthetic Degradation Factor Over Time")
        ax_deg.set_xlabel("Time")
        ax_deg.set_ylabel("Degradation Factor")
        ax_deg.set_ylim(0, 1.05)
        ax_deg.grid(True, alpha=0.3)
        ax_deg.legend()
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig_deg)


def validation_summary_table():
    rows = []
    for name, path in VALIDATION_DATASETS.items():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if len(df) == 0:
            continue
        hi_col = "HI_used" if "HI_used" in df.columns else "HI_cal" if "HI_cal" in df.columns else None
        row = {"Dataset": name}
        if hi_col:
            row["Mean HI"] = round(df[hi_col].mean(), 3)
            row["Min HI"] = round(df[hi_col].min(), 3)
        if "drift_magnitude" in df.columns:
            row["Max Drift"] = round(df["drift_magnitude"].max(), 3)
        if "condition_state" in df.columns:
            row["Dominant Condition"] = df["condition_state"].mode().iloc[0]
        if "maintenance_required" in df.columns:
            row["Maintenance Count"] = int(df["maintenance_required"].sum())
        rows.append(row)
    return pd.DataFrame(rows)

# ==========================
# LOAD DATA
# ==========================
live_df = load_live_data()
if live_df is None:
    st.error("No live PV data found yet.")
    st.info("Run synthetic_streaming_engine.ipynb first until it creates live data files.")
    st.stop()
hi_df, drift_df = load_health_data()

# ==========================
# DASHBOARD SECTIONS
# ==========================
if section == "Overview":
    st.header("Overview")
    st.subheader("Live Sensor Reading")
    show_live_sensor_cards(live_df)
    st.divider()
    st.subheader("System Health Summary")
    show_system_health_summary(hi_df, drift_df)
    st.divider()
    st.subheader("Key Health Trend")
    show_health_index_chart(hi_df)

elif section == "Live Sensor Monitoring":
    st.header("Live Sensor Monitoring")
    show_live_sensor_cards(live_df)
    st.divider()
    show_sensor_trends(live_df)

elif section == "Health Index Analysis":
    st.header("Health Index Analysis")
    show_system_health_summary(hi_df, drift_df)
    st.divider()
    show_health_index_chart(hi_df)

elif section == "Drift & Anomaly Detection":
    st.header("Drift & Anomaly Detection")
    show_system_health_summary(hi_df, drift_df)
    st.divider()
    show_drift_chart(drift_df)
    st.caption("Drift magnitude measures deviation from healthy baseline behaviour.")

elif section == "Maintenance Forecast":
    st.header("Maintenance Forecast")
    show_system_health_summary(hi_df, drift_df)
    st.divider()
    show_damage_chart(hi_df)
    st.divider()
    show_maintenance_forecast(hi_df)

elif section == "Event Log":
    st.header("Monitoring Event Log")
    event_log = build_event_log(hi_df)
    if len(event_log) > 0:
        st.dataframe(event_log, use_container_width=True)
    else:
        st.info("No major monitoring events detected yet.")

elif section == "Validation Results":
    st.header("Validation Results")
    st.info(
        "This section compares Normal, Mild, Moderate, and Severe synthetic validation results. "
        "It is for evaluation only, not real-time operation."
    )
    validation_df = validation_summary_table()
    if len(validation_df) > 0:
        st.subheader("Synthetic Dataset Comparison")
        st.dataframe(validation_df, use_container_width=True)
        if "Mean HI" in validation_df.columns:
            st.bar_chart(validation_df.set_index("Dataset")["Mean HI"])
        if "Max Drift" in validation_df.columns:
            st.bar_chart(validation_df.set_index("Dataset")["Max Drift"])
    else:
        st.warning(
            "Validation CSV files are not available yet. Expected files: "
            "health_index_results_Normal.csv, health_index_results_Mild.csv, "
            "health_index_results_Moderate.csv, and health_index_results_Severe.csv."
        )

elif section == "Data Table":
    st.header("Data Table")
    st.subheader("Latest Live Stream Data")
    st.dataframe(live_df.tail(50), use_container_width=True)
    if hi_df is not None and len(hi_df) > 0:
        st.subheader("Latest Health Monitoring Windows")
        st.dataframe(hi_df.tail(20), use_container_width=True)
    else:
        st.info("Health monitoring table is not available yet.")

st.caption(
    "For real IAMMETER implementation, replace the synthetic streaming engine with an IAMMETER API collector "
    "while keeping the same health monitoring pipeline."
)
