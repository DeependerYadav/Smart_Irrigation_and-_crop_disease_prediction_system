"""Streamlit app: Smart Irrigation & Cotton Disease Prediction System."""

from datetime import datetime
import html
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st

from disease_predict import analyze_disease, get_cotton_disease_reference
from irrigation_predict import predict_irrigation


CROP_TYPES = {
    "Cotton": 4,
}
OPENWEATHER_API_KEY = "78a31fd2def4072831d935a2a61e28fe"
LANGUAGE_CODES = {
    "English": "en",
    "हिंदी": "hi",
}
SUPPORT_COPY = {
    "en": {
        "welcome": (
            "Hi! I am FarmBot Support. Ask me how to use Dashboard, Disease Scanner, "
            "Weather, or Analytics."
        ),
        "caption": "Live support and how-to guidance for this app.",
        "quick_help": "Quick Help",
        "quick_dashboard": "Dashboard",
        "quick_disease": "Disease",
        "quick_weather": "Weather",
        "quick_analytics": "Analytics",
        "query_dashboard": "How do I use Dashboard?",
        "query_disease": "How do I run disease scanner?",
        "query_weather": "How do I fetch weather and map?",
        "query_analytics": "How do I read analytics?",
        "chat_input": "Ask support about using this app...",
        "close_chat": "Close Chat",
        "clear_chat": "Clear Chat",
        "open_chat": "Open Support Chat",
        "language_label": "Language / भाषा",
        "cleared": "Chat cleared. Ask how to use Dashboard, Disease Scanner, Weather, or Analytics.",
    },
    "hi": {
        "welcome": (
            "नमस्ते! मैं FarmBot सहायता हूँ। आप मुझसे डैशबोर्ड, डिजीज स्कैनर, मौसम "
            "या एनालिटिक्स उपयोग करने के बारे में पूछ सकते हैं।"
        ),
        "caption": "ऐप उपयोग के लिए लाइव सहायता और मार्गदर्शन।",
        "quick_help": "त्वरित सहायता",
        "quick_dashboard": "डैशबोर्ड",
        "quick_disease": "रोग",
        "quick_weather": "मौसम",
        "quick_analytics": "एनालिटिक्स",
        "query_dashboard": "डैशबोर्ड कैसे उपयोग करें?",
        "query_disease": "डिजीज स्कैनर कैसे चलाएँ?",
        "query_weather": "मौसम और मैप कैसे लाएँ?",
        "query_analytics": "एनालिटिक्स कैसे पढ़ें?",
        "chat_input": "ऐप उपयोग से जुड़ा सवाल पूछें...",
        "close_chat": "चैट बंद करें",
        "clear_chat": "चैट साफ करें",
        "open_chat": "सहायता चैट खोलें",
        "language_label": "Language / भाषा",
        "cleared": "चैट साफ हो गई। डैशबोर्ड, डिजीज स्कैनर, मौसम या एनालिटिक्स के बारे में पूछें।",
    },
}


def support_copy(key: str, lang: str = "en") -> str:
    """Return localized text for support chatbot UI."""
    selected_lang = lang if lang in SUPPORT_COPY else "en"
    return SUPPORT_COPY.get(selected_lang, SUPPORT_COPY["en"]).get(key, SUPPORT_COPY["en"].get(key, key))


def calculate_efficiency_score(soil_moisture: float, predicted_water: float) -> float:
    """Simple heuristic efficiency score for UI display."""
    expected = max(1.0, (65 - soil_moisture) * 0.4)
    diff = abs(predicted_water - expected)
    score = max(0.0, 100.0 - (diff / expected) * 100.0)
    return round(score, 2)


def classify_sensor_conditions(soil_moisture: float, temperature: float, humidity: float) -> dict[str, str]:
    """Return readable sensor health labels."""
    if soil_moisture < 25:
        soil_status = "Very Dry"
    elif soil_moisture < 45:
        soil_status = "Moderate Moisture"
    else:
        soil_status = "High Moisture"

    if temperature > 36:
        temp_status = "High Heat"
    elif temperature < 20:
        temp_status = "Cool Condition"
    else:
        temp_status = "Normal Temperature"

    if humidity < 40:
        humidity_status = "Dry Air"
    elif humidity > 75:
        humidity_status = "Humid Air"
    else:
        humidity_status = "Balanced Humidity"

    return {
        "soil": soil_status,
        "temperature": temp_status,
        "humidity": humidity_status,
    }


def estimate_runtime_minutes(water_required: float, flow_rate_lpm: float) -> float:
    """Estimate pump runtime in minutes."""
    if flow_rate_lpm <= 0:
        return 0.0
    return round(water_required / flow_rate_lpm, 2)


def generate_weekly_projection(
    water_required: float,
    rain_forecast: int,
    irrigation_on: bool,
) -> pd.DataFrame:
    """Simulate 7-day irrigation demand projection."""
    days = np.arange(1, 8)
    baseline = water_required * (0.88 + 0.14 * np.sin(days / 1.4))

    if rain_forecast:
        baseline[1] *= 0.55
        baseline[2] *= 0.75

    if not irrigation_on:
        baseline *= 0.35

    baseline = np.clip(baseline, 0, None)
    return pd.DataFrame({"Day": days, "Projected Water (L)": baseline})


def generate_factor_breakdown(
    soil_moisture: float,
    temperature: float,
    humidity: float,
    rain_forecast: int,
) -> pd.DataFrame:
    """Create a simple feature-contribution style view for display."""
    factors = {
        "Soil Dryness": max(0.0, 70 - soil_moisture) * 0.8,
        "Temperature Stress": max(0.0, temperature - 28) * 0.7,
        "Low Humidity Impact": max(0.0, 55 - humidity) * 0.55,
        "Rain Relief": -12.0 if rain_forecast else 0.0,
    }
    return pd.DataFrame({"Factor": list(factors.keys()), "Impact Score": list(factors.values())})


def build_irrigation_schedule(
    water_required: float,
    rain_forecast: int,
    irrigation_on: bool,
) -> pd.DataFrame:
    """Create a practical same-day irrigation schedule."""
    if not irrigation_on:
        return pd.DataFrame(
            [
                {"Window": "Morning (6:00-8:00)", "Water (L)": 0.0},
                {"Window": "Evening (5:30-7:00)", "Water (L)": 0.0},
            ]
        )

    if rain_forecast:
        morning_share, evening_share = 0.45, 0.25
    else:
        morning_share, evening_share = 0.60, 0.40

    return pd.DataFrame(
        [
            {"Window": "Morning (6:00-8:00)", "Water (L)": round(water_required * morning_share, 2)},
            {"Window": "Evening (5:30-7:00)", "Water (L)": round(water_required * evening_share, 2)},
        ]
    )


def fetch_current_weather(city: str, api_key: str, country_code: str = "") -> dict[str, object]:
    """Fetch current weather from OpenWeatherMap and map it to app fields."""
    location = f"{city},{country_code}" if country_code else city
    encoded_location = quote_plus(location)
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={encoded_location}&appid={api_key}&units=metric"
    )

    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = "Weather API request failed."
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            message = error_payload.get("message", message)
        except Exception:
            pass
        raise ValueError(message) from exc
    except URLError as exc:
        raise ValueError("Network error while fetching weather data.") from exc

    weather_items = payload.get("weather", [])
    weather_main = weather_items[0].get("main", "").lower() if weather_items else ""
    weather_desc = weather_items[0].get("description", "N/A") if weather_items else "N/A"

    has_rain_signal = bool(payload.get("rain")) or bool(payload.get("snow"))
    rain_classes = {"rain", "drizzle", "thunderstorm", "snow"}
    rain_forecast = 1 if (has_rain_signal or weather_main in rain_classes) else 0

    main_data = payload.get("main", {})
    if "temp" not in main_data or "humidity" not in main_data:
        raise ValueError("Weather response missing temperature/humidity.")
    coord_data = payload.get("coord", {})
    latitude = coord_data.get("lat")
    longitude = coord_data.get("lon")

    return {
        "city_name": payload.get("name", city),
        "country": payload.get("sys", {}).get("country", country_code.upper() if country_code else ""),
        "condition": str(weather_desc).title(),
        "temperature": round(float(main_data["temp"]), 1),
        "feels_like": round(float(main_data.get("feels_like", main_data["temp"])), 1),
        "humidity": round(float(main_data["humidity"]), 1),
        "pressure": int(main_data.get("pressure", 0)),
        "wind_speed": round(float(payload.get("wind", {}).get("speed", 0.0)), 1),
        "rain_forecast": rain_forecast,
        "latitude": round(float(latitude), 6) if latitude is not None else None,
        "longitude": round(float(longitude), 6) if longitude is not None else None,
    }


def weather_icon(condition: str, rain_signal: int = 0) -> str:
    """Map weather text to a compact icon symbol."""
    text = condition.lower()
    if rain_signal or "rain" in text or "drizzle" in text:
        return "\U0001F327"
    if "thunder" in text or "storm" in text:
        return "\u26C8"
    if "snow" in text or "sleet" in text:
        return "\u2744"
    if "mist" in text or "fog" in text or "haze" in text:
        return "\U0001F32B"
    if "cloud" in text:
        return "\u2601"
    return "\u2600"


def build_hourly_weather_outlook(weather_snapshot: dict[str, object]) -> pd.DataFrame:
    """Create a realistic-looking short hourly outlook from current weather state."""
    labels = ["Now", "+2h", "+4h", "+6h", "+8h", "+10h"]
    base_temp = float(weather_snapshot["temperature"])
    base_humidity = float(weather_snapshot["humidity"])
    base_wind = float(weather_snapshot["wind_speed"])
    rain_signal = int(weather_snapshot["rain_forecast"])

    rows: list[dict[str, object]] = []
    for idx, label in enumerate(labels):
        temp = base_temp + 1.35 * np.sin((idx + 1) / 1.9) - (0.75 if rain_signal and idx >= 2 else 0.0)
        rain_prob = np.clip(
            14 + idx * 7 + (base_humidity - 52) * 0.32 + (17 if rain_signal else 0) + 9 * np.sin(idx / 2.0),
            4,
            96,
        )
        wind = np.clip(base_wind + 0.8 * np.cos(idx / 1.7), 0.2, None)

        if rain_prob >= 66:
            cond = "Rain"
        elif rain_prob >= 42:
            cond = "Cloudy"
        else:
            cond = "Clear"

        rows.append(
            {
                "Time": label,
                "Temperature": round(float(temp), 1),
                "RainChance": int(round(float(rain_prob))),
                "Wind": round(float(wind), 1),
                "Condition": cond,
                "Icon": weather_icon(cond, 1 if cond == "Rain" else 0),
            }
        )

    return pd.DataFrame(rows)


def initialize_session_defaults() -> None:
    """Initialize app state keys used across pages."""
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "en"
    if "ui_lang_label" not in st.session_state:
        st.session_state.ui_lang_label = "English"
    if "irrigation_history" not in st.session_state:
        st.session_state.irrigation_history = []
    if "last_irrigation_result" not in st.session_state:
        st.session_state.last_irrigation_result = None
    if "soil_moisture_input" not in st.session_state:
        st.session_state.soil_moisture_input = 35.0
    if "temperature_input" not in st.session_state:
        st.session_state.temperature_input = 28.0
    if "humidity_input" not in st.session_state:
        st.session_state.humidity_input = 55.0
    if "rain_forecast_input" not in st.session_state:
        st.session_state.rain_forecast_input = False
    if "weather_city" not in st.session_state:
        st.session_state.weather_city = "New York"
    if "weather_country" not in st.session_state:
        st.session_state.weather_country = "US"
    if "last_weather_result" not in st.session_state:
        st.session_state.last_weather_result = None
    if "last_disease_result" not in st.session_state:
        st.session_state.last_disease_result = None
    if "support_popup_open" not in st.session_state:
        st.session_state.support_popup_open = False
    if "support_chat_history" not in st.session_state:
        st.session_state.support_chat_history = [
            {
                "role": "assistant",
                "content": support_copy("welcome", st.session_state.ui_lang),
            }
        ]


def close_support_popup() -> None:
    """Close support chatbot popup."""
    st.session_state.support_popup_open = False


def support_chat_reply(user_message: str, active_page: str, lang: str = "en") -> str:
    """Generate localized rule-based support guidance for app usage."""
    message = user_message.strip().lower()
    selected_lang = lang if lang in {"en", "hi"} else "en"

    if selected_lang == "hi":
        active_page_hi = {
            "Dashboard": "डैशबोर्ड",
            "Disease Scanner": "डिजीज स्कैनर",
            "Weather": "मौसम",
            "Analytics": "एनालिटिक्स",
        }.get(active_page, active_page)

        if not message:
            return "आप ऐप उपयोग के बारे में कोई भी सवाल पूछ सकते हैं।"

        if any(token in message for token in ["hello", "hi", "hey", "नमस्ते", "हाय"]):
            return (
                "नमस्ते! आप सिंचाई प्रेडिक्शन, रोग पहचान, मौसम सिंक या एनालिटिक्स समझने के बारे में पूछ सकते हैं।"
            )

        if any(token in message for token in ["dashboard", "irrigation", "predict", "water", "डैशबोर्ड", "सिंचाई", "पानी", "इरिगेशन"]):
            return (
                "डैशबोर्ड: Soil Moisture, Temperature, Humidity, Crop Type और Rain Forecast भरें। "
                "`Predict Irrigation` दबाने पर पानी की मात्रा, ON/OFF निर्णय, efficiency, runtime और risk संकेत मिलते हैं।"
            )

        if any(token in message for token in ["disease", "scanner", "leaf", "image", "upload", "रोग", "बीमारी", "पत्ता", "इमेज", "अपलोड", "स्कैन"]):
            return (
                "डिजीज स्कैनर: कपास की पत्ती/पौधे की इमेज अपलोड करें, फिर `Run Disease Analysis` दबाएँ। "
                "इसके बाद disease नाम, confidence, affected area %, severity और सुझाव दिखेंगे।"
            )

        if any(token in message for token in ["weather", "city", "country", "forecast", "api", "मौसम", "शहर", "पूर्वानुमान", "मैप"]):
            return (
                "मौसम सेक्शन: City (और optional Country Code) डालकर `Fetch Current Weather` दबाएँ। "
                "इससे live weather, hourly outlook, forecast charts और location map अपडेट हो जाता है।"
            )

        if any(token in message for token in ["analytics", "trend", "graph", "anomaly", "एनालिटिक्स", "ट्रेंड", "ग्राफ", "विश्लेषण"]):
            return (
                "एनालिटिक्स: demand trend, rolling average, anomaly, distribution और short-term forecast देखें। "
                "यदि history नहीं है, तो ऐप sample data बनाकर visualization दिखाता है।"
            )

        if any(token in message for token in ["train", "model", "retrain", "प्रशिक्षण", "ट्रेन"]):
            return (
                "मॉडल ट्रेनिंग के लिए `python irrigation_training.py` और `python disease_training.py` चलाएँ, "
                "फिर Streamlit ऐप दोबारा शुरू करें।"
            )

        if any(token in message for token in ["error", "offline", "not working", "failed", "समस्या", "काम नहीं"]):
            return (
                "समस्या आने पर इंटरनेट/API कनेक्शन जाँचें, मॉडल फाइलें मौजूद हैं या नहीं देखें, "
                "और `pip install -r requirements.txt` से dependencies इंस्टॉल करें।"
            )

        return (
            f"मैं अभी `{active_page_hi}` से जुड़ी सहायता के लिए तैयार हूँ। "
            "आप सिंचाई, रोग पहचान, मौसम/मैप, एनालिटिक्स, ट्रेनिंग या troubleshooting के बारे में पूछ सकते हैं।"
        )

    if not message:
        return "Ask me anything about using this app."

    if any(token in message for token in ["hello", "hi", "hey"]):
        return (
            "Hello! You can ask me about irrigation prediction, disease scan, weather sync, "
            "or analytics interpretation."
        )

    if any(token in message for token in ["dashboard", "irrigation", "predict", "water"]):
        return (
            "Dashboard: Enter soil moisture, temperature, humidity, crop type, and rain signal. "
            "Click `Predict Irrigation` to get water required, irrigation ON/OFF, efficiency, "
            "runtime, and risk indicators."
        )

    if any(token in message for token in ["disease", "scanner", "leaf", "image", "upload"]):
        return (
            "Disease Scanner: Upload a cotton leaf/plant image, click `Run Disease Analysis`, "
            "then review disease name, confidence, affected area %, severity, and recommended actions."
        )

    if any(token in message for token in ["weather", "city", "country", "forecast", "api"]):
        return (
            "Weather: Enter city (and optional country code), then click `Fetch Current Weather`. "
            "This updates live conditions, hourly outlook, forecast charts, and the location map."
        )

    if any(token in message for token in ["analytics", "trend", "graph", "anomaly"]):
        return (
            "Analytics: View demand trends, rolling averages, anomalies, distribution, and short-term "
            "forecast. If no history exists, the app generates sample data automatically."
        )

    if any(token in message for token in ["train", "model", "retrain"]):
        return (
            "Training: run `python irrigation_training.py` for irrigation model and "
            "`python disease_training.py` for disease model, then restart Streamlit."
        )

    if any(token in message for token in ["error", "offline", "not working", "failed"]):
        return (
            "Troubleshooting: check internet/API access, verify model files exist, and confirm "
            "dependencies are installed with `pip install -r requirements.txt`."
        )

    return (
        f"I am currently focused on `{active_page}` guidance. Ask about irrigation, disease scan, "
        "weather fetch/map, analytics, training, or troubleshooting."
    )


@st.dialog("FarmBot Support", width="large", on_dismiss=close_support_popup)
def support_chatbot_dialog(active_page: str, lang: str) -> None:
    """Popup support chatbot shown on app open and on-demand."""
    selected_lang = lang if lang in {"en", "hi"} else "en"
    st.caption(support_copy("caption", selected_lang))
    st.markdown(f"#### {support_copy('quick_help', selected_lang)}")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button(
            support_copy("quick_dashboard", selected_lang),
            use_container_width=True,
            key=f"support_quick_dashboard_{active_page}",
        ):
            st.session_state.support_chat_history.append(
                {"role": "user", "content": support_copy("query_dashboard", selected_lang)}
            )
            st.session_state.support_chat_history.append(
                {
                    "role": "assistant",
                    "content": support_chat_reply("dashboard irrigation predict", active_page, selected_lang),
                }
            )
            st.rerun()
    with q2:
        if st.button(
            support_copy("quick_disease", selected_lang),
            use_container_width=True,
            key=f"support_quick_disease_{active_page}",
        ):
            st.session_state.support_chat_history.append(
                {"role": "user", "content": support_copy("query_disease", selected_lang)}
            )
            st.session_state.support_chat_history.append(
                {
                    "role": "assistant",
                    "content": support_chat_reply("disease scanner upload image", active_page, selected_lang),
                }
            )
            st.rerun()
    with q3:
        if st.button(
            support_copy("quick_weather", selected_lang),
            use_container_width=True,
            key=f"support_quick_weather_{active_page}",
        ):
            st.session_state.support_chat_history.append(
                {"role": "user", "content": support_copy("query_weather", selected_lang)}
            )
            st.session_state.support_chat_history.append(
                {
                    "role": "assistant",
                    "content": support_chat_reply("weather city forecast map", active_page, selected_lang),
                }
            )
            st.rerun()
    with q4:
        if st.button(
            support_copy("quick_analytics", selected_lang),
            use_container_width=True,
            key=f"support_quick_analytics_{active_page}",
        ):
            st.session_state.support_chat_history.append(
                {"role": "user", "content": support_copy("query_analytics", selected_lang)}
            )
            st.session_state.support_chat_history.append(
                {
                    "role": "assistant",
                    "content": support_chat_reply("analytics trend anomaly graph", active_page, selected_lang),
                }
            )
            st.rerun()

    for msg in st.session_state.support_chat_history[-14:]:
        with st.chat_message(msg["role"]):
            st.markdown(str(msg["content"]))

    prompt = st.chat_input(support_copy("chat_input", selected_lang))
    if prompt:
        st.session_state.support_chat_history.append({"role": "user", "content": prompt})
        st.session_state.support_chat_history.append(
            {"role": "assistant", "content": support_chat_reply(prompt, active_page, selected_lang)}
        )
        st.rerun()

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(
            support_copy("close_chat", selected_lang),
            use_container_width=True,
            key=f"support_close_btn_{active_page}",
        ):
            close_support_popup()
            st.rerun()
    with c2:
        if st.button(
            support_copy("clear_chat", selected_lang),
            use_container_width=True,
            key=f"support_clear_btn_{active_page}",
        ):
            st.session_state.support_chat_history = [
                {
                    "role": "assistant",
                    "content": support_copy("cleared", selected_lang),
                }
            ]
            st.rerun()


def inject_custom_css() -> None:
    """Apply a dark neon dashboard theme with strong contrast."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Fraunces:opsz,wght@9..144,600&display=swap');

        :root {
            --bg-a: #f8f2e8;
            --bg-b: #efe4d0;
            --bg-c: #f3eadb;
            --glass: rgba(255, 255, 255, 0.88);
            --glass-soft: rgba(255, 255, 255, 0.94);
            --stroke: rgba(166, 124, 82, 0.28);
            --text: #263238;
            --muted: #5b645e;
            --accent: #00bcd4;
            --accent-2: #0097a7;
            --accent-3: #4caf50;
            --accent-4: #2e7d32;
            --metric-label: #43514a;
            --status-bg: rgba(247, 242, 233, 0.98);
            --status-stroke: rgba(166, 124, 82, 0.34);
            --glow-a: rgba(0, 188, 212, 0.18);
            --glow-b: rgba(76, 175, 80, 0.16);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(0, 188, 212, 0.16) 0%, rgba(0, 0, 0, 0) 36%),
                radial-gradient(circle at 88% 14%, rgba(76, 175, 80, 0.14) 0%, rgba(0, 0, 0, 0) 34%),
                radial-gradient(circle at 84% 90%, rgba(215, 204, 200, 0.44) 0%, rgba(0, 0, 0, 0) 36%),
                radial-gradient(circle at 14% 84%, rgba(166, 124, 82, 0.16) 0%, rgba(0, 0, 0, 0) 34%),
                linear-gradient(138deg, var(--bg-a) 0%, var(--bg-c) 48%, var(--bg-b) 100%);
            color: var(--text);
            font-family: 'Manrope', sans-serif;
        }

        @keyframes fade-up {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes soft-pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(88, 183, 243, 0.0);
            }
            50% {
                box-shadow: 0 0 0 8px rgba(88, 183, 243, 0.06);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(88, 183, 243, 0.0);
            }
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .main .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            animation: fade-up 0.42s ease;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(249, 243, 232, 0.98) 0%, rgba(241, 232, 216, 0.98) 100%);
            backdrop-filter: blur(10px);
            border-right: 1px solid var(--stroke);
        }

        section[data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        .side-intro {
            border: 1px solid rgba(166, 124, 82, 0.40);
            border-radius: 14px;
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(245, 237, 222, 0.92));
            padding: 12px 12px 10px 12px;
            margin-bottom: 0.8rem;
            box-shadow: inset 0 0 0 1px rgba(166, 124, 82, 0.08);
        }

        .side-intro .title {
            font-size: 1rem;
            font-weight: 800;
            color: #2e7d32;
            margin-bottom: 4px;
        }

        .side-intro .sub {
            font-size: 0.78rem;
            color: #5d6962;
            line-height: 1.35;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] > label p {
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            color: #374740 !important;
        }

        section[data-testid="stSidebar"] label[data-baseweb="radio"],
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
            margin-bottom: 0.58rem !important;
            padding: 0.82rem 0.92rem !important;
            border: 1px solid rgba(166, 124, 82, 0.32) !important;
            border-radius: 0.82rem !important;
            background: rgba(255, 255, 255, 0.90) !important;
            transition: all 0.18s ease !important;
            cursor: pointer !important;
        }

        section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover,
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            border-color: rgba(0, 188, 212, 0.58) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 18px rgba(124, 106, 82, 0.16) !important;
        }

        section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        section[data-testid="stSidebar"] label[data-baseweb="radio"] p,
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
            font-size: 1.08rem !important;
            line-height: 1.25 !important;
            font-weight: 700 !important;
            margin: 0 !important;
            color: #2f3b36 !important;
        }

        section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked),
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            border-color: rgba(0, 151, 167, 0.72) !important;
            background: linear-gradient(130deg, rgba(226, 247, 244, 0.98) 0%, rgba(217, 245, 237, 0.98) 52%, rgba(237, 249, 241, 0.98) 100%) !important;
            box-shadow: 0 0 0 1px rgba(0, 151, 167, 0.16), 0 8px 18px rgba(124, 106, 82, 0.14) !important;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            color: var(--text);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--glass);
            border: 1px solid var(--stroke) !important;
            border-radius: 20px !important;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(10px);
            padding: 0.4rem 0.75rem 0.8rem 0.75rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
            border-color: rgba(150, 214, 246, 0.58) !important;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.34), 0 0 0 1px rgba(143, 205, 240, 0.08);
        }

        div[data-testid="stMetric"] {
            background: var(--glass-soft);
            border: 1px solid var(--stroke);
            border-radius: 14px;
            padding: 10px 12px;
            min-height: 108px;
            transition: transform 0.18s ease, border-color 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(143, 210, 244, 0.62);
        }

        div[data-testid="stMetricLabel"],
        label[data-testid="stMetricLabel"] {
            color: var(--metric-label) !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            font-size: 0.82rem !important;
            line-height: 1.25 !important;
            font-weight: 700 !important;
            text-shadow: 0 0 1px rgba(6, 20, 34, 0.38);
        }

        div[data-testid="stMetricLabel"] p,
        label[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] div,
        label[data-testid="stMetricLabel"] div,
        div[data-testid="stMetric"] label {
            color: var(--metric-label) !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text) !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            font-size: 1.6rem !important;
            line-height: 1.2 !important;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(19, 44, 67, 0.95) !important;
            border: 1px dashed rgba(136, 199, 233, 0.68) !important;
            border-radius: 12px !important;
        }

        div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzoneInstructions"] p,
        div[data-testid="stFileUploaderDropzoneInstructions"] small {
            color: #e6f4ff !important;
            opacity: 1 !important;
        }

        div[data-testid="stFileUploaderDropzone"] svg {
            fill: #c6e3f8 !important;
            color: #c6e3f8 !important;
        }

        div[data-testid="stFileUploaderDropzone"] button {
            background: rgba(43, 99, 142, 0.96) !important;
            color: #eef9ff !important;
            border: 1px solid rgba(145, 209, 242, 0.62) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFileUploaderDropzone"] button *,
        div[data-testid="stFileUploaderDropzone"] [role="button"] *,
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"] * {
            color: #eef9ff !important;
            fill: #eef9ff !important;
            -webkit-text-fill-color: #eef9ff !important;
            opacity: 1 !important;
        }

        div[data-testid="stFileUploaderDropzone"] button:hover {
            background: rgba(58, 122, 172, 0.98) !important;
            color: #ffffff !important;
            border-color: rgba(170, 222, 249, 0.74) !important;
        }

        div[data-testid="stFileUploaderDropzone"] button:disabled {
            color: #deeffc !important;
            -webkit-text-fill-color: #deeffc !important;
            opacity: 1 !important;
        }

        div[data-testid="stFileUploaderDropzone"] button:disabled *,
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"][disabled] * {
            color: #deeffc !important;
            fill: #deeffc !important;
            -webkit-text-fill-color: #deeffc !important;
            opacity: 1 !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(36, 112, 158, 0.3);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        }

        .stButton > button:hover {
            color: white;
            background: linear-gradient(135deg, #3a9ed2 0%, #246e9e 100%);
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(36, 112, 158, 0.42);
            filter: saturate(1.08);
        }

        .stNumberInput input, .stTextInput input {
            background: rgba(19, 43, 66, 0.95) !important;
            color: var(--text) !important;
            border: 1px solid var(--stroke) !important;
            border-radius: 10px !important;
        }

        .stNumberInput input::placeholder, .stTextInput input::placeholder {
            color: #9ec8df !important;
            opacity: 1 !important;
        }

        .stNumberInput input:disabled, .stTextInput input:disabled {
            color: #d8ecf9 !important;
            -webkit-text-fill-color: #d8ecf9 !important;
            opacity: 1 !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: #d9edfb !important;
            font-weight: 600 !important;
        }

        [data-baseweb="select"] > div {
            background: rgba(19, 43, 66, 0.95);
            border: 1px solid var(--stroke);
            border-radius: 10px;
        }

        [data-baseweb="select"] * {
            color: var(--text) !important;
        }

        [data-baseweb="select"] svg {
            fill: #bde1f5 !important;
        }

        [role="listbox"] {
            background: rgba(15, 36, 56, 0.98) !important;
            border: 1px solid rgba(125, 188, 224, 0.45) !important;
        }

        [role="option"] {
            color: #e8f6ff !important;
            background: rgba(15, 36, 56, 0.98) !important;
        }

        [role="option"]:hover {
            background: rgba(41, 82, 118, 0.88) !important;
        }

        [role="option"][aria-selected="true"] {
            background: rgba(56, 101, 143, 0.92) !important;
        }

        div[data-testid="stExpander"] > details {
            background: rgba(16, 40, 61, 0.92) !important;
            border: 1px solid rgba(119, 184, 220, 0.45) !important;
            border-radius: 12px !important;
        }

        div[data-testid="stExpander"] > details > summary {
            color: #e5f4ff !important;
            font-weight: 700 !important;
        }

        div[data-testid="stExpander"] > details > summary svg {
            fill: #b7ddf3 !important;
        }

        [data-testid="stProgress"] > div > div {
            background: rgba(24, 58, 86, 0.72) !important;
            border-radius: 999px !important;
            border: 1px solid rgba(128, 190, 225, 0.34);
        }

        [data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #46c9f0 0%, #8c78ff 48%, #40d8b6 100%) !important;
            border-radius: 999px !important;
        }

        [data-testid="stDataFrame"] {
            border-radius: 14px !important;
            border: 1px solid rgba(126, 188, 224, 0.38) !important;
            overflow: hidden;
            box-shadow: inset 0 0 0 1px rgba(136, 200, 236, 0.08);
        }

        .top-shell {
            background: linear-gradient(140deg, rgba(14, 33, 52, 0.90) 0%, rgba(17, 44, 66, 0.90) 100%);
            border: 1px solid var(--stroke);
            border-radius: 22px;
            padding: 11px 16px 14px 16px;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.36);
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
            animation: fade-up 0.44s ease;
        }

        .top-shell::after {
            content: "";
            position: absolute;
            right: -70px;
            top: -70px;
            width: 190px;
            height: 190px;
            border-radius: 999px;
            background: radial-gradient(circle, var(--glow-a) 0%, rgba(0, 0, 0, 0) 68%);
            pointer-events: none;
        }

        .chrome-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            border-bottom: 1px solid rgba(92, 130, 102, 0.22);
            padding-bottom: 8px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }

        .chrome-dots {
            display: flex;
            gap: 8px;
        }

        .chrome-dots span {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: #87a48c;
            display: inline-block;
        }

        .chrome-dots span:nth-child(1) { background: #e1837a; }
        .chrome-dots span:nth-child(2) { background: #c8cb84; }
        .chrome-dots span:nth-child(3) { background: #7ab67f; }

        .mini-tabs {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .mini-tab {
            font-size: 0.82rem;
            color: var(--muted);
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid transparent;
        }

        .mini-tab.active {
            color: #eaf6ed;
            background: linear-gradient(135deg, rgba(75, 147, 212, 0.28) 0%, rgba(132, 101, 237, 0.26) 100%);
            border-color: rgba(116, 176, 214, 0.52);
            font-weight: 700;
        }

        .alert-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .alert-box {
            background: rgba(19, 45, 69, 0.96);
            border: 1px solid rgba(120, 180, 214, 0.42);
            border-radius: 14px;
            padding: 8px 12px;
            min-width: 270px;
        }

        .alert-box strong {
            display: block;
            color: #d9edde;
        }

        .alert-box span {
            font-size: 0.85rem;
            color: #b6d7ea;
        }

        .section-label {
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #93c8e6;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .section-label::after {
            content: "";
            width: 22px;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg, #46c8ef 0%, #8f7cff 100%);
        }

        .small-muted {
            color: #a9cfe4;
            font-size: 0.84rem;
        }

        .status-card {
            background: var(--status-bg);
            border: 1px solid var(--status-stroke);
            border-radius: 12px;
            padding: 10px 12px;
            min-height: 92px;
        }

        .status-card .label {
            color: #9ccce5;
            font-size: 0.82rem;
            margin-bottom: 4px;
        }

        .status-card .value {
            color: #e9f6ff;
            font-size: 1.28rem;
            font-weight: 700;
            white-space: normal;
            line-height: 1.25;
        }

        .hud-banner {
            background: linear-gradient(130deg, rgba(29, 72, 108, 0.88) 0%, rgba(30, 63, 120, 0.86) 60%, rgba(42, 93, 128, 0.86) 100%);
            border: 1px solid rgba(121, 198, 244, 0.52);
            border-radius: 12px;
            padding: 9px 12px;
            font-weight: 700;
            color: #cbeaff;
            margin-bottom: 0.7rem;
            box-shadow: inset 0 0 0 1px rgba(166, 225, 255, 0.08), 0 4px 14px rgba(53, 120, 170, 0.24);
            animation: soft-pulse 2.8s ease-in-out infinite;
        }

        .hud-banner.warn {
            border-color: rgba(238, 185, 95, 0.70);
            background: rgba(82, 57, 24, 0.78);
            color: #ffd88f;
        }

        .hud-actions {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .hud-dot {
            width: 30px;
            height: 30px;
            border-radius: 999px;
            border: 1px solid rgba(120, 190, 226, 0.45);
            background: rgba(20, 50, 73, 0.88);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #bfe7ff;
            font-size: 0.9rem;
        }

        .hud-dot:nth-child(1) {
            border-color: rgba(122, 200, 247, 0.62);
            box-shadow: 0 0 14px rgba(81, 190, 255, 0.32);
        }

        .hud-dot:nth-child(2) {
            border-color: rgba(180, 153, 255, 0.62);
            box-shadow: 0 0 14px rgba(153, 121, 255, 0.30);
        }

        .risk-card {
            border-radius: 14px;
            padding: 12px 14px;
            border: 1px solid rgba(121, 173, 202, 0.34);
            background: linear-gradient(145deg, rgba(20, 49, 75, 0.82), rgba(24, 57, 88, 0.76));
            min-height: 118px;
        }

        .risk-card h4 {
            margin: 0 0 6px 0;
            font-family: 'Manrope', sans-serif;
            font-size: 0.95rem;
            color: #a8d1ea;
        }

        .risk-card .risk-value {
            font-size: 1.5rem;
            font-weight: 800;
            margin: 0;
            color: #e9f6ff;
        }

        .risk-card .risk-sub {
            margin-top: 6px;
            font-size: 0.84rem;
            color: #b8d9ec;
        }

        .risk-card.safe {
            border-color: rgba(117, 228, 178, 0.52);
            background: linear-gradient(145deg, rgba(20, 90, 86, 0.75), rgba(18, 68, 77, 0.70));
        }

        .risk-card.warn {
            border-color: rgba(255, 195, 112, 0.60);
            background: linear-gradient(145deg, rgba(122, 82, 34, 0.70), rgba(95, 66, 35, 0.65));
        }

        .risk-card.danger {
            border-color: rgba(255, 130, 130, 0.60);
            background: linear-gradient(145deg, rgba(121, 45, 54, 0.74), rgba(98, 36, 46, 0.72));
        }

        .kpi-tile {
            border-radius: 16px;
            border: 1px solid rgba(121, 173, 202, 0.42);
            background: linear-gradient(145deg, rgba(17, 44, 68, 0.90), rgba(20, 54, 81, 0.86));
            padding: 14px 16px;
            min-height: 112px;
            box-shadow: inset 0 0 0 1px rgba(146, 202, 233, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .kpi-tile:hover {
            transform: translateY(-2px);
            border-color: rgba(152, 216, 247, 0.68);
            box-shadow: inset 0 0 0 1px rgba(166, 220, 247, 0.12), 0 14px 26px rgba(0, 0, 0, 0.24);
        }

        .kpi-tile .kpi-label {
            color: #a7cfe8;
            font-size: 0.86rem;
            margin-bottom: 8px;
            letter-spacing: 0.03em;
        }

        .kpi-tile .kpi-value {
            color: #eaf6ff;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .kpi-tile .kpi-sub {
            margin-top: 6px;
            color: #b8daee;
            font-size: 0.83rem;
        }

        .kpi-tile.aqua {
            border-color: rgba(114, 212, 246, 0.58);
            background: linear-gradient(145deg, rgba(25, 75, 112, 0.90), rgba(21, 63, 95, 0.88));
        }

        .kpi-tile.violet {
            border-color: rgba(168, 142, 255, 0.62);
            background: linear-gradient(145deg, rgba(63, 56, 128, 0.88), rgba(49, 51, 104, 0.86));
        }

        .kpi-tile.mint {
            border-color: rgba(108, 228, 188, 0.62);
            background: linear-gradient(145deg, rgba(20, 86, 90, 0.88), rgba(18, 69, 85, 0.86));
        }

        .disease-panel {
            border: 1px solid rgba(119, 186, 223, 0.40);
            border-radius: 14px;
            background:
                radial-gradient(circle at 50% 26%, rgba(101, 231, 170, 0.20), rgba(0, 0, 0, 0) 56%),
                radial-gradient(circle at center, rgba(68, 144, 220, 0.20), rgba(17, 40, 61, 0.88));
            padding: 14px;
        }

        .scan-result {
            border-radius: 12px;
            border: 1px solid rgba(130, 193, 229, 0.46);
            background: rgba(26, 58, 84, 0.90);
            color: #eaf6ff;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 12px 14px;
            margin-bottom: 10px;
        }

        .scan-result.ok {
            border-color: rgba(111, 232, 186, 0.62);
            background: linear-gradient(135deg, rgba(21, 92, 83, 0.90), rgba(17, 75, 85, 0.88));
            color: #d9fff0;
        }

        .scan-result.warn {
            border-color: rgba(255, 194, 97, 0.70);
            background: linear-gradient(135deg, rgba(110, 76, 31, 0.90), rgba(85, 60, 32, 0.88));
            color: #ffe4a8;
        }

        .confidence-card {
            border-radius: 12px;
            border: 1px solid rgba(128, 194, 233, 0.52);
            background: rgba(19, 51, 77, 0.92);
            padding: 12px 14px;
            margin-bottom: 8px;
        }

        .confidence-card .label {
            color: #cfe9fb;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .confidence-card .value {
            color: #f2f9ff;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .scan-severity {
            border-radius: 12px;
            border: 1px solid rgba(133, 199, 235, 0.48);
            padding: 10px 12px;
            margin: 8px 0 10px 0;
            font-weight: 800;
            color: #e9f6ff;
            background: rgba(20, 53, 78, 0.9);
        }

        .scan-severity.healthy {
            border-color: rgba(111, 232, 186, 0.62);
            background: linear-gradient(135deg, rgba(21, 92, 83, 0.9), rgba(17, 75, 85, 0.88));
            color: #d9fff0;
        }

        .scan-severity.low {
            border-color: rgba(158, 235, 184, 0.62);
            background: linear-gradient(135deg, rgba(33, 99, 73, 0.9), rgba(24, 82, 69, 0.88));
            color: #dcffe9;
        }

        .scan-severity.moderate {
            border-color: rgba(255, 202, 127, 0.7);
            background: linear-gradient(135deg, rgba(120, 88, 36, 0.9), rgba(95, 69, 35, 0.88));
            color: #ffe7b8;
        }

        .scan-severity.high {
            border-color: rgba(255, 148, 148, 0.72);
            background: linear-gradient(135deg, rgba(120, 52, 58, 0.9), rgba(96, 42, 50, 0.88));
            color: #ffd4d4;
        }

        .suggestion-card {
            border-radius: 12px;
            border: 1px solid rgba(128, 194, 233, 0.52);
            background: rgba(19, 51, 77, 0.92);
            padding: 12px 14px 10px 14px;
            margin-top: 8px;
        }

        .suggestion-card .label {
            color: #d6edfc;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .suggestion-card ul {
            margin: 0;
            padding-left: 18px;
        }

        .suggestion-card li {
            color: #e9f6ff;
            font-size: 0.88rem;
            margin: 0 0 6px 0;
            line-height: 1.3;
        }

        .weather-hero {
            border: 1px solid rgba(136, 198, 233, 0.52);
            border-radius: 18px;
            background:
                radial-gradient(circle at 78% 18%, rgba(158, 131, 255, 0.20), rgba(0, 0, 0, 0) 48%),
                radial-gradient(circle at 20% 88%, rgba(70, 215, 179, 0.18), rgba(0, 0, 0, 0) 48%),
                linear-gradient(140deg, rgba(23, 60, 89, 0.92), rgba(20, 52, 77, 0.90));
            padding: 16px 18px;
            box-shadow: inset 0 0 0 1px rgba(177, 226, 252, 0.08);
            margin-bottom: 0.9rem;
        }

        .weather-hero-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
        }

        .weather-city {
            font-size: 1.45rem;
            font-weight: 800;
            color: #eef9ff;
            line-height: 1.2;
        }

        .weather-meta {
            color: #b6daf0;
            font-size: 0.86rem;
            margin-top: 2px;
        }

        .weather-icon {
            font-size: 2rem;
            line-height: 1;
            filter: drop-shadow(0 0 12px rgba(118, 205, 255, 0.38));
        }

        .weather-temp-row {
            display: flex;
            align-items: baseline;
            gap: 12px;
            margin-top: 10px;
            flex-wrap: wrap;
        }

        .weather-temp {
            font-size: 3rem;
            font-weight: 900;
            line-height: 1;
            color: #f2fbff;
        }

        .weather-condition {
            font-size: 1rem;
            color: #d4ebfa;
            font-weight: 700;
        }

        .weather-chip-row {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            flex-wrap: wrap;
        }

        .weather-chip {
            border: 1px solid rgba(137, 197, 232, 0.45);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.8rem;
            color: #d9effd;
            background: rgba(23, 58, 84, 0.82);
        }

        .weather-detail-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }

        .weather-detail-card {
            border: 1px solid rgba(134, 196, 232, 0.44);
            border-radius: 12px;
            background: rgba(21, 52, 78, 0.88);
            padding: 10px 12px;
        }

        .weather-detail-card .label {
            color: #b9def2;
            font-size: 0.78rem;
            margin-bottom: 2px;
        }

        .weather-detail-card .value {
            color: #eef9ff;
            font-size: 1.15rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .weather-forecast-grid {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding-bottom: 4px;
            scroll-behavior: smooth;
        }

        .weather-forecast-grid::-webkit-scrollbar {
            height: 8px;
        }

        .weather-forecast-grid::-webkit-scrollbar-thumb {
            background: rgba(146, 204, 236, 0.45);
            border-radius: 999px;
        }

        .weather-forecast-card {
            border: 1px solid rgba(123, 187, 225, 0.38);
            border-radius: 12px;
            background: rgba(20, 51, 76, 0.86);
            padding: 8px 10px;
            text-align: center;
            min-width: 108px;
            flex: 0 0 auto;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .weather-forecast-card:hover {
            transform: translateY(-2px);
            border-color: rgba(151, 214, 246, 0.66);
        }

        .command-pill-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        .command-pill {
            border-radius: 999px;
            border: 1px solid rgba(130, 196, 233, 0.42);
            background: rgba(20, 51, 76, 0.86);
            padding: 5px 11px;
            font-size: 0.78rem;
            color: #d7eefc;
            font-weight: 600;
        }

        .command-pill strong {
            color: #eff9ff;
            font-weight: 800;
            margin-right: 5px;
        }

        .command-pill.good {
            border-color: rgba(106, 226, 183, 0.58);
            background: rgba(24, 85, 88, 0.84);
            color: #d8fff0;
        }

        .command-pill.warn {
            border-color: rgba(247, 189, 102, 0.66);
            background: rgba(93, 65, 30, 0.84);
            color: #ffe5b3;
        }

        .command-pill.alert {
            border-color: rgba(255, 143, 143, 0.68);
            background: rgba(96, 42, 50, 0.86);
            color: #ffd9d9;
        }

        .weather-forecast-time {
            color: #b9dff4;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .weather-forecast-icon {
            font-size: 1.2rem;
            margin-bottom: 2px;
        }

        .weather-forecast-temp {
            color: #edf8ff;
            font-weight: 800;
            font-size: 1rem;
            line-height: 1.1;
        }

        .weather-forecast-sub {
            color: #aacee4;
            font-size: 0.72rem;
            margin-top: 2px;
        }

        /* Earthy light palette overrides */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(166, 124, 82, 0.30) !important;
            box-shadow: 0 14px 28px rgba(116, 96, 74, 0.14);
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(0, 151, 167, 0.42) !important;
            box-shadow: 0 16px 30px rgba(116, 96, 74, 0.17);
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(166, 124, 82, 0.24);
        }

        div[data-testid="stMetricValue"] {
            color: #213029 !important;
        }

        div[data-testid="stMetricDelta"] {
            color: #2e7d32 !important;
        }

        .top-shell {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.93), rgba(245, 237, 222, 0.93));
            border: 1px solid rgba(166, 124, 82, 0.32);
            box-shadow: 0 16px 30px rgba(116, 96, 74, 0.16);
            padding: 14px 20px 18px 20px;
            border-radius: 24px;
        }

        .top-shell::after {
            background: radial-gradient(circle, var(--glow-b) 0%, rgba(0, 0, 0, 0) 68%);
        }

        .chrome-row {
            border-bottom: 1px solid rgba(166, 124, 82, 0.22);
        }

        .mini-tab {
            color: #54645e;
            background: rgba(255, 255, 255, 0.64);
            border-color: rgba(166, 124, 82, 0.20);
            font-size: 0.9rem;
            padding: 6px 14px;
            font-weight: 600;
        }

        .mini-tab.active {
            color: #1f3a32;
            background: linear-gradient(135deg, rgba(196, 241, 247, 0.76) 0%, rgba(212, 243, 223, 0.72) 100%);
            border-color: rgba(0, 151, 167, 0.45);
        }

        .alert-box {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(166, 124, 82, 0.30);
        }

        .alert-box strong {
            color: #2e7d32;
        }

        .alert-box span {
            color: #4d5f57;
        }

        .section-label {
            color: #2e7d32;
        }

        .section-label::after {
            background: linear-gradient(90deg, #00bcd4 0%, #2e7d32 100%);
        }

        .small-muted {
            color: #5a6962;
        }

        .hud-banner {
            background: linear-gradient(130deg, rgba(223, 246, 249, 0.96) 0%, rgba(215, 243, 236, 0.95) 100%);
            border: 1px solid rgba(0, 151, 167, 0.40);
            color: #1f5860;
            box-shadow: inset 0 0 0 1px rgba(0, 151, 167, 0.08), 0 4px 12px rgba(0, 151, 167, 0.12);
        }

        .hud-banner.warn {
            border-color: rgba(166, 124, 82, 0.58);
            background: rgba(255, 243, 223, 0.96);
            color: #7d5a32;
        }

        .hud-dot {
            border-color: rgba(0, 151, 167, 0.35);
            background: rgba(255, 255, 255, 0.86);
            color: #007d8a;
        }

        .status-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(166, 124, 82, 0.30);
        }

        .status-card .label {
            color: #5a6962;
        }

        .status-card .value {
            color: #24322c;
        }

        .risk-card {
            border: 1px solid rgba(166, 124, 82, 0.32);
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(244, 236, 223, 0.88));
        }

        .risk-card h4 {
            color: #4f5e57;
        }

        .risk-card .risk-value {
            color: #213129;
        }

        .risk-card .risk-sub {
            color: #607068;
        }

        .risk-card.safe {
            border-color: rgba(76, 175, 80, 0.52);
            background: linear-gradient(145deg, rgba(226, 247, 230, 0.92), rgba(212, 239, 220, 0.88));
        }

        .risk-card.warn {
            border-color: rgba(166, 124, 82, 0.52);
            background: linear-gradient(145deg, rgba(250, 237, 220, 0.93), rgba(244, 228, 208, 0.89));
        }

        .risk-card.danger {
            border-color: rgba(192, 94, 94, 0.54);
            background: linear-gradient(145deg, rgba(255, 231, 228, 0.92), rgba(250, 219, 214, 0.88));
        }

        .kpi-tile {
            border: 1px solid rgba(166, 124, 82, 0.30);
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(247, 241, 230, 0.90));
            box-shadow: inset 0 0 0 1px rgba(166, 124, 82, 0.06);
        }

        .kpi-tile .kpi-label {
            color: #55675f;
        }

        .kpi-tile .kpi-value {
            color: #1f2f28;
        }

        .kpi-tile .kpi-sub {
            color: #5f7168;
        }

        .kpi-tile.aqua {
            border-color: rgba(0, 151, 167, 0.44);
            background: linear-gradient(145deg, rgba(228, 246, 248, 0.95), rgba(212, 242, 246, 0.9));
        }

        .kpi-tile.violet {
            border-color: rgba(166, 124, 82, 0.42);
            background: linear-gradient(145deg, rgba(250, 240, 227, 0.95), rgba(244, 231, 212, 0.9));
        }

        .kpi-tile.mint {
            border-color: rgba(76, 175, 80, 0.44);
            background: linear-gradient(145deg, rgba(231, 248, 233, 0.95), rgba(216, 242, 220, 0.9));
        }

        .disease-panel {
            border: 1px solid rgba(76, 175, 80, 0.36);
            background: linear-gradient(145deg, rgba(234, 249, 236, 0.94), rgba(224, 244, 228, 0.92));
        }

        .scan-result {
            border: 1px solid rgba(166, 124, 82, 0.34);
            background: rgba(255, 255, 255, 0.92);
            color: #22322b;
        }

        .scan-result.ok {
            border-color: rgba(76, 175, 80, 0.54);
            background: linear-gradient(135deg, rgba(223, 247, 227, 0.95), rgba(211, 240, 218, 0.92));
            color: #1f5b31;
        }

        .scan-result.warn {
            border-color: rgba(166, 124, 82, 0.56);
            background: linear-gradient(135deg, rgba(251, 239, 223, 0.95), rgba(246, 229, 206, 0.92));
            color: #7a5935;
        }

        .confidence-card {
            border: 1px solid rgba(0, 151, 167, 0.34);
            background: rgba(255, 255, 255, 0.94);
        }

        .confidence-card .label {
            color: #4f615a;
        }

        .confidence-card .value {
            color: #1e2f28;
        }

        .scan-severity {
            border: 1px solid rgba(166, 124, 82, 0.42);
            color: #263730;
            background: rgba(255, 255, 255, 0.94);
        }

        .scan-severity.healthy {
            border-color: rgba(76, 175, 80, 0.52);
            background: rgba(227, 247, 231, 0.95);
            color: #1f5c32;
        }

        .scan-severity.low {
            border-color: rgba(76, 175, 80, 0.44);
            background: rgba(238, 250, 241, 0.95);
            color: #2c6d40;
        }

        .scan-severity.moderate {
            border-color: rgba(166, 124, 82, 0.54);
            background: rgba(251, 241, 228, 0.95);
            color: #765634;
        }

        .scan-severity.high {
            border-color: rgba(192, 94, 94, 0.54);
            background: rgba(255, 233, 230, 0.95);
            color: #7c3e3e;
        }

        .suggestion-card {
            border: 1px solid rgba(166, 124, 82, 0.32);
            background: rgba(255, 255, 255, 0.94);
        }

        .suggestion-card .label {
            color: #2e7d32;
        }

        .suggestion-card li {
            color: #33463d;
        }

        .weather-hero {
            border: 1px solid rgba(0, 151, 167, 0.36);
            background: linear-gradient(145deg, rgba(227, 246, 248, 0.95), rgba(214, 242, 236, 0.92));
            box-shadow: inset 0 0 0 1px rgba(0, 151, 167, 0.08);
        }

        .weather-city {
            color: #20443a;
        }

        .weather-meta {
            color: #4d665e;
        }

        .weather-temp {
            color: #1f3b33;
        }

        .weather-condition {
            color: #365b4f;
        }

        .weather-chip {
            border: 1px solid rgba(0, 151, 167, 0.32);
            color: #35534a;
            background: rgba(255, 255, 255, 0.82);
        }

        .weather-detail-card {
            border: 1px solid rgba(166, 124, 82, 0.30);
            background: rgba(255, 255, 255, 0.92);
        }

        .weather-detail-card .label {
            color: #5b6c64;
        }

        .weather-detail-card .value {
            color: #24342d;
        }

        .weather-forecast-card {
            border: 1px solid rgba(166, 124, 82, 0.28);
            background: rgba(255, 255, 255, 0.92);
        }

        .weather-forecast-time {
            color: #5b6d64;
        }

        .weather-forecast-temp {
            color: #213229;
        }

        .weather-forecast-sub {
            color: #5e7067;
        }

        .command-pill {
            border: 1px solid rgba(166, 124, 82, 0.30);
            background: rgba(255, 255, 255, 0.9);
            color: #4d6058;
        }

        .command-pill strong {
            color: #284139;
        }

        .command-pill.good {
            border-color: rgba(76, 175, 80, 0.52);
            background: rgba(228, 247, 232, 0.94);
            color: #1f5d34;
        }

        .command-pill.warn {
            border-color: rgba(166, 124, 82, 0.54);
            background: rgba(250, 239, 224, 0.94);
            color: #7a5935;
        }

        .command-pill.alert {
            border-color: rgba(192, 94, 94, 0.56);
            background: rgba(255, 233, 230, 0.94);
            color: #7a3f3f;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.94) !important;
            border: 1px dashed rgba(166, 124, 82, 0.56) !important;
        }

        div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzoneInstructions"] p,
        div[data-testid="stFileUploaderDropzoneInstructions"] small {
            color: #3a4d45 !important;
        }

        div[data-testid="stFileUploaderDropzone"] svg {
            fill: #2e7d32 !important;
            color: #2e7d32 !important;
        }

        div[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, #00bcd4 0%, #0097a7 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(0, 151, 167, 0.66) !important;
        }

        div[data-testid="stFileUploaderDropzone"] button *,
        div[data-testid="stFileUploaderDropzone"] [role="button"] *,
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        [data-testid="stProgress"] > div > div {
            background: rgba(215, 204, 200, 0.48) !important;
            border: 1px solid rgba(166, 124, 82, 0.24);
        }

        [data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #00bcd4 0%, #4caf50 100%) !important;
        }

        .stNumberInput input, .stTextInput input {
            background: rgba(255, 255, 255, 0.96) !important;
            color: #263238 !important;
            -webkit-text-fill-color: #263238 !important;
            border: 1px solid rgba(166, 124, 82, 0.44) !important;
        }

        .stNumberInput input::placeholder, .stTextInput input::placeholder {
            color: #7b8b84 !important;
            opacity: 1 !important;
        }

        .stNumberInput input:focus, .stTextInput input:focus {
            border-color: rgba(0, 151, 167, 0.72) !important;
            box-shadow: 0 0 0 1px rgba(0, 151, 167, 0.22) !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: #4f5e57 !important;
        }

        /* Fix selectbox + expander contrast for light theme */
        [data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.96) !important;
            border: 1px solid rgba(166, 124, 82, 0.44) !important;
            border-radius: 10px !important;
        }

        [data-baseweb="select"] *,
        [data-baseweb="select"] input,
        [data-baseweb="select"] span,
        [data-baseweb="select"] div {
            color: #263238 !important;
            -webkit-text-fill-color: #263238 !important;
            opacity: 1 !important;
        }

        [data-baseweb="select"] svg {
            fill: #4f5e57 !important;
        }

        [role="listbox"] {
            background: rgba(255, 251, 243, 0.98) !important;
            border: 1px solid rgba(166, 124, 82, 0.40) !important;
        }

        [role="option"] {
            color: #263238 !important;
            background: rgba(255, 251, 243, 0.98) !important;
        }

        [role="option"]:hover {
            background: rgba(0, 188, 212, 0.14) !important;
        }

        [role="option"][aria-selected="true"] {
            background: rgba(76, 175, 80, 0.16) !important;
        }

        div[data-testid="stExpander"] > details {
            background: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid rgba(166, 124, 82, 0.42) !important;
            border-radius: 12px !important;
        }

        div[data-testid="stExpander"] > details > summary,
        div[data-testid="stExpander"] > details > summary * {
            color: #263238 !important;
            font-weight: 700 !important;
        }

        div[data-testid="stExpander"] > details > summary svg {
            fill: #2e7d32 !important;
        }

        /* Final contrast accessibility pass */
        .stApp,
        .stApp p,
        .stApp span,
        .stApp li,
        .stApp label,
        .stApp div {
            color: #263238;
        }

        h1, h2, h3, h4 {
            color: #1d2d26 !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"] p,
        .small-muted {
            color: #44534d !important;
        }

        [data-testid="stWidgetLabel"] p,
        .stCheckbox label p,
        .stRadio label p,
        .stSelectbox label,
        .stTextInput label,
        .stNumberInput label {
            color: #36463f !important;
            font-weight: 600 !important;
        }

        .section-label {
            color: #1f7d3b !important;
        }

        .weather-meta,
        .weather-condition,
        .weather-chip,
        .weather-forecast-time,
        .weather-forecast-sub,
        .weather-detail-card .label,
        .kpi-tile .kpi-label,
        .kpi-tile .kpi-sub,
        .risk-card .risk-sub,
        .status-card .label,
        .alert-box span,
        .command-pill {
            color: #44554d !important;
        }

        .kpi-tile .kpi-value,
        .risk-card .risk-value,
        .status-card .value,
        .weather-city,
        .weather-temp,
        .weather-detail-card .value,
        .weather-forecast-temp,
        .confidence-card .value {
            color: #1f2f28 !important;
        }

        .alert-box strong,
        .suggestion-card .label,
        .command-pill strong {
            color: #205f33 !important;
        }

        [data-testid="stAlertContainer"] p,
        [data-testid="stAlertContainer"] li,
        [data-testid="stAlertContainer"] span {
            color: #1f2f28 !important;
            font-weight: 600;
        }

        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] * {
            color: #263238 !important;
        }

        .stButton > button,
        .stButton > button * {
            color: #ffffff !important;
        }

        div[data-testid="stExpander"] > details > summary,
        div[data-testid="stExpander"] > details > summary * {
            color: #2b3b34 !important;
        }

        /* Modern colorful UI layer */
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(61, 173, 255, 0.18) 0%, rgba(0, 0, 0, 0) 28%),
                radial-gradient(circle at 88% 14%, rgba(121, 115, 255, 0.16) 0%, rgba(0, 0, 0, 0) 28%),
                radial-gradient(circle at 85% 88%, rgba(50, 205, 164, 0.16) 0%, rgba(0, 0, 0, 0) 30%),
                radial-gradient(circle at 12% 86%, rgba(255, 184, 92, 0.14) 0%, rgba(0, 0, 0, 0) 30%),
                linear-gradient(135deg, #f4f8ff 0%, #eef5ff 48%, #f4fbf4 100%) !important;
        }

        .main .block-container {
            max-width: 1280px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:
                linear-gradient(160deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.82)) !important;
            border: 1px solid rgba(107, 151, 231, 0.22) !important;
            border-radius: 20px !important;
            box-shadow: 0 12px 28px rgba(54, 87, 122, 0.12) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(49, 163, 204, 0.42) !important;
            box-shadow: 0 16px 32px rgba(54, 87, 122, 0.16) !important;
        }

        .top-shell {
            background:
                radial-gradient(circle at 92% 12%, rgba(133, 111, 255, 0.20), rgba(0, 0, 0, 0) 42%),
                radial-gradient(circle at 10% 90%, rgba(60, 211, 173, 0.16), rgba(0, 0, 0, 0) 44%),
                linear-gradient(140deg, rgba(252, 255, 255, 0.94), rgba(242, 248, 255, 0.92)) !important;
            border: 1px solid rgba(108, 168, 228, 0.32) !important;
            border-radius: 24px !important;
            box-shadow: 0 16px 34px rgba(43, 78, 115, 0.16) !important;
        }

        .mini-tab {
            border: 1px solid rgba(127, 164, 216, 0.26) !important;
            background: rgba(255, 255, 255, 0.78) !important;
            color: #33485f !important;
            font-weight: 700 !important;
        }

        .mini-tab.active {
            background: linear-gradient(135deg, rgba(106, 204, 255, 0.28) 0%, rgba(145, 131, 255, 0.26) 100%) !important;
            border-color: rgba(75, 162, 231, 0.58) !important;
            color: #123752 !important;
            box-shadow: 0 6px 16px rgba(87, 149, 217, 0.18);
        }

        .hud-banner {
            background: linear-gradient(120deg, rgba(75, 204, 255, 0.22), rgba(77, 222, 174, 0.20)) !important;
            border: 1px solid rgba(58, 174, 214, 0.40) !important;
            color: #10495a !important;
        }

        .hud-banner.warn {
            background: linear-gradient(120deg, rgba(255, 205, 106, 0.28), rgba(255, 168, 122, 0.22)) !important;
            border-color: rgba(217, 141, 72, 0.5) !important;
            color: #7a4c22 !important;
        }

        .section-label {
            color: #167546 !important;
            letter-spacing: 0.10em !important;
        }

        .section-label::after {
            background: linear-gradient(90deg, #1cc5ea 0%, #35c981 52%, #8a7cff 100%) !important;
        }

        .alert-box {
            background: rgba(255, 255, 255, 0.85) !important;
            border: 1px solid rgba(129, 173, 227, 0.34) !important;
            box-shadow: inset 0 0 0 1px rgba(143, 191, 243, 0.10);
        }

        .command-pill {
            border-color: rgba(125, 174, 222, 0.34) !important;
            background: rgba(255, 255, 255, 0.86) !important;
            color: #2d455a !important;
        }

        .command-pill.good {
            background: linear-gradient(120deg, rgba(86, 217, 143, 0.22), rgba(55, 196, 165, 0.18)) !important;
            border-color: rgba(59, 172, 122, 0.44) !important;
            color: #1f6c45 !important;
        }

        .command-pill.warn {
            background: linear-gradient(120deg, rgba(255, 213, 121, 0.28), rgba(255, 173, 112, 0.22)) !important;
            border-color: rgba(209, 140, 74, 0.44) !important;
            color: #7a4e26 !important;
        }

        .command-pill.alert {
            background: linear-gradient(120deg, rgba(255, 161, 161, 0.24), rgba(255, 128, 128, 0.2)) !important;
            border-color: rgba(210, 101, 101, 0.42) !important;
            color: #7b3535 !important;
        }

        .kpi-tile {
            border: 1px solid rgba(120, 168, 224, 0.30) !important;
            box-shadow: 0 10px 22px rgba(50, 90, 130, 0.10) !important;
        }

        .kpi-tile.aqua {
            background: linear-gradient(145deg, rgba(221, 247, 255, 0.95), rgba(206, 239, 255, 0.88)) !important;
            border-color: rgba(72, 177, 233, 0.44) !important;
        }

        .kpi-tile.violet {
            background: linear-gradient(145deg, rgba(236, 231, 255, 0.95), rgba(224, 216, 255, 0.88)) !important;
            border-color: rgba(138, 122, 236, 0.42) !important;
        }

        .kpi-tile.mint {
            background: linear-gradient(145deg, rgba(224, 250, 239, 0.95), rgba(206, 244, 226, 0.88)) !important;
            border-color: rgba(64, 188, 133, 0.40) !important;
        }

        .risk-card {
            box-shadow: 0 8px 18px rgba(61, 95, 131, 0.08);
        }

        .weather-hero {
            background:
                radial-gradient(circle at 85% 18%, rgba(133, 111, 255, 0.18), rgba(0, 0, 0, 0) 44%),
                radial-gradient(circle at 12% 86%, rgba(46, 213, 165, 0.16), rgba(0, 0, 0, 0) 44%),
                linear-gradient(145deg, rgba(230, 247, 252, 0.95), rgba(223, 244, 236, 0.92)) !important;
            border: 1px solid rgba(69, 175, 220, 0.36) !important;
        }

        .weather-detail-card,
        .weather-forecast-card,
        .status-card,
        .confidence-card,
        .suggestion-card {
            border-color: rgba(117, 166, 222, 0.28) !important;
            box-shadow: 0 6px 16px rgba(61, 95, 131, 0.07);
        }

        .scan-result.ok {
            background: linear-gradient(135deg, rgba(214, 246, 227, 0.96), rgba(199, 239, 221, 0.9)) !important;
        }

        .scan-result.warn {
            background: linear-gradient(135deg, rgba(255, 236, 214, 0.96), rgba(249, 224, 201, 0.9)) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1bc8e8 0%, #24b7ca 45%, #2ecf8b 100%) !important;
            border: none !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 22px rgba(45, 171, 201, 0.26) !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #08b7dc 0%, #12a9bb 45%, #25bd7d 100%) !important;
            box-shadow: 0 14px 26px rgba(45, 171, 201, 0.30) !important;
        }

        .stNumberInput input, .stTextInput input, [data-baseweb="select"] > div {
            border: 1px solid rgba(121, 170, 225, 0.42) !important;
            background: rgba(255, 255, 255, 0.97) !important;
            color: #203244 !important;
        }

        .stNumberInput input:focus, .stTextInput input:focus {
            border-color: rgba(40, 182, 225, 0.78) !important;
            box-shadow: 0 0 0 2px rgba(40, 182, 225, 0.20) !important;
        }

        [data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #1cc5ea 0%, #35c981 52%, #8a7cff 100%) !important;
        }

        /* Ensure file uploader text/button stay visible on dark dropzone */
        div[data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(145deg, rgba(20, 30, 46, 0.95), rgba(29, 39, 56, 0.94)) !important;
            border: 1px solid rgba(117, 167, 229, 0.36) !important;
        }

        div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] small,
        div[data-testid="stFileUploaderDropzone"] small,
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzoneInstructions"] p,
        div[data-testid="stFileUploaderDropzoneInstructions"] small {
            color: #dff3ff !important;
            -webkit-text-fill-color: #dff3ff !important;
            opacity: 1 !important;
        }

        div[data-testid="stFileUploaderDropzone"] svg {
            fill: #9fd8ff !important;
            color: #9fd8ff !important;
        }

        div[data-testid="stFileUploaderDropzone"] button,
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"] {
            background: rgba(24, 36, 56, 0.88) !important;
            border: 1px solid rgba(132, 188, 244, 0.36) !important;
            border-radius: 10px !important;
            color: #ecf7ff !important;
            -webkit-text-fill-color: #ecf7ff !important;
        }

        div[data-testid="stFileUploaderDropzone"] button *,
        div[data-testid="stFileUploaderDropzone"] [role="button"] *,
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"] * {
            color: #ecf7ff !important;
            fill: #ecf7ff !important;
            -webkit-text-fill-color: #ecf7ff !important;
            opacity: 1 !important;
        }

        /* Strong fallback to force uploader text visibility */
        div[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stFileUploaderDropzoneInstructions"] *,
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"],
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"] *,
        div[data-testid="stFileUploaderDropzone"] [role="button"],
        div[data-testid="stFileUploaderDropzone"] [role="button"] * {
            color: #f4fbff !important;
            fill: #f4fbff !important;
            -webkit-text-fill-color: #f4fbff !important;
            opacity: 1 !important;
        }

        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"],
        div[data-testid="stFileUploaderDropzone"] [role="button"],
        div[data-testid="stFileUploaderDropzone"] button {
            background: rgba(24, 36, 56, 0.92) !important;
            border: 1px solid rgba(146, 199, 250, 0.42) !important;
        }

        /* ===============================
           Final Dark Neon Overrides
           =============================== */
        :root {
            --bg-a: #040913;
            --bg-b: #071524;
            --bg-c: #0b1f33;
            --glass: rgba(10, 24, 42, 0.84);
            --glass-soft: rgba(12, 28, 48, 0.88);
            --stroke: rgba(95, 203, 255, 0.34);
            --text: #ecf6ff;
            --muted: #a8c7e6;
            --metric-label: #b7d8f3;
            --status-bg: rgba(12, 31, 50, 0.86);
            --status-stroke: rgba(102, 204, 255, 0.32);
            --glow-a: rgba(44, 217, 255, 0.20);
            --glow-b: rgba(130, 110, 255, 0.18);
        }

        @keyframes neon-bg-drift {
            0% { background-position: 0% 0%, 100% 0%, 100% 100%, 0% 100%; }
            50% { background-position: 8% 4%, 92% 6%, 94% 92%, 6% 96%; }
            100% { background-position: 0% 0%, 100% 0%, 100% 100%, 0% 100%; }
        }

        @keyframes neon-pulse {
            0%, 100% { box-shadow: 0 0 0 rgba(80, 188, 255, 0), 0 0 0 rgba(138, 113, 255, 0); }
            50% { box-shadow: 0 0 18px rgba(80, 188, 255, 0.24), 0 0 26px rgba(138, 113, 255, 0.16); }
        }

        @keyframes panel-rise {
            from { opacity: 0; transform: translateY(10px) scale(0.996); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes hud-sweep {
            0% { transform: translateX(-130%) skewX(-18deg); opacity: 0; }
            22% { opacity: 0.45; }
            38% { opacity: 0; }
            100% { transform: translateX(180%) skewX(-18deg); opacity: 0; }
        }

        @keyframes button-sheen {
            0% { transform: translateX(-140%) skewX(-20deg); opacity: 0; }
            22% { opacity: 0.38; }
            46% { opacity: 0; }
            100% { transform: translateX(180%) skewX(-20deg); opacity: 0; }
        }

        @keyframes float-soft {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-2px); }
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 8%, rgba(45, 212, 255, 0.16), rgba(0, 0, 0, 0) 34%),
                radial-gradient(circle at 86% 12%, rgba(139, 103, 255, 0.15), rgba(0, 0, 0, 0) 36%),
                radial-gradient(circle at 84% 88%, rgba(51, 255, 179, 0.10), rgba(0, 0, 0, 0) 34%),
                linear-gradient(135deg, #040913 0%, #071524 44%, #0a1e33 100%) !important;
            color: #ecf6ff !important;
            background-size: 120% 120%, 120% 120%, 120% 120%, 100% 100% !important;
            animation: neon-bg-drift 22s ease-in-out infinite;
        }

        .main .block-container {
            max-width: 1320px;
            padding-top: 1.05rem;
            padding-bottom: 2.6rem;
        }

        p, span, label, li, small, strong, h1, h2, h3, h4, h5 {
            color: #e8f4ff !important;
        }

        .small-muted {
            color: #a7c8e8 !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7, 20, 34, 0.97) 0%, rgba(8, 28, 46, 0.96) 100%) !important;
            border-right: 1px solid rgba(83, 182, 236, 0.32) !important;
        }

        section[data-testid="stSidebar"] * {
            color: #dff1ff !important;
        }

        .side-intro {
            border: 1px solid rgba(95, 196, 247, 0.40) !important;
            border-radius: 14px !important;
            background: linear-gradient(145deg, rgba(9, 26, 43, 0.96), rgba(12, 33, 54, 0.94)) !important;
            box-shadow: inset 0 0 0 1px rgba(122, 215, 255, 0.10), 0 10px 20px rgba(3, 10, 17, 0.34) !important;
        }

        .side-intro .title {
            color: #9fe3ff !important;
        }

        .side-intro .sub {
            color: #c4e4ff !important;
        }

        section[data-testid="stSidebar"] label[data-baseweb="radio"],
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: linear-gradient(130deg, rgba(13, 33, 53, 0.95), rgba(10, 27, 44, 0.94)) !important;
            border: 1px solid rgba(87, 179, 236, 0.34) !important;
            border-radius: 12px !important;
            padding: 0.86rem 0.96rem !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] > label p,
        section[data-testid="stSidebar"] label[data-baseweb="radio"] p,
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
            color: #dff2ff !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked),
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(130deg, rgba(20, 55, 89, 0.98) 0%, rgba(39, 78, 128, 0.95) 52%, rgba(32, 69, 110, 0.96) 100%) !important;
            border-color: rgba(92, 214, 255, 0.72) !important;
            box-shadow: 0 0 0 1px rgba(111, 217, 255, 0.22), 0 10px 22px rgba(7, 20, 33, 0.42) !important;
        }

        .top-shell {
            background: linear-gradient(145deg, rgba(7, 22, 38, 0.94), rgba(10, 30, 50, 0.92)) !important;
            border: 1px solid rgba(92, 196, 247, 0.38) !important;
            box-shadow: 0 20px 42px rgba(2, 8, 14, 0.48), inset 0 0 0 1px rgba(112, 213, 255, 0.08) !important;
            position: relative;
            overflow: hidden;
            animation: panel-rise 0.5s ease both;
        }

        .top-shell::before {
            content: "";
            position: absolute;
            top: -28%;
            left: -22%;
            width: 35%;
            height: 160%;
            background: linear-gradient(90deg, rgba(125, 214, 255, 0), rgba(125, 214, 255, 0.28), rgba(125, 214, 255, 0));
            filter: blur(2px);
            pointer-events: none;
            animation: hud-sweep 7.8s linear infinite;
        }

        .mini-tab {
            color: #9ec2e4 !important;
            border-color: rgba(83, 173, 230, 0.28) !important;
            background: rgba(9, 25, 41, 0.72) !important;
        }

        .mini-tab.active {
            color: #eaf7ff !important;
            background: linear-gradient(135deg, rgba(32, 162, 229, 0.30) 0%, rgba(132, 99, 255, 0.28) 100%) !important;
            border-color: rgba(118, 215, 255, 0.64) !important;
            box-shadow: 0 0 18px rgba(67, 197, 255, 0.20) !important;
            animation: neon-pulse 2.8s ease-in-out infinite;
        }

        .alert-box {
            background: rgba(10, 28, 46, 0.90) !important;
            border: 1px solid rgba(98, 198, 248, 0.34) !important;
        }

        .alert-box strong {
            color: #8ce3ff !important;
        }

        .alert-box span {
            color: #c5e6ff !important;
        }

        .section-label {
            color: #67d4ff !important;
        }

        .section-label::after {
            background: linear-gradient(90deg, #38d3ff 0%, #8a71ff 55%, #37efb7 100%) !important;
        }

        .hud-banner {
            background: linear-gradient(130deg, rgba(28, 80, 110, 0.92), rgba(23, 96, 122, 0.90)) !important;
            border: 1px solid rgba(102, 216, 255, 0.56) !important;
            color: #dff5ff !important;
            text-shadow: 0 0 12px rgba(68, 199, 255, 0.16);
            animation: neon-pulse 3.6s ease-in-out infinite;
        }

        .hud-banner.warn {
            background: linear-gradient(130deg, rgba(112, 74, 19, 0.92), rgba(130, 86, 26, 0.90)) !important;
            border-color: rgba(255, 205, 114, 0.62) !important;
            color: #ffe9bf !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(8, 24, 40, 0.85), rgba(10, 29, 49, 0.84)) !important;
            border: 1px solid rgba(98, 198, 248, 0.30) !important;
            box-shadow: 0 16px 36px rgba(1, 8, 14, 0.44), inset 0 0 0 1px rgba(130, 214, 255, 0.06) !important;
            animation: panel-rise 0.45s ease both;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(2n) {
            animation-delay: 0.05s;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(3n) {
            animation-delay: 0.1s;
        }

        div[data-testid="stMetric"],
        .status-card,
        .risk-card,
        .kpi-tile,
        .disease-panel,
        .scan-result,
        .confidence-card,
        .scan-severity,
        .suggestion-card,
        .weather-hero,
        .weather-detail-card,
        .weather-forecast-card,
        .command-pill {
            background: linear-gradient(145deg, rgba(10, 28, 46, 0.90), rgba(11, 31, 52, 0.88)) !important;
            border: 1px solid rgba(102, 202, 250, 0.32) !important;
            color: #e9f6ff !important;
        }

        .risk-card.safe,
        .scan-result.ok,
        .scan-severity.healthy,
        .scan-severity.low,
        .command-pill.good {
            background: linear-gradient(135deg, rgba(12, 73, 72, 0.90), rgba(12, 90, 86, 0.86)) !important;
            border-color: rgba(69, 238, 193, 0.56) !important;
            color: #dcfff4 !important;
        }

        .risk-card.warn,
        .scan-result.warn,
        .scan-severity.moderate,
        .command-pill.warn {
            background: linear-gradient(135deg, rgba(95, 72, 26, 0.90), rgba(118, 84, 33, 0.86)) !important;
            border-color: rgba(255, 197, 96, 0.62) !important;
            color: #ffe8bc !important;
        }

        .risk-card.danger,
        .scan-severity.high,
        .command-pill.alert {
            background: linear-gradient(135deg, rgba(110, 40, 58, 0.90), rgba(132, 47, 70, 0.86)) !important;
            border-color: rgba(255, 136, 151, 0.62) !important;
            color: #ffdbe2 !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #15cdf5 0%, #4b8dff 45%, #2fe7b0 100%) !important;
            color: #f5fcff !important;
            border: 1px solid rgba(128, 221, 255, 0.50) !important;
            border-radius: 14px !important;
            box-shadow: 0 12px 24px rgba(27, 165, 224, 0.28), 0 0 18px rgba(80, 188, 255, 0.20) !important;
            position: relative;
            overflow: hidden;
            animation: neon-pulse 3.1s ease-in-out infinite;
        }

        .stButton > button::after {
            content: "";
            position: absolute;
            top: -24%;
            left: -18%;
            width: 32%;
            height: 150%;
            background: linear-gradient(90deg, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.32), rgba(255, 255, 255, 0));
            transform: skewX(-20deg);
            pointer-events: none;
            animation: button-sheen 5.8s linear infinite;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #08bee6 0%, #417cf1 45%, #29d39f 100%) !important;
            box-shadow: 0 14px 28px rgba(28, 169, 228, 0.34), 0 0 20px rgba(87, 195, 255, 0.28) !important;
        }

        .weather-hero,
        .disease-panel {
            animation: float-soft 5.6s ease-in-out infinite;
        }

        .stNumberInput input,
        .stTextInput input,
        [data-baseweb="select"] > div,
        [role="listbox"],
        [role="option"],
        div[data-testid="stExpander"] > details {
            background: rgba(8, 28, 46, 0.93) !important;
            border: 1px solid rgba(93, 187, 241, 0.42) !important;
            color: #e8f4ff !important;
            -webkit-text-fill-color: #e8f4ff !important;
        }

        .stNumberInput input::placeholder,
        .stTextInput input::placeholder {
            color: #9bc0e2 !important;
            opacity: 1 !important;
        }

        [data-baseweb="select"] *,
        [role="option"] *,
        [data-testid="stWidgetLabel"] p {
            color: #dff0ff !important;
            -webkit-text-fill-color: #dff0ff !important;
        }

        [data-testid="stProgress"] > div > div {
            background: rgba(13, 35, 56, 0.85) !important;
            border: 1px solid rgba(96, 193, 247, 0.30) !important;
        }

        [data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #31d7ff 0%, #8a71ff 52%, #2fe7b0 100%) !important;
        }

        /* Uploader contrast hard-fix for dark neon theme */
        div[data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(145deg, rgba(10, 24, 40, 0.96), rgba(12, 29, 47, 0.94)) !important;
            border: 1px dashed rgba(110, 210, 255, 0.52) !important;
            box-shadow: inset 0 0 0 1px rgba(116, 214, 255, 0.10) !important;
        }

        div[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stFileUploaderDropzoneInstructions"] *,
        div[data-testid="stFileUploaderDropzone"] button,
        div[data-testid="stFileUploaderDropzone"] button *,
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"],
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"] * {
            color: #f1f9ff !important;
            fill: #f1f9ff !important;
            -webkit-text-fill-color: #f1f9ff !important;
            opacity: 1 !important;
        }

        div[data-testid="stFileUploaderDropzone"] button,
        div[data-testid="stFileUploaderDropzone"] [data-baseweb="button"] {
            background: rgba(16, 38, 60, 0.92) !important;
            border: 1px solid rgba(128, 219, 255, 0.48) !important;
            border-radius: 10px !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(103, 200, 248, 0.34) !important;
            box-shadow: inset 0 0 0 1px rgba(122, 216, 255, 0.08);
        }

        /* Force readable markdown/body text in cards and lists */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] ol,
        [data-testid="stMarkdownContainer"] ul,
        [data-testid="stCaptionContainer"] p,
        [data-testid="stAlertContainer"] p,
        [data-testid="stAlertContainer"] li,
        [data-testid="stAlertContainer"] span {
            color: #dff0ff !important;
        }

        [data-testid="stMarkdownContainer"] li::marker {
            color: #83d7ff !important;
        }

        @media (prefers-reduced-motion: reduce) {
            .stApp,
            .top-shell,
            .top-shell::before,
            .mini-tab.active,
            .hud-banner,
            div[data-testid="stVerticalBlockBorderWrapper"],
            .stButton > button,
            .stButton > button::after,
            .weather-hero,
            .disease-panel {
                animation: none !important;
                transition: none !important;
            }
        }

        @media (max-width: 900px) {
            .weather-forecast-grid {
                gap: 8px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_chart_theme() -> None:
    """Set chart colors for dark neon dashboard readability."""
    plt.rcParams.update(
        {
            "axes.edgecolor": "#66b9ef",
            "axes.labelcolor": "#d9eeff",
            "axes.titlecolor": "#f0f8ff",
            "xtick.color": "#b9daf3",
            "ytick.color": "#b9daf3",
            "grid.color": "#2f5774",
            "text.color": "#e6f4ff",
        }
    )


def render_top_shell(active_page: str, title: str, subtitle: str) -> None:
    """Render a top chrome bar and page summary, inspired by dashboard mockups."""
    pages = ["Dashboard", "Disease Scanner", "Weather", "Analytics"]
    tabs = []
    for page in pages:
        active_cls = "active" if page == active_page else ""
        tabs.append(f'<span class="mini-tab {active_cls}">{html.escape(page)}</span>')
    tabs_html = "".join(tabs)

    weather_snapshot = st.session_state.get("last_weather_result")
    irrigation_snapshot = st.session_state.get("last_irrigation_result")
    disease_snapshot = st.session_state.get("last_disease_result")
    banner_class = "hud-banner"
    if weather_snapshot:
        city = html.escape(str(weather_snapshot.get("city_name", "Unknown")))
        condition = html.escape(str(weather_snapshot.get("condition", "N/A")))
        temp = weather_snapshot.get("temperature", "-")
        if weather_snapshot.get("rain_forecast", 0):
            weather_line = f"{city}: {condition}, {temp} C. Rain signal detected."
            banner_class = "hud-banner warn"
            banner_text = "Impending storm window: heavy rain likely soon. Secure irrigation cycles."
        else:
            weather_line = f"{city}: {condition}, {temp} C. No rain signal."
            banner_text = "Weather stable. Automation is active for irrigation and disease monitoring."
    else:
        weather_line = "Weather page not synced yet. Load live weather for alerts."
        banner_text = "Load current weather feed to activate live storm alerting."

    if irrigation_snapshot:
        irri_state = str(irrigation_snapshot.get("irrigation_state", "N/A"))
        irri_water = float(irrigation_snapshot.get("water_required", 0.0))
        irri_class = "warn" if irri_state == "ON" else "good"
        irri_value = f"{irri_state} | {irri_water:.1f} L"
    else:
        irri_class = ""
        irri_value = "Pending"

    if disease_snapshot:
        disease_name = str(disease_snapshot.get("disease_name", "Unknown"))
        disease_conf = float(disease_snapshot.get("confidence", 0.0)) * 100.0
        if disease_name.startswith("No Disease"):
            disease_class = "good"
            disease_value = f"Healthy | {disease_conf:.0f}%"
        else:
            disease_class = "alert"
            disease_value = f"Risk | {disease_conf:.0f}%"
    else:
        disease_class = ""
        disease_value = "No scan"

    if weather_snapshot:
        weather_temp = float(weather_snapshot.get("temperature", 0.0))
        weather_condition = str(weather_snapshot.get("condition", "N/A"))
        weather_class = "warn" if int(weather_snapshot.get("rain_forecast", 0)) else "good"
        weather_value = f"{weather_temp:.1f} C | {weather_condition}"
    else:
        weather_class = ""
        weather_value = "Not loaded"

    pills_html = (
        f'<span class="command-pill {irri_class}"><strong>Irrigation</strong>{html.escape(irri_value)}</span>'
        f'<span class="command-pill {disease_class}"><strong>Disease</strong>{html.escape(disease_value)}</span>'
        f'<span class="command-pill {weather_class}"><strong>Weather</strong>{html.escape(weather_value)}</span>'
    )

    st.markdown(
        f"""
        <div class="top-shell">
          <div class="chrome-row">
            <div class="mini-tabs">{tabs_html}</div>
            <div class="hud-actions">
              <span class="hud-dot">N</span>
              <span class="hud-dot">U</span>
            </div>
          </div>
          <div class="{banner_class}">{html.escape(banner_text)}</div>
          <div class="alert-row">
            <div>
              <div class="section-label">{html.escape(title)}</div>
              <div style="font-size: 1.35rem; font-weight: 700; color: #e7f5ff;">{html.escape(subtitle)}</div>
            </div>
            <div class="alert-box">
              <strong>Weather Alerts</strong>
              <span>{weather_line}</span>
            </div>
          </div>
          <div class="command-pill-row">{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_alert_messages(
    result: dict[str, object] | None,
    weather_snapshot: dict[str, object] | None,
    statuses: dict[str, str],
) -> list[str]:
    """Create short operational alerts for the dashboard."""
    alerts: list[str] = []

    if statuses["soil"] == "Very Dry":
        alerts.append("Soil moisture is critically low and needs attention.")
    if statuses["temperature"] == "High Heat":
        alerts.append("High heat may increase evapotranspiration losses.")
    if statuses["humidity"] == "Dry Air":
        alerts.append("Dry air condition may accelerate plant water stress.")

    if weather_snapshot and weather_snapshot.get("rain_forecast", 0):
        alerts.append("Rain signal detected from live weather source.")

    if result:
        if str(result.get("irrigation_state")) == "ON":
            alerts.append("Irrigation should run in the current cycle.")
        if float(result.get("efficiency", 0.0)) < 60:
            alerts.append("Efficiency score is low. Review threshold and pump flow.")
    else:
        alerts.append("Run irrigation prediction to activate full advisory mode.")

    if not alerts:
        alerts.append("All indicators are stable. No immediate warning.")
    return alerts[:5]


def create_ring_chart(value: float, label: str, color: str) -> plt.Figure:
    """Create a compact ring chart for KPI visualization."""
    bounded = float(np.clip(value, 0, 100))
    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    ax.pie(
        [bounded, 100 - bounded],
        startangle=90,
        counterclock=False,
        colors=[color, "#2a4960"],
        wedgeprops={"width": 0.28, "edgecolor": "white"},
    )
    ax.text(0, 0, f"{bounded:.0f}%", ha="center", va="center", fontsize=16, fontweight="bold", color="#eaf6ff")
    ax.set_title(label, fontsize=10, color="#a6cfe6")
    ax.axis("equal")
    fig.patch.set_alpha(0.0)
    return fig


def calculate_crop_health_score(
    soil_moisture: float,
    humidity: float,
    disease_snapshot: dict[str, object] | None,
) -> float:
    """Estimate crop health score for dashboard KPI."""
    score = 88.0
    score -= abs(55.0 - soil_moisture) * 0.45
    score -= abs(62.0 - humidity) * 0.20

    if disease_snapshot:
        name = str(disease_snapshot.get("disease_name", ""))
        confidence = float(disease_snapshot.get("confidence", 0.0))
        affected_area = float(disease_snapshot.get("diseased_area_percent", 0.0))
        if name and not name.startswith("No Disease"):
            score -= 18.0 + 22.0 * confidence
            score -= min(22.0, affected_area * 0.45)
        else:
            score += 3.5

    return round(float(np.clip(score, 0.0, 100.0)), 1)


def render_status_card(label: str, value: str) -> None:
    """Render compact status card with full readable text."""
    st.markdown(
        f"""
        <div class="status-card">
          <div class="label">{html.escape(label)}</div>
          <div class="value">{html.escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(title: str, value: str, subtitle: str, level: str) -> None:
    """Render a themed risk status card."""
    css_level = level if level in {"safe", "warn", "danger"} else "warn"
    st.markdown(
        f"""
        <div class="risk-card {css_level}">
          <h4>{html.escape(title)}</h4>
          <p class="risk-value">{html.escape(value)}</p>
          <div class="risk-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_tile(label: str, value: str, subtitle: str, tone: str = "aqua") -> None:
    """Render large KPI tile for command-center style dashboard."""
    tone_class = tone if tone in {"aqua", "violet", "mint"} else "aqua"
    st.markdown(
        f"""
        <div class="kpi-tile {tone_class}">
          <div class="kpi-label">{html.escape(label)}</div>
          <div class="kpi-value">{html.escape(value)}</div>
          <div class="kpi-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_css_class(severity: str) -> str:
    """Map severity text to CSS class names."""
    normalized = severity.strip().lower()
    if normalized in {"healthy", "minimal"}:
        return "healthy"
    if normalized in {"low"}:
        return "low"
    if normalized in {"moderate"}:
        return "moderate"
    return "high"


def render_suggestion_card(title: str, suggestions: list[str]) -> None:
    """Render disease suggestions inside styled panel."""
    safe_items = [item.strip() for item in suggestions if item and item.strip()]
    if not safe_items:
        return

    list_html = "".join(f"<li>{html.escape(item)}</li>" for item in safe_items)
    st.markdown(
        (
            '<div class="suggestion-card">'
            f'<div class="label">{html.escape(title)}</div>'
            f"<ul>{list_html}</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def dashboard_page() -> None:
    initialize_session_defaults()
    render_top_shell(
        active_page="Dashboard",
        title="Irrigation Operations",
        subtitle="Smart field control, alerting, and crop intelligence",
    )

    weather_snapshot = st.session_state.last_weather_result
    disease_snapshot = st.session_state.last_disease_result
    result = st.session_state.last_irrigation_result

    left_col, right_col = st.columns([1.05, 1.0], gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Irrigation Control</div>', unsafe_allow_html=True)
            st.markdown("### Smart Irrigation")
            st.caption("Provide latest sensor values to generate the irrigation decision.")

            c1, c2 = st.columns(2)
            with c1:
                soil_moisture = st.number_input(
                    "Soil Moisture (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key="soil_moisture_input",
                )
                temperature = st.number_input(
                    "Temperature (C)",
                    min_value=-10.0,
                    max_value=60.0,
                    step=0.1,
                    key="temperature_input",
                )
            with c2:
                humidity = st.number_input(
                    "Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key="humidity_input",
                )
                rain_forecast = st.checkbox("Rain Forecast Available", key="rain_forecast_input")

            crop_name = st.selectbox("Crop Type", options=list(CROP_TYPES.keys()), index=0)

            with st.expander("Advanced Controls", expanded=False):
                irrigation_threshold = st.number_input(
                    "Irrigation ON Threshold (L)",
                    min_value=0.0,
                    value=8.0,
                    step=0.5,
                )
                field_area = st.number_input("Field Area (m2)", min_value=1.0, value=500.0, step=10.0)
                pump_flow_rate = st.number_input("Pump Flow Rate (L/min)", min_value=0.1, value=18.0, step=0.5)

            if weather_snapshot:
                rain_text = "Yes" if weather_snapshot.get("rain_forecast", 0) else "No"
                st.info(
                    f"Live weather synced from {weather_snapshot.get('city_name', 'Unknown')} | "
                    f"{weather_snapshot.get('temperature', '-')} C | Rain signal: {rain_text}"
                )

            predict_clicked = st.button(
                "Predict Irrigation",
                type="primary",
                use_container_width=True,
                key="predict_irrigation_dashboard_btn",
            )

    if predict_clicked:
        try:
            response = requests.post(
                "https://britni-chrismal-undeliciously.ngrok-free.dev/webhook-test/2d6807f2-82a6-46b3-b230-2e2c2152245e",
                json={"status": "dashboard_trigger"},
                timeout=10,
            )
            if response.status_code == 200:
                st.success("✅ SMS Sent!")
            else:
                st.error("❌ AI Offline")
        except Exception:
            st.error("❌ AI Offline")

        payload = {
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rain_forecast": 1 if rain_forecast else 0,
            "crop_type": CROP_TYPES[crop_name],
        }

        try:
            water_required = predict_irrigation(payload)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        irrigation_state = "ON" if water_required >= irrigation_threshold else "OFF"
        efficiency = calculate_efficiency_score(soil_moisture, water_required)
        runtime = estimate_runtime_minutes(water_required, pump_flow_rate)
        water_per_m2 = round(water_required / max(field_area, 1.0), 4)

        result = {
            "water_required": water_required,
            "irrigation_state": irrigation_state,
            "efficiency": efficiency,
            "runtime": runtime,
            "water_per_m2": water_per_m2,
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rain_forecast": 1 if rain_forecast else 0,
            "crop_name": crop_name,
            "threshold": irrigation_threshold,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.last_irrigation_result = result
        st.session_state.irrigation_history.append(
            {
                "Timestamp": result["timestamp"],
                "Water Required (L)": round(result["water_required"], 2),
                "Irrigation": result["irrigation_state"],
                "Efficiency (%)": round(result["efficiency"], 2),
                "Soil Moisture (%)": result["soil_moisture"],
                "Temperature (C)": result["temperature"],
                "Humidity (%)": result["humidity"],
                "Rain Forecast": "Yes" if result["rain_forecast"] else "No",
            }
        )
        st.session_state.irrigation_history = st.session_state.irrigation_history[-25:]

    result = st.session_state.last_irrigation_result
    statuses = classify_sensor_conditions(soil_moisture, temperature, humidity)

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Current Alerts</div>', unsafe_allow_html=True)
            st.markdown("### Field Risk Signals")
            alerts = build_alert_messages(result, weather_snapshot, statuses)
            if disease_snapshot and not str(disease_snapshot.get("disease_name", "")).startswith("No Disease"):
                alerts.insert(0, f"Disease watch: {disease_snapshot.get('disease_name', 'Unknown')}")
            for item in alerts[:6]:
                st.write(f"- {item}")

            s1, s2, s3 = st.columns(3)
            with s1:
                render_status_card("Soil", statuses["soil"])
            with s2:
                render_status_card("Temperature", statuses["temperature"])
            with s3:
                render_status_card("Humidity", statuses["humidity"])

            temp_progress = int(np.clip(((temperature + 10) / 70) * 100, 0, 100))
            st.caption("Live sensor profile")
            st.progress(int(soil_moisture), text="Soil Moisture")
            st.progress(temp_progress, text="Temperature")
            st.progress(int(humidity), text="Humidity")

            st.markdown("#### Disease Detection Panel")
            latest_disease = st.session_state.last_disease_result
            if latest_disease:
                disease_name = str(latest_disease.get("disease_name", "Unknown"))
                confidence_pct = float(latest_disease.get("confidence", 0.0)) * 100.0
                severity = str(
                    latest_disease.get(
                        "severity",
                        "Healthy" if disease_name.startswith("No Disease") else "High",
                    )
                )
                affected_pct = float(latest_disease.get("diseased_area_percent", 0.0))
                if disease_name.startswith("No Disease"):
                    disease_summary = "Healthy"
                else:
                    disease_summary = disease_name
                st.markdown(
                    f"""
                    <div class="disease-panel">
                      <div class="section-label">AI Diagnosis</div>
                      <div style="font-size:1.3rem; font-weight:800; margin:4px 0 8px 0;">{html.escape(disease_summary)}</div>
                      <div class="small-muted">Confidence Score: {confidence_pct:.2f}%</div>
                      <div class="small-muted">Estimated Affected Area: {affected_pct:.1f}%</div>
                      <div class="small-muted">Severity: {html.escape(severity)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No disease scan yet. Open Disease Scanner to run automated diagnosis.")

        with st.container(border=True):
            st.markdown('<div class="section-label">Decision Metrics</div>', unsafe_allow_html=True)
            st.markdown("### Irrigation Summary")
            if result is None:
                st.info("Predict irrigation to unlock KPI cards and scheduling insights.")
            else:
                row1_col1, row1_col2, row1_col3 = st.columns(3)
                with row1_col1:
                    render_kpi_tile(
                        "Water Required (L)",
                        f"{result['water_required']:.2f}",
                        "Predicted for current cycle",
                        "aqua",
                    )
                with row1_col2:
                    state_tone = "mint" if str(result["irrigation_state"]) == "OFF" else "violet"
                    render_kpi_tile(
                        "Irrigation State",
                        str(result["irrigation_state"]),
                        f"Threshold: {result['threshold']:.1f} L",
                        state_tone,
                    )
                with row1_col3:
                    render_kpi_tile(
                        "Efficiency Score",
                        f"{result['efficiency']:.1f}%",
                        "Model vs expected requirement",
                        "mint",
                    )

                row2_col1, row2_col2 = st.columns(2)
                with row2_col1:
                    render_kpi_tile(
                        "Estimated Runtime",
                        f"{result['runtime']:.1f} min",
                        "Based on pump flow rate",
                        "violet",
                    )
                with row2_col2:
                    render_kpi_tile(
                        "Water Intensity",
                        f"{result['water_per_m2']:.4f} L/m2",
                        "Distribution per field area",
                        "aqua",
                    )

                if result["irrigation_state"] == "ON":
                    st.warning("Action Required: Irrigation should be ON for this cycle.")
                else:
                    st.success("No immediate irrigation action required.")

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Rainfall Prediction</div>', unsafe_allow_html=True)
            st.markdown("### Weekly Water Trend")
            projection_source = result if result else {
                "water_required": 16.0,
                "rain_forecast": 1 if rain_forecast else 0,
                "irrigation_state": "ON",
            }
            projection_df = generate_weekly_projection(
                water_required=float(projection_source["water_required"]),
                rain_forecast=int(projection_source["rain_forecast"]),
                irrigation_on=(projection_source["irrigation_state"] == "ON"),
            )
            fig, ax = plt.subplots(figsize=(7.4, 3.1))
            ax.plot(
                projection_df["Day"],
                projection_df["Projected Water (L)"],
                marker="o",
                linewidth=2.2,
                color="#44c7ef",
            )
            ax.fill_between(
                projection_df["Day"],
                projection_df["Projected Water (L)"],
                alpha=0.18,
                color="#8a78f8",
            )
            ax.set_title("7-Day Irrigation Demand")
            ax.set_xlabel("Day")
            ax.set_ylabel("Liters")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))
            st.pyplot(fig)

    with st.container(border=True):
        st.markdown('<div class="section-label">Crop Analysis</div>', unsafe_allow_html=True)
        st.markdown("### Performance and Sensor Analysis")

        efficiency_view = float(result["efficiency"]) if result else float(np.clip(100 - abs(50 - soil_moisture) * 1.4, 0, 100))
        ring1, ring2, ring3 = st.columns(3)
        with ring1:
            st.pyplot(create_ring_chart(efficiency_view, "Efficiency", "#42c1ea"))
        with ring2:
            st.pyplot(create_ring_chart(float(soil_moisture), "Soil Moisture", "#9a86ff"))
        with ring3:
            st.pyplot(create_ring_chart(float(humidity), "Humidity Level", "#47d7b3"))

        feature_source = result if result else {
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rain_forecast": 1 if rain_forecast else 0,
        }
        factor_df = generate_factor_breakdown(
            soil_moisture=float(feature_source["soil_moisture"]),
            temperature=float(feature_source["temperature"]),
            humidity=float(feature_source["humidity"]),
            rain_forecast=int(feature_source["rain_forecast"]),
        )
        fc1, fc2 = st.columns([1.0, 1.05], gap="large")
        with fc1:
            fig, ax = plt.subplots(figsize=(6.6, 3.4))
            colors = ["#58d8ef" if val >= 0 else "#f1b46a" for val in factor_df["Impact Score"]]
            ax.barh(factor_df["Factor"], factor_df["Impact Score"], color=colors)
            ax.set_title("Irrigation Driver Breakdown")
            ax.set_xlabel("Impact Score")
            ax.axvline(0, color="#335642", linewidth=0.9)
            ax.grid(axis="x", linestyle="--", alpha=0.25)
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))
            st.pyplot(fig)

        with fc2:
            categories = ["Soil", "Temp", "Humidity", "Rain"]
            radar_values = [
                float(feature_source["soil_moisture"]) / 100.0,
                float(np.clip((float(feature_source["temperature"]) + 10.0) / 70.0, 0.0, 1.0)),
                float(feature_source["humidity"]) / 100.0,
                float(feature_source["rain_forecast"]),
            ]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            radar_values += radar_values[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(5.7, 4.2), subplot_kw=dict(polar=True))
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, radar_values, linewidth=2, color="#44c7ef")
            ax.fill(angles, radar_values, color="#9f8cff", alpha=0.28)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.tick_params(axis="x", pad=11)
            ax.set_rlabel_position(20)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)
            ax.set_title("Normalized Sensor Profile", y=1.16, pad=12)
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))
            st.pyplot(fig)

    with st.container(border=True):
        st.markdown('<div class="section-label">Risk Matrix</div>', unsafe_allow_html=True)
        st.markdown("### Crop Health and Threat Overview")

        disease_name = ""
        disease_confidence = 0.0
        disease_area_pct = 0.0
        if disease_snapshot:
            disease_name = str(disease_snapshot.get("disease_name", ""))
            disease_confidence = float(disease_snapshot.get("confidence", 0.0))
            disease_area_pct = float(disease_snapshot.get("diseased_area_percent", 0.0))

        is_healthy = bool(disease_name.startswith("No Disease")) if disease_name else True
        health_status_value = "Healthy" if is_healthy else disease_name
        if disease_name:
            if is_healthy:
                health_subtitle = f"AI confidence: {disease_confidence * 100:.2f}%"
            else:
                health_subtitle = (
                    f"AI confidence: {disease_confidence * 100:.2f}% | Affected: {disease_area_pct:.1f}%"
                )
        else:
            health_subtitle = "No scan yet"
        health_level = "safe" if is_healthy else "danger"

        dryness_risk = max(0.0, 50.0 - soil_moisture) * 1.35
        heat_risk = max(0.0, temperature - 33.0) * 3.0
        low_humidity_risk = max(0.0, 42.0 - humidity) * 1.8
        rain_gap_risk = 4.0 if rain_forecast else 15.0
        moderate_risk = float(np.clip(dryness_risk * 0.45 + heat_risk * 0.35 + low_humidity_risk * 0.2 + rain_gap_risk, 0.0, 100.0))
        high_risk_base = (
            (40.0 if soil_moisture < 25.0 else 0.0)
            + (35.0 if temperature > 38.0 else 0.0)
            + (18.0 if humidity < 30.0 else 0.0)
            + (20.0 if (disease_name and not is_healthy) else 0.0)
            + (min(24.0, disease_area_pct * 0.55) if (disease_name and not is_healthy) else 0.0)
        )
        high_risk = float(np.clip(high_risk_base, 0.0, 100.0))

        r1, r2, r3 = st.columns(3)
        with r1:
            render_risk_card("Crop Health Status", health_status_value, health_subtitle, health_level)
        with r2:
            moderate_level = "warn" if moderate_risk >= 45.0 else "safe"
            render_risk_card("Moderate Risk", f"{moderate_risk:.1f}%", "Soil/temperature/humidity stress", moderate_level)
        with r3:
            high_level = "danger" if high_risk >= 45.0 else "warn"
            render_risk_card("High Risk", f"{high_risk:.1f}%", "Critical combined threat signal", high_level)

    with st.container(border=True):
        st.markdown('<div class="section-label">Command Metrics</div>', unsafe_allow_html=True)
        st.markdown("### Core Operational KPIs")

        if result is not None:
            water_efficiency = float(result["efficiency"])
            irrigation_demand = float(result["water_required"])
            soil_for_health = float(result["soil_moisture"])
            humidity_for_health = float(result["humidity"])
        else:
            synthetic_prediction = max(0.0, (70.0 - soil_moisture) * 0.9 + max(0.0, temperature - 26.0) * 0.7 - (8.0 if rain_forecast else 0.0))
            water_efficiency = calculate_efficiency_score(soil_moisture, synthetic_prediction)
            irrigation_demand = round(synthetic_prediction, 2)
            soil_for_health = soil_moisture
            humidity_for_health = humidity

        crop_health_score = calculate_crop_health_score(
            soil_moisture=soil_for_health,
            humidity=humidity_for_health,
            disease_snapshot=disease_snapshot,
        )

        k1, k2, k3 = st.columns(3)
        with k1:
            render_kpi_tile("Water Efficiency", f"{water_efficiency:.1f}%", "Adaptive irrigation performance", "aqua")
        with k2:
            render_kpi_tile("Crop Health Score", f"{crop_health_score:.1f}", "Sensor + disease confidence fusion", "violet")
        with k3:
            render_kpi_tile("Irrigation Demand", f"{irrigation_demand:.1f} L/day", "Current recommended output", "mint")

    with st.container(border=True):
        st.markdown('<div class="section-label">Field Reports</div>', unsafe_allow_html=True)
        st.markdown("### Schedule and Activity Log")

        if result is None:
            st.info("No report available until you run at least one prediction.")
        else:
            schedule_df = build_irrigation_schedule(
                water_required=float(result["water_required"]),
                rain_forecast=int(result["rain_forecast"]),
                irrigation_on=(result["irrigation_state"] == "ON"),
            )
            h1, h2 = st.columns([0.9, 1.1], gap="large")
            with h1:
                st.markdown("#### Recommended Same-Day Schedule")
                st.dataframe(schedule_df, use_container_width=True, hide_index=True)
                st.caption(f"Last prediction time: {result['timestamp']}")
            with h2:
                history = pd.DataFrame(st.session_state.irrigation_history)
                if history.empty:
                    st.info("Prediction history will appear here.")
                else:
                    history_plot = history.copy()
                    history_plot["Index"] = np.arange(1, len(history_plot) + 1)
                    fig, ax = plt.subplots(figsize=(6.9, 3.0))
                    ax.plot(
                        history_plot["Index"],
                        history_plot["Water Required (L)"],
                        marker="o",
                        linewidth=2,
                        color="#44c7ef",
                    )
                    ax.set_title("Recent Predicted Water Requirement Trend")
                    ax.set_xlabel("Prediction Sequence")
                    ax.set_ylabel("Liters")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor((0, 0, 0, 0))
                    st.pyplot(fig)

            history = pd.DataFrame(st.session_state.irrigation_history)
            if not history.empty:
                st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)


def disease_scanner_page() -> None:
    initialize_session_defaults()
    render_top_shell(
        active_page="Disease Scanner",
        title="Disease Scanner",
        subtitle="Cotton disease inspection and condition tracking",
    )

    left_col, right_col = st.columns([1.05, 0.95], gap="large")
    uploaded_image: Image.Image | None = None

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Image Scan</div>', unsafe_allow_html=True)
            st.markdown("### Upload Cotton Leaf Image")
            file = st.file_uploader("Select image", type=["png", "jpg", "jpeg"], key="cotton_disease_image_input")
            if file is not None:
                try:
                    uploaded_image = Image.open(file).convert("RGB")
                    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not open image: {exc}")
                    uploaded_image = None

            run_scan = st.button("Run Disease Analysis", type="primary", use_container_width=True, key="scan_btn")
            if run_scan:
                if uploaded_image is None:
                    st.error("Upload a valid image before scanning.")
                else:
                    try:
                        analysis = analyze_disease(uploaded_image)
                    except Exception as exc:
                        st.error(f"Disease prediction failed: {exc}")
                    else:
                        disease_name = str(analysis["disease_name"])
                        confidence = float(analysis["confidence"])
                        diseased_area_percent = float(analysis["diseased_area_percent"])
                        severity = str(analysis["severity"])
                        suggestions = [str(item) for item in analysis.get("suggestions", [])]
                        st.session_state.last_disease_result = {
                            "disease_name": disease_name,
                            "confidence": confidence,
                            "diseased_area_percent": diseased_area_percent,
                            "severity": severity,
                            "suggestions": suggestions,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        if disease_name.startswith("No Disease"):
                            st.markdown(
                                f'<div class="scan-result ok">Healthy condition: {html.escape(disease_name)}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="scan-result warn">Detected: {html.escape(disease_name)}</div>',
                                unsafe_allow_html=True,
                            )

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(
                                (
                                    '<div class="confidence-card">'
                                    '<div class="label">Confidence</div>'
                                    f'<div class="value">{confidence * 100:.2f}%</div>'
                                    "</div>"
                                ),
                                unsafe_allow_html=True,
                            )
                        with c2:
                            st.markdown(
                                (
                                    '<div class="confidence-card">'
                                    '<div class="label">Affected Area</div>'
                                    f'<div class="value">{diseased_area_percent:.1f}%</div>'
                                    "</div>"
                                ),
                                unsafe_allow_html=True,
                            )

                        severity_class = severity_css_class(severity)
                        st.markdown(
                            f'<div class="scan-severity {severity_class}">Severity: {html.escape(severity)}</div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(
                            int(np.clip(round(diseased_area_percent), 0, 100)),
                            text=f"Estimated diseased area on leaf/plant: {diseased_area_percent:.1f}%",
                        )

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Model Classes</div>', unsafe_allow_html=True)
            st.markdown("### Cotton Labels Supported")
            for idx, label in enumerate(get_cotton_disease_reference(), start=1):
                st.write(f"{idx}. {label}")

        with st.container(border=True):
            st.markdown('<div class="section-label">Current Alerts</div>', unsafe_allow_html=True)
            st.markdown("### Latest Disease Status")
            last = st.session_state.last_disease_result
            if not last:
                st.info("No scan completed yet.")
            else:
                disease_name = str(last.get("disease_name", "Unknown"))
                confidence = float(last.get("confidence", 0.0))
                diseased_area_percent = float(last.get("diseased_area_percent", 0.0))
                severity = str(
                    last.get(
                        "severity",
                        "Healthy" if disease_name.startswith("No Disease") else "High",
                    )
                )
                suggestions = [str(item) for item in last.get("suggestions", []) if str(item).strip()]

                if disease_name.startswith("No Disease"):
                    st.markdown(
                        f'<div class="scan-result ok">{html.escape(disease_name)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="scan-result warn">{html.escape(disease_name)}</div>',
                        unsafe_allow_html=True,
                    )

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        (
                            '<div class="confidence-card">'
                            '<div class="label">Confidence</div>'
                            f'<div class="value">{confidence * 100:.2f}%</div>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        (
                            '<div class="confidence-card">'
                            '<div class="label">Affected Area</div>'
                            f'<div class="value">{diseased_area_percent:.1f}%</div>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

                severity_class = severity_css_class(severity)
                st.markdown(
                    f'<div class="scan-severity {severity_class}">Severity: {html.escape(severity)}</div>',
                    unsafe_allow_html=True,
                )
                st.progress(
                    int(np.clip(round(diseased_area_percent), 0, 100)),
                    text=f"Estimated diseased area on leaf/plant: {diseased_area_percent:.1f}%",
                )
                render_suggestion_card("Recommended Actions", suggestions)
                st.caption(f"Last scan: {last.get('timestamp', '-')}")


def weather_page() -> None:
    initialize_session_defaults()
    render_top_shell(
        active_page="Weather",
        title="Weather Intelligence",
        subtitle="Live weather sync and irrigation-ready field inputs",
    )

    with st.container(border=True):
        st.markdown('<div class="section-label">Current Weather</div>', unsafe_allow_html=True)
        st.markdown("### OpenWeather Live Feed")
        st.caption("Real-time feed powered by your built-in API key configuration.")
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            st.text_input("City", key="weather_city")
        with c2:
            st.text_input("Country Code", key="weather_country", max_chars=2, help="Optional (e.g., US, IN)")

        if st.button("Fetch Current Weather", type="primary", use_container_width=True, key="fetch_weather_live_button"):
            city = st.session_state.weather_city.strip()
            country = st.session_state.weather_country.strip().upper()
            if not city:
                st.error("Enter a city name.")
            else:
                try:
                    weather = fetch_current_weather(city=city, api_key=OPENWEATHER_API_KEY.strip(), country_code=country)
                except Exception as exc:
                    st.error(f"Could not fetch weather: {exc}")
                else:
                    weather["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.last_weather_result = weather
                    st.session_state.temperature_input = weather["temperature"]
                    st.session_state.humidity_input = weather["humidity"]
                    st.session_state.rain_forecast_input = bool(weather["rain_forecast"])
                    st.success("Weather fetched and synced to Dashboard inputs.")

    weather_snapshot = st.session_state.last_weather_result
    if not weather_snapshot:
        st.info("No weather data loaded yet. Fetch live weather to continue.")
        return

    city_name = str(weather_snapshot.get("city_name", "Unknown"))
    country_name = str(weather_snapshot.get("country", "-")) if weather_snapshot.get("country") else "-"
    condition_text = str(weather_snapshot.get("condition", "N/A"))
    condition_icon = weather_icon(condition_text, int(weather_snapshot.get("rain_forecast", 0)))
    rain_text = "Rain Expected" if weather_snapshot["rain_forecast"] else "No Rain Signal"
    fetched_at = str(weather_snapshot.get("fetched_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    temp_c = float(weather_snapshot["temperature"])
    feels_like_c = float(weather_snapshot["feels_like"])
    humidity = float(weather_snapshot["humidity"])
    wind_speed = float(weather_snapshot["wind_speed"])
    pressure = int(weather_snapshot["pressure"])
    latitude = weather_snapshot.get("latitude")
    longitude = weather_snapshot.get("longitude")

    comfort_index = float(np.clip(100 - abs(temp_c - 24) * 2.0 - abs(humidity - 55) * 0.62, 0.0, 100.0))
    dew_point = float(temp_c - ((100 - humidity) / 5.0))
    visibility_km = float(np.clip(10.5 - max(0.0, humidity - 62) * 0.055 + wind_speed * 0.08, 2.0, 12.0))
    aqi_proxy = float(np.clip(76 - max(0.0, humidity - 70) * 0.35 + wind_speed * 1.9, 20.0, 99.0))
    pressure_trend = "Rising" if pressure >= 1015 else ("Steady" if pressure >= 1008 else "Falling")

    hourly_df = build_hourly_weather_outlook(weather_snapshot)
    cards_html = []
    for _, row in hourly_df.iterrows():
        cards_html.append(
            (
                f'<div class="weather-forecast-card">'
                f'<div class="weather-forecast-time">{html.escape(str(row["Time"]))}</div>'
                f'<div class="weather-forecast-icon">{html.escape(str(row["Icon"]))}</div>'
                f'<div class="weather-forecast-temp">{float(row["Temperature"]):.1f} C</div>'
                f'<div class="weather-forecast-sub">Rain {int(row["RainChance"])}%</div>'
                f'<div class="weather-forecast-sub">Wind {float(row["Wind"]):.1f} m/s</div>'
                f"</div>"
            )
        )

    with st.container(border=True):
        st.markdown('<div class="section-label">Weather Snapshot</div>', unsafe_allow_html=True)
        st.markdown("### Weather Center")
        left, right = st.columns([1.35, 0.95], gap="large")

        with left:
            st.markdown(
                f"""
                <div class="weather-hero">
                  <div class="weather-hero-top">
                    <div>
                      <div class="weather-city">{html.escape(city_name)} ({html.escape(country_name)})</div>
                      <div class="weather-meta">{html.escape(fetched_at)}</div>
                    </div>
                    <div class="weather-icon">{html.escape(condition_icon)}</div>
                  </div>
                  <div class="weather-temp-row">
                    <div class="weather-temp">{temp_c:.1f} C</div>
                    <div class="weather-condition">{html.escape(condition_text)}</div>
                  </div>
                  <div class="weather-chip-row">
                    <span class="weather-chip">Feels Like: {feels_like_c:.1f} C</span>
                    <span class="weather-chip">{html.escape(rain_text)}</span>
                    <span class="weather-chip">Pressure Trend: {html.escape(pressure_trend)}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Hourly Outlook")
            st.markdown(f'<div class="weather-forecast-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)

        with right:
            st.markdown("#### Atmospheric Details")
            st.markdown(
                f"""
                <div class="weather-detail-grid">
                  <div class="weather-detail-card"><div class="label">Humidity</div><div class="value">{humidity:.0f}%</div></div>
                  <div class="weather-detail-card"><div class="label">Wind</div><div class="value">{wind_speed:.1f} m/s</div></div>
                  <div class="weather-detail-card"><div class="label">Pressure</div><div class="value">{pressure} hPa</div></div>
                  <div class="weather-detail-card"><div class="label">Air Quality (Proxy)</div><div class="value">{aqi_proxy:.0f}/100</div></div>
                  <div class="weather-detail-card"><div class="label">Dew Point</div><div class="value">{dew_point:.1f} C</div></div>
                  <div class="weather-detail-card"><div class="label">Visibility</div><div class="value">{visibility_km:.1f} km</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="confidence-card">
                  <div class="label">Comfort Index</div>
                  <div class="value">{comfort_index:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown('<div class="section-label">Forecast Visuals</div>', unsafe_allow_html=True)
        st.markdown("### Local Outlook + Sensor Sync")
        c_left, c_right = st.columns([1.25, 0.75], gap="large")

        with c_left:
            forecast_plot_df = hourly_df.copy()
            forecast_plot_df["Index"] = np.arange(1, len(forecast_plot_df) + 1)

            fig, ax1 = plt.subplots(figsize=(8.6, 3.4))
            ax1.plot(
                forecast_plot_df["Index"],
                forecast_plot_df["Temperature"],
                marker="o",
                linewidth=2.2,
                color="#44c7ef",
                label="Temperature (C)",
            )
            ax1.set_xlabel("Time Slot")
            ax1.set_ylabel("Temp (C)", color="#bfe5fb")
            ax1.tick_params(axis="y", colors="#bfe5fb")
            ax1.set_xticks(forecast_plot_df["Index"])
            ax1.set_xticklabels(forecast_plot_df["Time"])
            ax1.grid(alpha=0.26, linestyle="--")

            ax2 = ax1.twinx()
            ax2.plot(
                forecast_plot_df["Index"],
                forecast_plot_df["RainChance"],
                marker="o",
                linewidth=2.0,
                color="#9b85ff",
                label="Rain Chance (%)",
            )
            ax2.set_ylabel("Rain (%)", color="#dacdff")
            ax2.tick_params(axis="y", colors="#dacdff")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, frameon=False, loc="upper left")

            fig.patch.set_alpha(0.0)
            ax1.set_facecolor((0, 0, 0, 0))
            ax2.set_facecolor((0, 0, 0, 0))
            st.pyplot(fig)

        with c_right:
            if st.button("Apply Latest Weather to Dashboard Inputs", use_container_width=True, key="apply_weather_sync_btn"):
                st.session_state.temperature_input = weather_snapshot["temperature"]
                st.session_state.humidity_input = weather_snapshot["humidity"]
                st.session_state.rain_forecast_input = bool(weather_snapshot["rain_forecast"])
                st.success("Latest weather values applied to Dashboard inputs.")

            fig, ax = plt.subplots(figsize=(5.0, 3.1))
            labels = ["Temp", "Hum", "Wind", "Press/20"]
            values = [temp_c, humidity, wind_speed, pressure / 20.0]
            ax.bar(labels, values, color=["#44c7ef", "#9b85ff", "#47d7b3", "#f3b768"])
            ax.set_title("Current Components")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))
            st.pyplot(fig)

    with st.container(border=True):
        st.markdown('<div class="section-label">Location Map</div>', unsafe_allow_html=True)
        st.markdown("### Weather Geo View")
        if latitude is None or longitude is None:
            st.info("Coordinates are unavailable in the current weather response. Fetch weather again to load map data.")
        else:
            lat_val = float(latitude)
            lon_val = float(longitude)
            map_df = pd.DataFrame({"lat": [lat_val], "lon": [lon_val]})
            map_left, map_right = st.columns([1.25, 0.75], gap="large")
            with map_left:
                st.map(map_df, use_container_width=True)
            with map_right:
                st.markdown(
                    f"""
                    <div class="weather-detail-grid">
                      <div class="weather-detail-card"><div class="label">City</div><div class="value">{html.escape(city_name)}</div></div>
                      <div class="weather-detail-card"><div class="label">Country</div><div class="value">{html.escape(country_name)}</div></div>
                      <div class="weather-detail-card"><div class="label">Latitude</div><div class="value">{lat_val:.5f}</div></div>
                      <div class="weather-detail-card"><div class="label">Longitude</div><div class="value">{lon_val:.5f}</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                maps_link = f"https://www.google.com/maps/search/?api=1&query={lat_val:.6f},{lon_val:.6f}"
                st.markdown(f"[Open in Google Maps]({maps_link})")


def analytics_page() -> None:
    initialize_session_defaults()
    render_top_shell(
        active_page="Analytics",
        title="Field Analytics",
        subtitle="Performance trends, efficiency patterns, and water behavior",
    )

    history = pd.DataFrame(st.session_state.irrigation_history)
    if history.empty:
        periods = 24
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="D")
        day_index = np.arange(1, periods + 1)
        seasonal_wave = 2.8 * np.sin(day_index / 2.4) + 1.4 * np.cos(day_index / 4.1)
        trend = 12.0 + 0.08 * day_index
        noise = np.random.normal(0, 0.85, size=periods)
        water_usage = np.clip(trend + seasonal_wave + noise, 5.0, None)
        efficiency = np.clip(88 - np.abs(water_usage - np.mean(water_usage)) * 3.1, 46, 98)
        analytics_df = pd.DataFrame(
            {
                "Date": dates,
                "Day": day_index,
                "Water Usage (L)": water_usage,
                "Efficiency (%)": efficiency,
            }
        )
    else:
        analytics_df = history.copy()
        analytics_df["Date"] = pd.to_datetime(analytics_df.get("Timestamp"), errors="coerce")
        if analytics_df["Date"].isna().all():
            analytics_df["Date"] = pd.date_range(
                end=pd.Timestamp.today().normalize(),
                periods=len(analytics_df),
                freq="D",
            )
        else:
            analytics_df["Date"] = analytics_df["Date"].fillna(pd.Timestamp.today())

        analytics_df["Water Usage (L)"] = pd.to_numeric(analytics_df["Water Required (L)"], errors="coerce")
        if analytics_df["Water Usage (L)"].isna().any():
            fallback = analytics_df["Water Usage (L)"].median()
            if np.isnan(fallback):
                fallback = 10.0
            analytics_df["Water Usage (L)"] = analytics_df["Water Usage (L)"].fillna(fallback)

        if "Efficiency (%)" in analytics_df.columns:
            analytics_df["Efficiency (%)"] = pd.to_numeric(analytics_df["Efficiency (%)"], errors="coerce")
        else:
            analytics_df["Efficiency (%)"] = np.nan

        if analytics_df["Efficiency (%)"].isna().any():
            synthetic_efficiency = np.clip(
                92 - np.abs(analytics_df["Water Usage (L)"] - analytics_df["Water Usage (L)"].mean()) * 2.8,
                42,
                98,
            )
            analytics_df["Efficiency (%)"] = analytics_df["Efficiency (%)"].fillna(synthetic_efficiency)

        analytics_df = analytics_df.sort_values("Date").reset_index(drop=True)
        analytics_df["Day"] = np.arange(1, len(analytics_df) + 1)

    usage_series = analytics_df["Water Usage (L)"].astype(float)
    analytics_df["Rolling 7D (L)"] = usage_series.rolling(window=7, min_periods=1).mean()
    analytics_df["Daily Change (L)"] = usage_series.diff().fillna(0.0)

    usage_std = float(usage_series.std(ddof=0))
    if usage_std == 0:
        zscore = np.zeros(len(usage_series))
    else:
        zscore = (usage_series - usage_series.mean()) / usage_std
    analytics_df["Anomaly"] = np.abs(zscore) > 1.35

    latest_usage = float(usage_series.iloc[-1])
    previous_usage = float(usage_series.iloc[-2]) if len(usage_series) > 1 else latest_usage
    usage_delta = latest_usage - previous_usage
    rolling_7d = float(analytics_df["Rolling 7D (L)"].iloc[-1])
    avg_efficiency = float(analytics_df["Efficiency (%)"].tail(min(7, len(analytics_df))).mean())
    usage_stability = max(0.0, 100.0 - usage_std * 9.0)
    anomaly_count = int(analytics_df["Anomaly"].sum())
    anomaly_free_rate = max(0.0, 100.0 * (1.0 - anomaly_count / max(len(analytics_df), 1)))

    with st.container(border=True):
        st.markdown('<div class="section-label">Snapshot</div>', unsafe_allow_html=True)
        st.markdown("### Analytics Command Deck")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_kpi_tile(
                "Latest Demand",
                f"{latest_usage:.2f} L",
                f"{usage_delta:+.2f} L vs previous cycle",
                "aqua",
            )
        with s2:
            render_kpi_tile(
                "7-Day Average",
                f"{rolling_7d:.2f} L",
                f"{(latest_usage - rolling_7d):+.2f} L vs current",
                "violet",
            )
        with s3:
            render_kpi_tile(
                "Avg Efficiency",
                f"{avg_efficiency:.1f}%",
                f"Stability score: {usage_stability:.1f}%",
                "mint",
            )
        with s4:
            render_kpi_tile(
                "Detected Anomalies",
                str(anomaly_count),
                f"{anomaly_free_rate:.1f}% anomaly-free",
                "aqua",
            )

    with st.container(border=True):
        st.markdown('<div class="section-label">Usage Trend</div>', unsafe_allow_html=True)
        st.markdown("### Water Consumption Timeline (with rolling signal)")
        fig, ax = plt.subplots(figsize=(9.2, 3.8))
        ax.plot(
            analytics_df["Day"],
            analytics_df["Water Usage (L)"],
            marker="o",
            linewidth=1.9,
            color="#47d7b3",
            label="Actual Usage",
        )
        ax.plot(
            analytics_df["Day"],
            analytics_df["Rolling 7D (L)"],
            linewidth=2.3,
            color="#9b85ff",
            label="7D Rolling Avg",
        )

        anomaly_rows = analytics_df[analytics_df["Anomaly"]]
        if not anomaly_rows.empty:
            ax.scatter(
                anomaly_rows["Day"],
                anomaly_rows["Water Usage (L)"],
                color="#f3b768",
                s=62,
                zorder=5,
                label="Anomaly",
            )

        ax.fill_between(
            analytics_df["Day"],
            analytics_df["Water Usage (L)"],
            alpha=0.10,
            color="#44c7ef",
        )
        ax.set_xlabel("Observation")
        ax.set_ylabel("Liters")
        ax.set_title("Irrigation Demand Signal")
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        st.pyplot(fig)

    split_left, split_right = st.columns([1.0, 1.0], gap="large")

    with split_left:
        with st.container(border=True):
            st.markdown('<div class="section-label">Demand Change</div>', unsafe_allow_html=True)
            st.markdown("### Day-to-Day Water Delta")
            fig, ax = plt.subplots(figsize=(6.9, 3.4))
            change_colors = ["#47d7b3" if v >= 0 else "#f3b768" for v in analytics_df["Daily Change (L)"]]
            ax.bar(analytics_df["Day"], analytics_df["Daily Change (L)"], color=change_colors)
            ax.axhline(0, color="#a9d7ef", linewidth=0.9)
            ax.set_xlabel("Observation")
            ax.set_ylabel("Delta (L)")
            ax.set_title("Demand Volatility by Step")
            ax.grid(axis="y", alpha=0.26, linestyle="--")
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))
            st.pyplot(fig)

    with split_right:
        with st.container(border=True):
            st.markdown('<div class="section-label">Distribution</div>', unsafe_allow_html=True)
            st.markdown("### Water Usage Spread")
            fig, ax = plt.subplots(figsize=(6.9, 3.4))
            ax.hist(usage_series, bins=min(10, max(5, len(usage_series) // 2)), color="#44c7ef", alpha=0.65, edgecolor="#d8f5ff")
            median_usage = float(np.median(usage_series))
            q3_usage = float(np.quantile(usage_series, 0.75))
            ax.axvline(median_usage, color="#9b85ff", linestyle="--", linewidth=1.8, label="Median")
            ax.axvline(q3_usage, color="#f3b768", linestyle="--", linewidth=1.6, label="Q3")
            ax.set_xlabel("Liters")
            ax.set_ylabel("Frequency")
            ax.set_title("Usage Distribution Profile")
            ax.grid(axis="y", alpha=0.23, linestyle="--")
            ax.legend(fontsize=8, frameon=False)
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))
            st.pyplot(fig)

    with st.container(border=True):
        st.markdown('<div class="section-label">Forecast</div>', unsafe_allow_html=True)
        st.markdown("### Next 5-Cycle Demand Outlook")

        horizon = 5
        x_hist = np.arange(1, len(usage_series) + 1)
        x_forecast = np.arange(len(usage_series) + 1, len(usage_series) + horizon + 1)

        if len(usage_series) >= 2:
            slope = float(np.polyfit(x_hist, usage_series.to_numpy(), 1)[0])
        else:
            slope = 0.0
        seasonal_bias = float(usage_series.tail(min(7, len(usage_series))).mean() - usage_series.mean())
        forecast_values = latest_usage + slope * np.arange(1, horizon + 1) + 0.35 * seasonal_bias * np.sin(np.arange(1, horizon + 1) / 2.2)
        forecast_values = np.clip(forecast_values, 0.0, None)
        forecast_lower = np.clip(forecast_values * 0.88, 0.0, None)
        forecast_upper = forecast_values * 1.12

        fig, ax = plt.subplots(figsize=(9.2, 3.8))
        ax.plot(x_hist[-min(12, len(x_hist)):], usage_series.tail(min(12, len(usage_series))), marker="o", linewidth=2.0, color="#47d7b3", label="Recent Actual")
        ax.plot(x_forecast, forecast_values, marker="o", linewidth=2.2, color="#9b85ff", label="Forecast")
        ax.fill_between(x_forecast, forecast_lower, forecast_upper, color="#9b85ff", alpha=0.16, label="Forecast Band")
        ax.set_xlabel("Observation")
        ax.set_ylabel("Liters")
        ax.set_title("Short-Term Irrigation Demand Projection")
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        st.pyplot(fig)

        insights: list[str] = []
        if latest_usage > rolling_7d * 1.15:
            insights.append("Current demand is above the 7-day baseline by more than 15%.")
        if latest_usage < rolling_7d * 0.85:
            insights.append("Current demand is below the 7-day baseline by more than 15%.")
        if avg_efficiency < 70:
            insights.append("Efficiency trend is low; re-check threshold and sensor quality.")
        if anomaly_count > 0:
            insights.append(f"{anomaly_count} anomaly point(s) detected in recent observations.")
        if not insights:
            insights.append("Demand pattern is stable with no critical drift signal.")

        st.markdown("#### AI Insights")
        for insight in insights:
            st.write(f"- {insight}")

    with st.container(border=True):
        st.markdown('<div class="section-label">Performance KPIs</div>', unsafe_allow_html=True)
        st.markdown("### Efficiency and Stability Gauges")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.pyplot(create_ring_chart(avg_efficiency, "Efficiency Score", "#44c7ef"))
        with c2:
            st.pyplot(create_ring_chart(usage_stability, "Stability Score", "#9b85ff"))
        with c3:
            st.pyplot(create_ring_chart(anomaly_free_rate, "Anomaly-Free Rate", "#47d7b3"))

        display_df = analytics_df[["Day", "Water Usage (L)", "Rolling 7D (L)", "Daily Change (L)", "Anomaly"]].copy()
        display_df["Water Usage (L)"] = display_df["Water Usage (L)"].round(2)
        display_df["Rolling 7D (L)"] = display_df["Rolling 7D (L)"].round(2)
        display_df["Daily Change (L)"] = display_df["Daily Change (L)"].round(2)
        st.dataframe(display_df.tail(12).iloc[::-1], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Smart Irrigation & Cotton Disease Prediction", layout="wide")
    inject_custom_css()
    configure_chart_theme()
    initialize_session_defaults()

    st.sidebar.markdown(
        """
        <div class="side-intro">
          <div class="title">Smart Farm Console</div>
          <div class="sub">Control irrigation, disease analysis, weather, and analytics from one command center.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    language_options = list(LANGUAGE_CODES.keys())
    if st.session_state.ui_lang_label not in language_options:
        st.session_state.ui_lang_label = "English"
    st.sidebar.selectbox(
        support_copy("language_label", st.session_state.ui_lang),
        options=language_options,
        key="ui_lang_label",
    )
    st.session_state.ui_lang = LANGUAGE_CODES.get(st.session_state.ui_lang_label, "en")

    nav_items = ["Dashboard", "Disease Scanner", "Weather", "Analytics"]
    nav_icons = {
        "Dashboard": "\U0001F4CA",
        "Disease Scanner": "\U0001FA7A",
        "Weather": "\U0001F326",
        "Analytics": "\U0001F4C8",
    }
    page = st.sidebar.radio(
        "Navigation",
        nav_items,
        format_func=lambda item: f"{nav_icons.get(item, '*')}  {item}",
    )
    if st.sidebar.button(
        support_copy("open_chat", st.session_state.ui_lang),
        use_container_width=True,
        key="open_support_chat_btn",
    ):
        st.session_state.support_popup_open = True

    if page == "Dashboard":
        dashboard_page()
    elif page == "Disease Scanner":
        disease_scanner_page()
    elif page == "Weather":
        weather_page()
    else:
        analytics_page()

    if st.session_state.support_popup_open:
        support_chatbot_dialog(page, st.session_state.ui_lang)


if __name__ == "__main__":
    main()
