from dotenv import load_dotenv
import os
import requests
import streamlit as st

from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

# =========================
# Load env
# =========================
load_dotenv()

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="City Agent",
    page_icon="🌆",
    layout="centered"
)

# =========================
# Session state init
# =========================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "last_user_prompt" not in st.session_state:
    st.session_state["last_user_prompt"] = ""

# =========================
# API clients
# =========================
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# =========================
# Tools
# =========================
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city in India."""
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Error: OPENWEATHER_API_KEY not found in environment variables."

    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch weather data. {e}"
    except Exception as e:
        return f"Error: Unexpected weather API error. {e}"

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    feels_like = data["main"].get("feels_like")
    humidity = data["main"].get("humidity")
    desc = data["weather"][0]["description"]
    wind_speed = data.get("wind", {}).get("speed")

    parts = [f"Weather in {city}: {desc}, {temp}°C"]

    if feels_like is not None:
        parts.append(f"feels like {feels_like}°C")
    if humidity is not None:
        parts.append(f"humidity {humidity}%")
    if wind_speed is not None:
        parts.append(f"wind {wind_speed} m/s")

    return ", ".join(parts)


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""
    if not tavily_client:
        return "Error: TAVILY_API_KEY not found in environment variables."

    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3
        )
    except Exception as e:
        return f"Error: Failed to fetch news. {e}"

    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"

    news_items = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "No title")
        url = item.get("url", "")
        snippet = item.get("content", "")

        formatted = (
            f"{idx}. {title}\n"
            f"Link: {url}\n"
            f"Summary: {snippet[:180]}..."
        )
        news_items.append(formatted)

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_items)

# =========================
# Agent builder
# =========================
@st.cache_resource
def build_agent():
    llm = ChatMistralAI(model="mistral-small-2506")
    agent = create_agent(
        model=llm,
        tools=[get_weather, get_news],
        system_prompt=(
            "You are a helpful city assistant. "
            "Use tools when needed. "
            "If the user asks for both weather and news, provide both in a clean format. "
            "If a tool returns an error, explain it simply."
        ),
    )
    return agent

agent = build_agent()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("Configuration")
    st.write("Add these variables in your `.env` file:")
    st.code(
        "OPENWEATHER_API_KEY=your_openweather_key\n"
        "TAVILY_API_KEY=your_tavily_key",
        language="bash"
    )

    st.subheader("Examples")
    st.write("- Weather in Delhi")
    st.write("- Latest news in Mumbai")
    st.write("- Tell me weather and news of Jaipur")

    if st.button("Clear chat"):
        st.session_state["chat_history"] = []
        st.session_state["last_user_prompt"] = ""
        st.rerun()

# =========================
# Main UI
# =========================
st.title("🌆 City Agent")
st.caption("Ask for weather and latest city news")

# Render history
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
prompt = st.chat_input("Ask something like: weather and news of Delhi")

if prompt:
    st.session_state["last_user_prompt"] = prompt
    st.session_state["chat_history"].append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                lc_messages = []

                for msg in st.session_state["chat_history"][:-1]:
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        lc_messages.append(AIMessage(content=msg["content"]))

                lc_messages.append(HumanMessage(content=prompt))

                result = agent.invoke({"messages": lc_messages})

                final_response = result["messages"][-1].content

                if isinstance(final_response, list):
                    final_response = "\n".join(
                        str(item) for item in final_response
                    )

                final_response = str(final_response).strip()

            except Exception as e:
                final_response = f"Error: {e}"

        st.markdown(final_response)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": final_response}
        )

st.divider()
st.subheader("Run command")
st.code("streamlit run Agent.py", language="bash")