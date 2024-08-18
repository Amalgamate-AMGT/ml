import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import threading
import queue
import smtplib
import ssl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from twilio.rest import Client
import asyncio
import stripe
from dotenv import load_dotenv
from PIL import Image
import json
import streamlit.components.v1 as components
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Float, Date, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Import the technical analysis library
from ta import add_all_ta_features
import ta.momentum  # Import momentum indicators
import ta.trend  # Import trend indicators
import ta.volatility  # Import volatility indicators
import ta.volume  # Import volume indicators
from ta.utils import dropna

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
database_url = os.getenv('DATABASE_URL')

# Initialize the OpenAI client
from openai import OpenAI

client = OpenAI(api_key=openai_api_key)

# Set up SQLAlchemy
Base = declarative_base()
engine = create_engine(database_url)
Session = sessionmaker(bind=engine)

# Define User and Admin models
class User(Base):
    __tablename__ = 'users'

    email = Column(String, primary_key=True)
    password = Column(String)
    role = Column(String)
    prediction_count = Column(Integer)
    last_prediction_date = Column(Date)
    subscription_approved = Column(Boolean, default=False)
    monitored_stocks = Column(ARRAY(String))

class Admin(Base):
    __tablename__ = 'admins'

    id = Column(Integer, primary_key=True, autoincrement=True)
    smtp_email = Column(String)
    smtp_password = Column(String)
    twilio_account_sid = Column(String)
    twilio_auth_token = Column(String)
    twilio_from_phone = Column(String)

# Create tables
Base.metadata.create_all(engine)

# Define prediction limits for each role
ROLE_PREDICTION_LIMITS = {
    "BASIC": 30,
    "HOBBYIST": 60,
    "STARTUP": 90,
    "STANDARD": 270,
    "PROFESSIONAL": 720,
    "ENTERPRISE": 1485,
    "ADMIN": float('inf')  # Unlimited access
}

# Define prices for each role
ROLE_PRICES = {
    "BASIC": 10,
    "HOBBYIST": 29,
    "STARTUP": 79,
    "STANDARD": 299,
    "PROFESSIONAL": 699,
    "ENTERPRISE": 1299
}

# Define stock monitoring limits for each role
ROLE_STOCK_LIMITS = {
    "BASIC": 2,
    "HOBBYIST": 3,
    "STARTUP": 5,
    "STANDARD": 20,
    "PROFESSIONAL": 45,
    "ENTERPRISE": 83,
    "ADMIN": float('inf')  # Unlimited access
}

# Global stop event for monitoring
stop_monitoring_event = threading.Event()

# Function to get AI insights using OpenAI's API
def ai_insights_page():
    st.subheader("AI Insights")

    # Input for stock symbol or any other data for insights
    ticker = st.text_input("Enter Stock Ticker for Insights", "AAPL")
    
    # Additional input for user queries or context
    user_query = st.text_area("Enter a question or context for AI insights")

    # Button to fetch insights
    if st.button("Get AI Insights"):
        # Validate input
        if not ticker.strip():
            st.warning("Please enter a valid stock ticker.")
            return

        # Fetch data for the ticker
        stock_data, error = get_realtime_data(ticker)
        if error:
            st.error(f"Error retrieving data: {error}")
            return

        # Display the last 5 rows of stock data for debugging purposes
        st.write("### Stock Data")
        st.write(stock_data.tail())

        # Calculate indicators
        stock_data = calculate_technical_indicators(stock_data)

        # Check if 'RSI' and 'MACD' are in the columns
        if 'RSI' not in stock_data.columns or 'MACD' not in stock_data.columns:
            st.warning("RSI and MACD indicators are not available in the data.")
            # Optionally handle this case, e.g., skip AI analysis or provide a default response
            return

        # Get the latest market news
        market_news = get_market_news(ticker)

        # Prepare prompt for OpenAI
        prompt = f"""
        You are a financial analyst. Provide a detailed analysis and insights for the stock ticker {ticker} based on the following data:

        ### Stock Data (Last 5 rows):
        {stock_data.tail().to_string(index=False)}

        ### Technical Indicators (Last 5 rows):
        {stock_data[['RSI', 'MACD']].tail().to_string(index=False)}

        ### Market News:
        {market_news}

        User query:
        {user_query}

        Provide actionable insights, potential risks, and any significant observations you can infer from the data and news.
        """

        # Get insights from OpenAI
        ai_response = get_openai_insight(prompt)

        # Display AI insights
        st.markdown(f"### Insights for {ticker}")
        st.write(ai_response)


# Function to calculate technical indicators
def calculate_technical_indicators(data):
    # Ensure no NaN values
    data = dropna(data)

    # Add all technical analysis features
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )

    # Manually calculate RSI and MACD if needed
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_signal'] = ta.trend.MACD(data['Close']).macd_signal()

    # Print column names for debugging
    st.write("### Columns in DataFrame:")
    st.write(data.columns)

    return data


# Function to fetch market news for a given ticker
def get_market_news(ticker):
    try:
        news_data = yf.Ticker(ticker).news
        latest_news = news_data[:5]  # Get the latest 5 news articles
        news_summary = "\n".join(
            [f"- {article['title']}: {article['link']}" for article in latest_news]
        )
        return news_summary
    except Exception as e:
        return f"Error retrieving market news: {e}"

# Updated get_openai_insight function using the new OpenAI API interface
def get_openai_insight(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo",
            max_tokens=500,  # Adjust tokens as needed
            n=1,
            temperature=0.7
        )

        # Access the content of the response
        message = response.choices[0].message.content.strip()
        return message
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to add a new user
def add_user(email, password, role, prediction_count=0, last_prediction_date=None, subscription_approved=False, monitored_stocks=None):
    session = Session()
    try:
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        user = User(email=email, password=hashed_password, role=role, prediction_count=prediction_count,
                    last_prediction_date=last_prediction_date, subscription_approved=subscription_approved,
                    monitored_stocks=monitored_stocks or [])
        session.add(user)
        session.commit()
    except SQLAlchemyError as e:
        st.error(f"An error occurred while adding a new user: {str(e)}")
        session.rollback()
    finally:
        session.close()

# Function to validate login credentials and reset prediction count if a new month has started
def validate_user(email, password):
    session = Session()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user and check_password_hash(user.password, password):
            # Check if a new month has started
            current_date = datetime.now()
            if user.last_prediction_date and current_date.month != user.last_prediction_date.month:
                user.prediction_count = 0
                user.last_prediction_date = current_date
                session.commit()
            if user.subscription_approved:
                return user.role, user.monitored_stocks
            else:
                return "Pending", []
    except SQLAlchemyError as e:
        st.error(f"An error occurred during user validation: {str(e)}")
    finally:
        session.close()
    return None, []

# Function to update user data
def update_user(email, **kwargs):
    session = Session()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user:
            for key, value in kwargs.items():
                setattr(user, key, value)
            session.commit()
    except SQLAlchemyError as e:
        st.error(f"An error occurred while updating user data: {str(e)}")
        session.rollback()
    finally:
        session.close()

# Function to get the user's prediction count and update it
def update_prediction_count(email):
    session = Session()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user:
            user.prediction_count += 1
            user.last_prediction_date = datetime.now()
            session.commit()
            return user.prediction_count
    except SQLAlchemyError as e:
        st.error(f"An error occurred while updating prediction count: {str(e)}")
        session.rollback()
    finally:
        session.close()
    return None

# Function to check if the user has reached their prediction limit
def check_prediction_limit(email, role):
    session = Session()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user and user.prediction_count < ROLE_PREDICTION_LIMITS[role]:
            return True
    except SQLAlchemyError as e:
        st.error(f"An error occurred while checking prediction limit: {str(e)}")
    finally:
        session.close()
    return False

# Function to update admin data
def update_admin_data(smtp_email, smtp_password, twilio_account_sid, twilio_auth_token, twilio_from_phone):
    session = Session()
    try:
        admin = session.query(Admin).first()
        if not admin:
            admin = Admin(smtp_email=smtp_email, smtp_password=smtp_password,
                          twilio_account_sid=twilio_account_sid, twilio_auth_token=twilio_auth_token,
                          twilio_from_phone=twilio_from_phone)
            session.add(admin)
        else:
            admin.smtp_email = smtp_email
            admin.smtp_password = smtp_password
            admin.twilio_account_sid = twilio_account_sid
            admin.twilio_auth_token = twilio_auth_token
            admin.twilio_from_phone = twilio_from_phone
        session.commit()
    except SQLAlchemyError as e:
        st.error(f"An error occurred while updating admin data: {str(e)}")
        session.rollback()
    finally:
        session.close()

# Function to get real-time stock data
def get_realtime_data(stock, interval='1m'):
    try:
        data = yf.download(tickers=stock, period='1d', interval=interval, progress=False)
        if data.empty:
            return None, f"No data found for stock {stock} with interval {interval}."

        return data, None
    except Exception as e:
        return None, f"Error retrieving data for {stock} with interval {interval}: {e}"

# Function to calculate indicators
def calculate_indicators(data, indicators, moving_averages):
    if isinstance(data, pd.DataFrame):
        for ma in moving_averages:
            data[f'MA{ma}'] = data['Close'].rolling(window=ma).mean()
        if 'Bollinger Bands' in indicators:
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['BB_upper'] = data['MA20'] + 2 * data['Close'].rolling(window=20).std()
            data['BB_lower'] = data['MA20'] - 2 * data['Close'].rolling(window=20).std()
    return data

# Function to send email notifications
def send_email(sender_email, password, receiver_email, subject, body):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

# Function to send SMS notifications (using Twilio)
def send_sms_notification(account_sid, auth_token, from_phone, to_phone, message):
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message,
        from_=from_phone,
        to=to_phone
    )
    return message.sid

# Function to monitor stocks
def monitor_stocks(tickers, threshold, interval, admin_data, receiver_email, to_phone, monitoring_queue):
    previous_prices = {ticker: None for ticker in tickers}

    # Map refresh intervals to yfinance-compatible intervals
    interval_map = {
        60: '1m',
        120: '2m',
        300: '5m',
        900: '15m',
        1800: '30m',
        3600: '1h'
    }

    data_interval = interval_map.get(interval, '1m')  # Use default interval if not found

    while not stop_monitoring_event.is_set():
        for ticker in tickers:
            # Use the mapped interval for fetching data
            data, error = get_realtime_data(ticker, data_interval)
            if error:
                monitoring_queue.put(('error', error))
                continue

            latest_close = data['Close'].iloc[-1]
            previous_close = previous_prices[ticker]

            if previous_close:
                percent_change = ((latest_close - previous_close) / previous_close) * 100
                if abs(percent_change) >= threshold:
                    subject = f"Significant Price Change for {ticker}"
                    body = f"The stock {ticker} has moved {percent_change:.2f}%.\n\nLatest price: {latest_close}\nPrevious price: {previous_close}"
                    send_email(admin_data[0], admin_data[1], receiver_email, subject, body)
                    send_sms_notification(admin_data[2], admin_data[3], admin_data[4], to_phone, body)
                    monitoring_queue.put(('success', f"Notification sent for {ticker}: {percent_change:.2f}% change"))

            previous_prices[ticker] = latest_close
            monitoring_data = {ticker: data}
            monitoring_queue.put(('data', monitoring_data))

        stop_monitoring_event.wait(interval)  # This uses interval as a delay in seconds

# Initialize session state
def initialize_session_state():
    if 'monitoring_data' not in st.session_state:
        st.session_state['monitoring_data'] = {}
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'email' not in st.session_state:
        st.session_state['email'] = ""
    if 'password' not in st.session_state:
        st.session_state['password'] = ""
    if 'role' not in st.session_state:
        st.session_state['role'] = ""
    if 'page' not in st.session_state:
        st.session_state['page'] = "Home"
    if 'monitoring_status' not in st.session_state:
        st.session_state['monitoring_status'] = "Idle"
    if 'current_price_data' not in st.session_state:
        st.session_state['current_price_data'] = pd.DataFrame()
    if 'monitoring_thread' not in st.session_state:
        st.session_state['monitoring_thread'] = None
    if 'monitoring_queue' not in st.session_state:
        st.session_state['monitoring_queue'] = queue.Queue()
    if 'admin_data' not in st.session_state:
        st.session_state['admin_data'] = []
    if 'monitored_stocks' not in st.session_state:
        st.session_state['monitored_stocks'] = []
    if 'monitoring_started' not in st.session_state:
        st.session_state['monitoring_started'] = False
    if 'future_predictions' not in st.session_state:
        st.session_state['future_predictions'] = []

# Header
def header(title):
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)

# Footer
def footer():
    st.markdown("<h4 style='text-align: center;'>© 2024 Stock Monitoring and Prediction App</h4>", unsafe_allow_html=True)

# Real-time chart updating
async def update_realtime_chart(ticker, interval):
    current_price_placeholder = st.empty()
    last_update_time = st.empty()

    while True:
        try:
            current_price_data, error = get_realtime_data(ticker, interval)
            if error:
                st.warning(error)
                await asyncio.sleep(15)
                continue

            current_price_data = calculate_indicators(current_price_data, ["Moving Average", "Bollinger Bands"], [10, 20, 50, 100, 200])
            fig = go.Figure()

            fig.add_trace(go.Candlestick(x=current_price_data.index,
                                         open=current_price_data['Open'],
                                         high=current_price_data['High'],
                                         low=current_price_data['Low'],
                                         close=current_price_data['Close'],
                                         name='Candlestick'))

            for ma in [10, 20, 50, 100, 200]:
                if f'MA{ma}' in current_price_data.columns:
                    fig.add_trace(go.Scatter(x=current_price_data.index, y=current_price_data[f'MA{ma}'], mode='lines', name=f'MA{ma}'))

            if 'BB_upper' in current_price_data.columns and 'BB_lower' in current_price_data.columns:
                fig.add_trace(go.Scatter(x=current_price_data.index, y=current_price_data['BB_upper'], mode='lines', name='BB_upper'))
                fig.add_trace(go.Scatter(x=current_price_data.index, y=current_price_data['BB_lower'], mode='lines', name='BB_lower'))

            fig.add_trace(go.Scatter(
                x=[current_price_data.index[-1]],
                y=[current_price_data['Close'].iloc[-1]],
                mode='markers+text',
                marker=dict(color='red', size=12),
                text=[f"{current_price_data['Close'].iloc[-1]:.2f}"],
                textposition='top center',
                name='Latest Price'
            ))

            fig.update_layout(title=f'Real-Time Current Price ({interval})', xaxis_title='Time', yaxis_title='Price', template='plotly_dark', xaxis_rangeslider_visible=False)

            current_price_placeholder.plotly_chart(fig, use_container_width=True)
            last_update_time.text(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await asyncio.sleep(15)
        except Exception as e:
            st.error(f"Error updating chart: {e}")
            break

# Login function
def login(email, password):
    role, monitored_stocks = validate_user(email, password)
    if role and role != "Pending":
        st.session_state.logged_in = True
        st.session_state.email = email
        st.session_state.role = role
        st.session_state.monitored_stocks = monitored_stocks
        st.session_state.page = "Stock Prediction"  # Redirect to Stock Prediction page
        st.success("Login successful")
    elif role == "Pending":
        st.warning("Your subscription is pending approval.")
    else:
        st.error("Invalid credentials")

# Role-based access control
def has_access(required_role):
    role_hierarchy = {
        "BASIC": 1,
        "HOBBYIST": 2,
        "STARTUP": 3,
        "STANDARD": 4,
        "PROFESSIONAL": 5,
        "ENTERPRISE": 6,
        "ADMIN": 7
    }
    return role_hierarchy[st.session_state.role] >= role_hierarchy[required_role]

# Stripe payment processing function
def process_payment(email, role):
    # Define detailed descriptions for each role
    role_descriptions = {
        "BASIC": "Basic Plan: Access to real-time monitoring of up to 2 stocks and 30 predictions per month.",
        "HOBBYIST": "Hobbyist Plan: Enjoy monitoring up to 3 stocks and 60 predictions per month with added features.",
        "STARTUP": "Startup Plan: Monitor up to 5 stocks, access 90 predictions per month, and gain insights into market trends.",
        "STANDARD": "Standard Plan: Comprehensive access with up to 20 stocks and 270 predictions per month, ideal for regular traders.",
        "PROFESSIONAL": "Professional Plan: Tailored for professional traders with monitoring of 45 stocks and 720 predictions monthly.",
        "ENTERPRISE": "Enterprise Plan: Full-scale access with up to 83 stocks and 1485 predictions per month for enterprise-level trading.",
        "ADMIN": "Admin Plan: Unlimited access to all features and tools with administrative privileges."
    }

    # Create a new Stripe checkout session with detailed product descriptions
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price_data': {
                'currency': 'usd',
                'product_data': {
                    'name': role,
                    'description': role_descriptions.get(role, "Standard Subscription Plan"),
                },
                'unit_amount': ROLE_PRICES[role] * 100,
            },
            'quantity': 1,
        }],
        mode='payment',
        success_url='http://localhost:8501/?success=true',
        cancel_url='http://localhost:8501/?canceled=true',
        metadata={
            'email': email,
            'role': role
        }
    )
    return session.url

# Admin approval of subscriptions
def admin_approve_subscriptions():
    st.subheader("Admin: Approve Subscriptions")
    session = Session()
    try:
        pending_subscriptions = session.query(User).filter(User.subscription_approved == False).all()
        if pending_subscriptions:
            for user in pending_subscriptions:
                st.write(f"Email: {user.email}, Role: {user.role}")
                if st.button(f"Approve {user.email}"):
                    user.subscription_approved = True
                    session.commit()
                    st.success(f"Approved subscription for {user.email}")
        else:
            st.write("No pending subscriptions.")
    except SQLAlchemyError as e:
        st.error(f"An error occurred while approving subscriptions: {str(e)}")
    finally:
        session.close()

    st.subheader("Admin: Set Notification Credentials")
    
    # Step 4: Use environment variables for admin credentials
    smtp_email = os.getenv('SMTP_EMAIL')
    smtp_password = os.getenv('SMTP_PASSWORD')
    twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    twilio_from_phone = os.getenv('TWILIO_FROM_PHONE')

    # Display the credentials (for testing purposes)
    st.write(f"SMTP Email: {smtp_email}")
    st.write(f"Twilio Account SID: {twilio_account_sid}")

    if st.button("Save Credentials"):
        update_admin_data(smtp_email, smtp_password, twilio_account_sid, twilio_auth_token, twilio_from_phone)
        st.success("Admin credentials updated successfully!")

# Function to manage users for the admin role
def admin_manage_users():
    st.subheader("Admin: Manage Users")
    session = Session()
    try:
        users = session.query(User).all()
        for user in users:
            st.write(f"Email: {user.email}, Role: {user.role}, Subscription: {'Approved' if user.subscription_approved else 'Pending'}")
            
            # Safely handle role selection
            role_options = list(ROLE_PRICES.keys())
            
            # Check if the current role is valid; if not, set a default role
            current_role = user.role if user.role in role_options else "BASIC"
            
            # Get the index of the current role, ensuring it exists in the list
            try:
                current_role_index = role_options.index(current_role)
            except ValueError:
                current_role_index = 0  # Default to the first role if not found
            
            new_role = st.selectbox(f"Change Role for {user.email}", role_options, index=current_role_index)
            
            if st.button(f"Update Role for {user.email}"):
                user.role = new_role
                session.commit()
                st.success(f"Updated role for {user.email} to {new_role}")
    except SQLAlchemyError as e:
        st.error(f"An error occurred while managing users: {str(e)}")
    finally:
        session.close()

# Function to display subscription pricing
def display_subscription_pricing():
    st.subheader("Subscription Pricing")
    role_data = {
        "Role": ["BASIC", "HOBBYIST", "STARTUP", "STANDARD", "PROFESSIONAL", "ENTERPRISE"],
        "Price (USD/month)": [10, 29, 79, 299, 699, 1299],
        "Max Predictions": [30, 60, 90, 270, 720, 1485],
        "Max Stocks": [2, 3, 5, 20, 45, 83]
    }
    role_df = pd.DataFrame(role_data)
    st.table(role_df)

    selected_role = st.selectbox("Select Role to Purchase", role_df['Role'].tolist())
    if st.button("Select Subscription"):
        payment_url = process_payment(st.session_state.email, selected_role)
        st.markdown(f"[Complete Payment]({payment_url})", unsafe_allow_html=True)

# Feature: Trade Alert Features
def trade_alert_features():
    st.subheader("Trade Alert Features")
    st.write("Receive real-time alerts on significant price movements, trade signals, and market news.")

    # Add UI for User Inputs
    with st.form("trade_alerts_form"):
        tickers_input = st.text_input("Enter Stock Tickers for Monitoring (comma-separated)", "BTC-USD,ETH-USD,AAPL,GOOGL")
        threshold = st.number_input("Threshold for Alerts (%)", min_value=0.1, max_value=100.0, value=2.0)
        alert_frequency = st.number_input("Alert Frequency (minutes)", min_value=1, max_value=1440, value=15)
        notify_email = st.text_input("Notification Email", st.session_state.email)
        notify_sms = st.text_input("Notification Phone Number", "+1234567890")

        submit_button = st.form_submit_button(label='Start Monitoring')

    if submit_button:
        st.session_state.monitored_stocks = [ticker.strip() for ticker in tickers_input.split(',')]
        st.session_state.threshold = threshold
        st.session_state.alert_frequency = alert_frequency
        st.session_state.notify_email = notify_email
        st.session_state.notify_sms = notify_sms

        if not st.session_state.monitoring_started:
            stop_monitoring_event.clear()
            st.session_state.monitoring_started = True
            st.session_state.monitoring_status = "Monitoring started..."

            if st.session_state.monitoring_thread is None or not st.session_state.monitoring_thread.is_alive():
                st.session_state.monitoring_thread = threading.Thread(
                    target=monitor_stocks, args=(
                        st.session_state.monitored_stocks,
                        st.session_state.threshold,
                        st.session_state.alert_frequency * 60,  # Convert minutes to seconds
                        st.session_state['admin_data'],
                        st.session_state.notify_email,
                        st.session_state.notify_sms,
                        st.session_state['monitoring_queue']
                    ))
                st.session_state.monitoring_thread.start()
        
        # Save user preferences
        st.session_state['monitored_stocks'] = st.session_state.monitored_stocks
        st.success("Monitoring started with the selected preferences!")

    # CSS styles for TradingView container
    st.markdown(
        """
        <style>
        .tradingview-widget-container {
            width: 100%;
            height: 500px;  /* Set a fixed height */
            margin: 0 auto;  /* Center the widget */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # TradingView Chart Integration
    if st.session_state.monitoring_started:
        st.markdown("### Live Trading Chart")
        
        # Use TradingView's Advanced Chart
        components.html(
            """
            <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container">
                <div class="tradingview-widget-container__widget"></div>
                <div class="tradingview-widget-copyright">
                    <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                        <span class="blue-text">Track all markets on TradingView</span>
                    </a>
                </div>
                <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
                {
                
                "width": "1200",
                "height": "610",
                "symbol": "BITSTAMP:BTCUSD",
                "interval": "15",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "withdateranges": true,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "watchlist": [
                    "BITSTAMP:BTCUSD",
                    "COINBASE:SOLUSD",
                    "NASDAQ:NVDA",
                    "NASDAQ:TSLA",
                    "BINANCE:ETHUSDT"
                ],
                "details": true,
                "hotlist": true,
                "calendar": false,
                "studies": [
                    "STD;Bollinger_Bands",
                    "STD;SMA",
                    "STD;Stochastic_RSI",
                    "STD;Supertrend"
                ],
                "support_host": "https://www.tradingview.com",
                "overrides": {
                    "plotTooltips": true,
                    "study_OverlayBollingerBands.default.plot.color": "rgba(255, 0, 0, 1)",
                    "study_OverlayBollingerBands.default.plot.width": 2
                }
                }
                </script>
            </div>
            <!-- TradingView Widget END -->
            """,
            height=500,
        )

    # Alert Status Updates
    st.subheader("Alert Status Updates")
    if st.session_state['monitoring_queue']:
        while not st.session_state['monitoring_queue'].empty():
            message_type, message = st.session_state['monitoring_queue'].get_nowait()
            if message_type == 'data':
                st.session_state['monitoring_data'].update(message)
            elif message_type == 'error':
                st.error(message)
            elif message_type == 'success':
                st.success(message)
    else:
        st.info("No alerts at the moment. Monitoring in progress...")

    # Stop Monitoring
    if st.button("Stop Monitoring"):
        stop_monitoring_event.set()
        st.session_state.monitoring_status = "Monitoring stopped."
        st.session_state.monitoring_started = False
        if st.session_state.monitoring_thread is not None:
            st.session_state.monitoring_thread.join()
        st.success("Monitoring has been stopped.")

# Function to predict future prices using the LSTM model
def predict_future_prices(model, scaler, last_100, days_ahead):
    predictions = []
    current_input = np.copy(last_100)

    for _ in range(days_ahead):
        next_day = model.predict(current_input)
        predictions.append(scaler.inverse_transform(next_day)[0][0])
        current_input = np.append(current_input[:, 1:, :], next_day.reshape(1, 1, 1), axis=1)

    return predictions

# Main app
def main():
    st.set_page_config(page_title="Stock Monitoring and Prediction App", layout="wide")
    initialize_session_state()
    header("Stock Monitoring and Prediction App")

    # Top Navigation Menu
    if st.session_state.logged_in:
        menu_options = ["Home", "Dashboard", "Stock Monitoring", "Stock Prediction", "Real-Time Chart", "Pricing", "Trade Alerts", "AI Insights"]
        if st.session_state.role == "ADMIN":
            menu_options.append("Admin")
    else:
        menu_options = ["Home", "Sign Up", "Login"]

    selected_page = st.selectbox("Navigation", menu_options, index=menu_options.index(st.session_state.page))
    st.session_state.page = selected_page

    if st.session_state.page == "Home":
        st.write("""
        ## Welcome to the Ultimate Stock Monitoring and Prediction App!

        ### Transform Your Trading Strategy with Cutting-Edge Technology

        Are you ready to take your stock trading to the next level? Our Stock Monitoring and Prediction App is designed to provide you with real-time insights, advanced predictions, and comprehensive alerts, all within a user-friendly platform. Whether you are a beginner or a professional trader, our app offers the tools and features you need to make informed decisions and stay ahead of the market.

        ### Why Choose Our App?

        **1. Real-Time Monitoring**: Stay updated with the latest stock prices and market trends. Our app provides real-time data to help you monitor your portfolio and react swiftly to market changes.

        **2. Advanced Predictions**: Leverage the power of AI and machine learning to predict future stock prices. Our sophisticated algorithms analyze historical data and market patterns to provide you with accurate predictions.

        **3. Comprehensive Alerts**: Never miss a significant market movement. Set custom alerts to receive notifications via email and SMS for critical price changes and market events.

        **4. User-Friendly Dashboard**: Manage and monitor your portfolio effortlessly. Our intuitive dashboard allows you to track your stocks, view predictions, and analyze market data all in one place.

        **5. Customizable Indicators**: Enhance your charts with a variety of technical indicators such as Moving Averages and Bollinger Bands. Customize your view to get the insights you need.

        **6. Secure and Reliable**: We prioritize your security. Your data is encrypted and securely stored, ensuring your information remains confidential and protected.

        ### Features Tailored to Your Needs

        - **Multiple Subscription Plans**: Choose a plan that suits your trading needs. From Basic to Enterprise, we offer various plans with different features and limits.
        - **Scalable Monitoring**: Depending on your subscription, monitor from 2 up to 83 stocks simultaneously.
        - **Flexible Predictions**: Make up to 180 predictions per month with our Enterprise plan.
        - **Admin Approval**: Ensuring only validated users access advanced features and market insights.

        ### Start Your Journey Today

        Join our community of traders who are transforming their trading strategies with our app. Sign up now and choose the subscription plan that best fits your needs. Empower your trading with data-driven insights and real-time monitoring.
        """)

        st.markdown("###")
        col1, col2, col3 = st.columns(3)

        with col1:
            try:
                image1 = Image.open("./images/stock_market.jpg")
                st.image(image1, caption="Stock Market", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")

        with col2:
            try:
                image2 = Image.open("./images/trading_analysis.jpg")
                st.image(image2, caption="Trading Analysis", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")

        with col3:
            try:
                image3 = Image.open("./images/investment_growth.jpg")
                st.image(image3, caption="Investment Growth", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")

        footer()

    elif st.session_state.page == "Sign Up":
        st.subheader("Sign Up")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        role = st.selectbox("Select Role", [f"{role} - ${price}/mo" for role, price in ROLE_PRICES.items()])
        selected_role = role.split(" - $")[0]
        if st.button("Sign Up"):
            if password == confirm_password:
                payment_url = process_payment(email, selected_role)
                st.markdown(f"[Complete Payment]({payment_url})", unsafe_allow_html=True)
            else:
                st.error("Passwords do not match.")
        footer()

    elif st.session_state.page == "Login":
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(email, password)
        footer()

    elif st.session_state.page == "Dashboard":
        if st.session_state.logged_in:
            st.subheader(f"Welcome, {st.session_state.email}!")

            st.write("### Currently Monitored Stocks")
            monitored_stocks = st.session_state.get('monitored_stocks', [])
            
            if monitored_stocks:
                stock_data_list = []
                for stock in monitored_stocks:
                    if stock.strip():  # Ensure you only display non-empty stock entries
                        stock_data, error = get_realtime_data(stock, '1m')
                        if error:
                            st.error(error)
                        else:
                            stock_info = {
                                'Stock': stock,
                                'Price': stock_data['Close'].iloc[-1],
                                'Open': stock_data['Open'].iloc[-1],
                                'High': stock_data['High'].iloc[-1],
                                'Low': stock_data['Low'].iloc[-1],
                                'Volume': stock_data['Volume'].iloc[-1],
                                'Date': stock_data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                            }
                            stock_data_list.append(stock_info)
                
                # Create a DataFrame from the stock data list
                if stock_data_list:
                    stock_df = pd.DataFrame(stock_data_list)
                    stock_df.sort_values(by='Date', ascending=False, inplace=True)
                    st.table(stock_df)
            else:
                st.write("You are not currently monitoring any stocks.")

            st.write("### Alerts")
            st.write("No new alerts.")

            st.write("### Update Profile")
            new_email = st.text_input("New Email", st.session_state.email)
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Update Profile"):
                if new_password == confirm_new_password:
                    st.session_state.email = new_email
                    st.session_state.password = generate_password_hash(new_password, method='pbkdf2:sha256')
                    st.success("Profile updated successfully!")
                else:
                    st.error("Passwords do not match.")
            
            st.write("### Activity Log")
            st.write("No recent activities.")

            if st.button("Logout"):
                st.session_state.logged_in = False
                st.experimental_rerun()

            footer()
        else:
            st.warning("Please login to access this page.")

    elif st.session_state.page == "Stock Monitoring":
        if st.session_state.logged_in:
            st.subheader("Stock Monitoring")

            if not has_access("STARTUP"):
                st.error("Your current subscription level does not allow access to this feature.")
                footer()
                return

            tickers_input = st.text_input("Enter Stock Tickers for Monitoring (comma-separated)", "BTC-USD,ETH-USD,XRP-USD,AAPL,GOOGL,MSFT")
            monitoring_threshold = st.number_input("Enter Threshold Percentage for Significant Moves", min_value=0.1, max_value=100.0, value=2.0)
            refresh_interval = st.number_input("Enter Refresh Interval in Seconds", min_value=10, max_value=3600, value=60)
            
            receiver_email = st.text_input("Enter the receiver email for notifications")
            to_phone = st.text_input("Enter the receiver's phone number")

            monitor_tickers = [ticker.strip() for ticker in tickers_input.split(',')]
            if len(monitor_tickers) > ROLE_STOCK_LIMITS[st.session_state.role]:
                st.error(f"Your current subscription level allows monitoring of up to {ROLE_STOCK_LIMITS[st.session_state.role]} stocks only.")
                return

            if st.button("Start Monitoring"):
                stop_monitoring_event.clear()
                st.session_state.monitoring_started = True
                st.session_state.monitoring_status = "Monitoring started..."
                if st.session_state.monitoring_thread is None or not st.session_state.monitoring_thread.is_alive():
                    st.session_state.monitoring_thread = threading.Thread(
                        target=monitor_stocks, args=(monitor_tickers, monitoring_threshold, refresh_interval, st.session_state['admin_data'], receiver_email, to_phone, st.session_state['monitoring_queue']))
                    st.session_state.monitoring_thread.start()

                # Update monitored stocks in session state
                st.session_state.monitored_stocks = monitor_tickers

            if st.button("Stop Monitoring") or st.session_state.monitoring_started:
                stop_monitoring_event.set()
                st.session_state.monitoring_status = "Monitoring stopped."
                st.session_state.monitoring_started = False
                if st.session_state.monitoring_thread is not None:
                    st.session_state.monitoring_thread.join()

            st.subheader(st.session_state.monitoring_status)

            placeholder = st.empty()

            while not st.session_state['monitoring_queue'].empty():
                message_type, message = st.session_state['monitoring_queue'].get_nowait()
                if message_type == 'data':
                    st.session_state['monitoring_data'].update(message)
                elif message_type == 'error':
                    st.error(message)
                elif message_type == 'success':
                    st.success(message)

            if 'monitoring_data' in st.session_state and st.session_state['monitoring_data']:
                data = st.session_state['monitoring_data']
                for ticker, df in data.items():
                    if df.empty:
                        st.warning(f"No price data found for {ticker}.")
                        continue
                    st.subheader(f"Stock: {ticker}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
                    fig.update_layout(title=f'{ticker} Close Price', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
                    st.plotly_chart(fig)
            footer()
        else:
            st.warning("Please login to access this page.")

    elif st.session_state.page == "Stock Prediction":
        if st.session_state.logged_in:
            st.subheader("Stock Prediction")

            if not has_access("PROFESSIONAL"):
                st.error("Your current subscription level does not allow access to this feature.")
                footer()
                return

            def create_model():
                model = Sequential([
                    Input(shape=(100, 1)),  # Correctly set input shape
                    LSTM(50, return_sequences=True),
                    Dropout(0.2),
                    LSTM(50, return_sequences=True),
                    Dropout(0.2),
                    LSTM(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
                return model

            @st.cache_resource
            def load_model_weights():
                model = create_model()
                model.load_weights("bitcoin2.weights.h5")
                return model

            model = load_model_weights()

            stock = "BTC-USD"  # Default stock symbol
            end = datetime.now()
            start = datetime(end.year - 15, end.month, end.day)
            stock = st.text_input("Enter the Stock Symbol for Prediction", stock, max_chars=10)

            progress_bar = st.progress(0)

            @st.cache_data
            def load_data(stock, start, end):
                return yf.download(stock, start, end)

            if stock:
                stock_data = load_data(stock, start, end)
                progress_bar.progress(30)

                if not stock_data.empty:
                    st.subheader("Stock Data")
                    st.write(stock_data)

                    splitting_len = int(len(stock_data) * 0.9)
                    x_test = pd.DataFrame(stock_data.Close[splitting_len:])

                    st.subheader('Original Close Price')
                    fig = plt.figure(figsize=(15, 6))
                    plt.plot(stock_data.Close, 'b')
                    plt.title('Original Close Price')
                    st.pyplot(fig)

                    st.subheader("Test Close Price")
                    st.write(x_test)

                    st.subheader('Test Close Price')
                    fig = plt.figure(figsize=(15, 6))
                    plt.plot(x_test, 'b')
                    plt.title('Test Close Price')
                    st.pyplot(fig)

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(stock_data[['Close']].values)

                    x_data = []
                    y_data = []
                    for i in range(100, len(scaled_data)):
                        x_data.append(scaled_data[i - 100:i])
                        y_data.append(scaled_data[i])

                    x_data, y_data = np.array(x_data), np.array(y_data)

                    predictions = model.predict(x_data)
                    inv_pre = scaler.inverse_transform(predictions)
                    inv_y_test = scaler.inverse_transform(y_data)

                    plotting_data = pd.DataFrame(
                        {
                            'original_test_data': inv_y_test[:len(predictions)].reshape(-1),
                            'predictions': inv_pre.reshape(-1)
                        },
                        index=stock_data.index[100:100 + len(predictions)]
                    )
                    st.subheader("Original values vs Predicted values")
                    st.write(plotting_data)

                    st.subheader('Original Close Price vs Predicted Close Price')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=plotting_data.index, y=plotting_data['original_test_data'], mode='lines', name='Original Test Data'))
                    fig.add_trace(go.Scatter(x=plotting_data.index, y=plotting_data['predictions'], mode='lines', name='Predicted Test Data'))
                    fig.update_layout(title='Original vs Predicted Close Price', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
                    st.plotly_chart(fig)

                    progress_bar.progress(70)

                    st.subheader("Future Price Values")

                    last_100 = stock_data[['Close']].tail(100)
                    last_100 = scaler.fit_transform(last_100['Close'].values.reshape(-1, 1)).reshape(1, -1, 1)
                    prev_100 = np.copy(last_100)

                    def predict_future(no_of_days, prev_100):
                        future_predictions = []
                        for _ in range(no_of_days):
                            next_day = model.predict(prev_100)
                            prev_100 = np.append(prev_100[:, 1:, :], next_day.reshape(1, 1, 1), axis=1)
                            future_predictions.append(scaler.inverse_transform(next_day)[0][0])
                        return future_predictions

                    no_of_days = int(st.text_input("Enter the number of days to predict from current date:", "10"))

                    if st.button("Predict"):
                        if check_prediction_limit(st.session_state.email, st.session_state.role):
                            future_results = predict_future(no_of_days, prev_100)
                            update_prediction_count(st.session_state.email)

                            st.subheader("Future Price Predictions")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=list(range(no_of_days)), y=future_results, mode='lines+markers', name='Predicted'))
                            fig.update_layout(
                                title=f'Future Closing Price of {stock}',
                                xaxis_title='Days',
                                yaxis_title='Close Price',
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig)

                            # Store the predictions in the session state for displaying on the TradingView chart
                            st.session_state.future_predictions = future_results

                            progress_bar.progress(100)
                        else:
                            st.error("You have reached your monthly prediction limit.")
                else:
                    st.error("No data found for the specified stock symbol.")
            else:
                st.info("Please enter a valid stock symbol to start.")
            
            footer()
        else:
            st.warning("Please login to access this page.")

    elif st.session_state.page == "Real-Time Chart":
        if st.session_state.logged_in:
            st.subheader("Real-Time Chart")

            stock = st.text_input("Enter the Stock Symbol for Real-Time Chart", "BTC-USD", max_chars=10)
            interval = st.selectbox("Select Chart View Interval", ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])
            chart_type = st.selectbox("Select Chart Type", ["Line", "Candlestick"])
            indicators = st.multiselect("Select Indicators", ["Moving Average", "Bollinger Bands"], default=["Moving Average"])
            moving_averages = st.multiselect("Select Moving Averages", [10, 20, 50, 100, 200], default=[50, 200])

            interval_map = {
                "1m": "1m",
                "2m": "2m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "60m",
                "1d": "1d",
                "5d": "1d",
                "1wk": "1wk",
                "1mo": "1mo",
                "3mo": "3mo"
            }

            period_map = {
                "1m": "1d",
                "2m": "1d",
                "5m": "5d",
                "15m": "1mo",
                "30m": "1mo",
                "1h": "3mo",
                "1d": "1y",
                "5d": "1y",
                "1wk": "2y",
                "1mo": "5y",
                "3mo": "5y"
            }

            if stock:
                current_price_placeholder = st.empty()
                last_update_time = st.empty()

                async def update_realtime_chart():
                    while not stop_monitoring_event.is_set():
                        current_price_data, error = get_realtime_data(stock, interval_map[interval])
                        if error:
                            st.warning(error)
                            await asyncio.sleep(15)
                            continue

                        current_price_data = calculate_indicators(current_price_data, indicators, moving_averages)
                        fig = go.Figure()

                        if chart_type == "Line":
                            fig.add_trace(go.Scatter(x=current_price_data.index, y=current_price_data['Close'], mode='lines', name='Current Price'))
                        elif chart_type == "Candlestick":
                            fig.add_trace(go.Candlestick(x=current_price_data.index, open=current_price_data['Open'], high=current_price_data['High'], low=current_price_data['Low'], close=current_price_data['Close'], name='Candlestick'))

                        for ma in moving_averages:
                            if f'MA{ma}' in current_price_data.columns:
                                fig.add_trace(go.Scatter(x=current_price_data.index, y=current_price_data[f'MA{ma}'], mode='lines', name=f'MA{ma}'))

                        if 'Bollinger Bands' in indicators:
                            fig.add_trace(go.Scatter(x=current_price_data.index, y=current_price_data['BB_upper'], mode='lines', name='BB_upper'))
                            fig.add_trace(go.Scatter(x=current_price_data.index, y=current_price_data['BB_lower'], mode='lines', name='BB_lower'))

                        # Add prediction markers
                        if st.session_state.future_predictions:
                            prediction_dates = [current_price_data.index[-1] + timedelta(days=i+1) for i in range(len(st.session_state.future_predictions))]
                            fig.add_trace(go.Scatter(
                                x=prediction_dates,
                                y=st.session_state.future_predictions,
                                mode='lines+markers',
                                marker=dict(color='black', size=10),
                                name='Predicted Prices'
                            ))

                        fig.add_trace(go.Scatter(
                            x=[current_price_data.index[-1]],
                            y=[current_price_data['Close'].iloc[-1]],
                            mode='markers+text',
                            marker=dict(color='red', size=12),
                            text=[f"{current_price_data['Close'].iloc[-1]:.2f}"],
                            textposition='top center',
                            name='Latest Price'
                        ))

                        fig.update_layout(title=f'Real-Time Current Price ({interval})', xaxis_title='Time', yaxis_title='Price', template='plotly_dark', xaxis_rangeslider_visible=False)

                        current_price_placeholder.plotly_chart(fig, use_container_width=True)
                        last_update_time.text(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        await asyncio.sleep(15)

                if st.button("Start Updating"):
                    asyncio.run(update_realtime_chart())

            footer()
        else:
            st.warning("Please login to access this page.")

    elif st.session_state.page == "Pricing":
        if st.session_state.logged_in:
            display_subscription_pricing()
            footer()
        else:
            st.warning("Please login to access this page.")

    elif st.session_state.page == "Trade Alerts":
        if st.session_state.logged_in:
            trade_alert_features()
            footer()
        else:
            st.warning("Please login to access this page.")

    elif st.session_state.page == "AI Insights":
        if st.session_state.logged_in:
            ai_insights_page()
            footer()
        else:
            st.warning("Please login to access this page.")

    elif st.session_state.page == "Admin":
        if st.session_state.logged_in and st.session_state.role == "ADMIN":
            admin_approve_subscriptions()
            admin_manage_users()
        else:
            st.warning("Access denied. Admins only.")
            st.experimental_rerun()

if __name__ == "__main__":
    main()
