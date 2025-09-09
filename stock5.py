import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
import sqlite3
import os
from contextlib import contextmanager

# Set page configuration
st.set_page_config(
    page_title="LINH' Stock Portfolio Tracker",
    page_icon="📈",
    layout="wide"
)

# Database setup
DB_FILE = "portfolio.db"

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_FILE)
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize the database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create portfolio table with market column
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_symbol TEXT NOT NULL,
                company_name TEXT NOT NULL,
                market TEXT NOT NULL DEFAULT 'US',
                shares REAL NOT NULL,
                buy_price REAL NOT NULL,
                current_price REAL NOT NULL,
                total_investment REAL NOT NULL,
                current_value REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                return_pct REAL NOT NULL,
                dividend_per_share REAL NOT NULL DEFAULT 0.0,
                total_dividends REAL NOT NULL DEFAULT 0.0,
                date_added TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create settings table for budget
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY,
                budget REAL NOT NULL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add market column if it doesn't exist (for existing databases)
        cursor.execute("PRAGMA table_info(portfolio)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'market' not in columns:
            cursor.execute("ALTER TABLE portfolio ADD COLUMN market TEXT NOT NULL DEFAULT 'US'")
        
        conn.commit()

def save_portfolio_to_db(portfolio_df):
    """Save portfolio DataFrame to database"""
    if portfolio_df.empty:
        return
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM portfolio")
        
        # Insert new data
        for _, row in portfolio_df.iterrows():
            cursor.execute("""
                INSERT INTO portfolio (
                    stock_symbol, company_name, market, shares, buy_price, current_price,
                    total_investment, current_value, unrealized_pnl, return_pct,
                    dividend_per_share, total_dividends, date_added
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['Stock Symbol'], row['Company Name'], row.get('Market', 'US'), row['Shares'],
                row['Buy Price'], row['Current Price'], row['Total Investment'],
                row['Current Value'], row['Unrealized P&L'], row['Return %'],
                row['Dividend per Share'], row['Total Dividends'], row['Date Added']
            ))
        
        conn.commit()

def load_portfolio_from_db():
    """Load portfolio from database"""
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query("""
                SELECT stock_symbol as 'Stock Symbol',
                       company_name as 'Company Name',
                       market as 'Market',
                       shares as 'Shares',
                       buy_price as 'Buy Price',
                       current_price as 'Current Price',
                       total_investment as 'Total Investment',
                       current_value as 'Current Value',
                       unrealized_pnl as 'Unrealized P&L',
                       return_pct as 'Return %',
                       dividend_per_share as 'Dividend per Share',
                       total_dividends as 'Total Dividends',
                       date_added as 'Date Added'
                FROM portfolio
                ORDER BY created_at DESC
            """, conn)
            return df
        except Exception as e:
            st.error(f"Error loading portfolio from database: {e}")
            return pd.DataFrame(columns=[
                'Stock Symbol', 'Company Name', 'Market', 'Shares', 'Buy Price',
                'Current Price', 'Total Investment', 'Current Value',
                'Unrealized P&L', 'Return %', 'Dividend per Share',
                'Total Dividends', 'Date Added'
            ])

def save_budget_to_db(budget):
    """Save budget to database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO settings (id, budget, updated_at)
            VALUES (1, ?, CURRENT_TIMESTAMP)
        """, (budget,))
        conn.commit()

def load_budget_from_db():
    """Load budget from database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT budget FROM settings WHERE id = 1")
        result = cursor.fetchone()
        return result[0] if result else 0.0

def get_ticker_symbol(symbol, market):
    """Get the correct ticker symbol based on market"""
    if market == 'CA':
        # For Canadian stocks, add .TO suffix if not present
        if not symbol.endswith('.TO') and not symbol.endswith('.V'):
            return f"{symbol}.TO"
    return symbol

def update_prices_from_yahoo():
    """Update current prices for all stocks in portfolio from Yahoo Finance"""
    if st.session_state.portfolio.empty:
        return
    
    updated = False
    for idx, row in st.session_state.portfolio.iterrows():
        try:
            ticker_symbol = get_ticker_symbol(row['Stock Symbol'], row.get('Market', 'US'))
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                shares = row['Shares']
                total_investment = row['Total Investment']
                
                # Update calculations
                new_current_value = shares * current_price
                new_unrealized_pnl = new_current_value - total_investment
                new_return_pct = (new_unrealized_pnl / total_investment * 100) if total_investment > 0 else 0
                
                # Update dataframe
                st.session_state.portfolio.loc[idx, 'Current Price'] = current_price
                st.session_state.portfolio.loc[idx, 'Current Value'] = new_current_value
                st.session_state.portfolio.loc[idx, 'Unrealized P&L'] = new_unrealized_pnl
                st.session_state.portfolio.loc[idx, 'Return %'] = new_return_pct
                
                updated = True
        except Exception as e:
            continue
    
    if updated:
        save_portfolio_to_db(st.session_state.portfolio)
        return True
    return False

# Initialize database
init_database()

# Initialize session state with database data
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_portfolio_from_db()

if 'budget' not in st.session_state:
    st.session_state.budget = load_budget_from_db()

# Main title and header
st.title("📈 Stock Portfolio Tracker")
st.markdown("---")

# Add database status indicator
col_header1, col_header2, col_header3, col_header4 = st.columns([2, 1, 1, 1])
with col_header1:
    if os.path.exists(DB_FILE):
        st.success("🟢 Database Connected")
    else:
        st.warning("🟡 Database Initializing...")

with col_header2:
    if st.button("🔄 Update All Prices", help="Fetch latest prices from Yahoo Finance"):
        with st.spinner("Updating prices..."):
            if update_prices_from_yahoo():
                st.success("Prices updated successfully!")
                st.rerun()
            else:
                st.warning("No updates available or error occurred")

with col_header3:
    # Display last update time
    if os.path.exists(DB_FILE):
        mod_time = os.path.getmtime(DB_FILE)
        last_saved = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        st.caption(f"Last saved: {last_saved}")

with col_header4:
    if st.button("💾 Sync to DB", help="Manually sync data to database"):
        save_portfolio_to_db(st.session_state.portfolio)
        save_budget_to_db(st.session_state.budget)
        st.success("Data synced to database!")

# Create tabs for portfolio and charts
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Management", "🇨🇦 Canadian Stocks", "🇺🇸 US Stocks"])

# TAB 1: Portfolio Management
with tab1:
    # Sidebar for budget and portfolio summary
    with st.sidebar:
        st.header("💰 Portfolio Summary")
        
        # Budget input
        budget = st.number_input(
            "Total Budget ($)", 
            min_value=0.0, 
            value=st.session_state.budget,
            step=100.0,
            help="Enter your total investment budget"
        )
        if budget != st.session_state.budget:
            st.session_state.budget = budget
            save_budget_to_db(budget)
        
        # Portfolio metrics
        if not st.session_state.portfolio.empty:
            total_invested = st.session_state.portfolio['Total Investment'].sum()
            total_current_value = st.session_state.portfolio['Current Value'].sum()
            total_pnl = st.session_state.portfolio['Unrealized P&L'].sum()
            total_dividends = st.session_state.portfolio['Total Dividends'].sum()
            overall_return_pct = ((total_current_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
            
            st.metric("Total Invested", f"${total_invested:,.2f}")
            st.metric("Current Value", f"${total_current_value:,.2f}")
            st.metric("Unrealized P&L", f"${total_pnl:,.2f}", f"{overall_return_pct:.2f}%")
            st.metric("Total Dividends", f"${total_dividends:,.2f}")
            st.metric("Remaining Budget", f"${budget - total_invested:,.2f}")
            
            # Portfolio allocation pie chart
            if len(st.session_state.portfolio) > 0:
                fig_pie = px.pie(
                    st.session_state.portfolio, 
                    values='Current Value', 
                    names='Stock Symbol',
                    title="Portfolio Allocation"
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("➕ Add New Stock")
        
        with st.form("add_stock_form"):
            # Market selection
            market = st.selectbox(
                "Market:",
                ["US", "CA"],
                format_func=lambda x: "🇺🇸 US Market" if x == "US" else "🇨🇦 Canadian Market"
            )
            
            stock_symbol = st.text_input(
                "Stock Symbol", 
                placeholder="e.g., AAPL (US) or TD (CA)"
            ).upper()
            
            # Remove .TO if user enters it for Canadian stocks
            if market == 'CA' and stock_symbol.endswith('.TO'):
                stock_symbol = stock_symbol[:-3]
            
            company_name = st.text_input("Company Name", placeholder="e.g., Apple Inc.")
            shares = st.number_input("Number of Shares", min_value=0.01, step=0.01)
            buy_price = st.number_input("Buy Price per Share ($)", min_value=0.01, step=0.01)
            
            # Add option to fetch current price
            fetch_current = st.checkbox("Fetch current price from Yahoo Finance")
            if not fetch_current:
                current_price = st.number_input("Current Price per Share ($)", min_value=0.01, step=0.01)
            else:
                current_price = 0.0
                
            dividend_per_share = st.number_input("Dividend per Share ($)", min_value=0.0, step=0.01, value=0.0)
            
            submitted = st.form_submit_button("Add Stock")
            
            if submitted and stock_symbol and company_name:
                # Fetch current price if requested
                if fetch_current:
                    try:
                        ticker_symbol = get_ticker_symbol(stock_symbol, market)
                        stock = yf.Ticker(ticker_symbol)
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                        else:
                            st.error("Could not fetch current price. Please enter manually.")
                            current_price = buy_price
                    except:
                        st.error("Could not fetch current price. Using buy price.")
                        current_price = buy_price
                
                # Calculate values
                total_investment = shares * buy_price
                current_value = shares * current_price
                unrealized_pnl = current_value - total_investment
                return_pct = (unrealized_pnl / total_investment * 100) if total_investment > 0 else 0
                total_dividends = shares * dividend_per_share
                
                # Check if enough budget
                if total_investment <= (st.session_state.budget - st.session_state.portfolio['Total Investment'].sum()):
                    # Add to portfolio
                    new_row = pd.DataFrame({
                        'Stock Symbol': [stock_symbol],
                        'Company Name': [company_name],
                        'Market': [market],
                        'Shares': [shares],
                        'Buy Price': [buy_price],
                        'Current Price': [current_price],
                        'Total Investment': [total_investment],
                        'Current Value': [current_value],
                        'Unrealized P&L': [unrealized_pnl],
                        'Return %': [return_pct],
                        'Dividend per Share': [dividend_per_share],
                        'Total Dividends': [total_dividends],
                        'Date Added': [datetime.now().strftime("%Y-%m-%d")]
                    })
                    
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    save_portfolio_to_db(st.session_state.portfolio)
                    st.success(f"Added {stock_symbol} ({market}) to portfolio!")
                    st.rerun()
                else:
                    st.error("Insufficient budget for this investment!")

    with col2:
        st.header("📊 Current Portfolio")
        
        if not st.session_state.portfolio.empty:
            # Display portfolio table
            display_df = st.session_state.portfolio.copy()
            
            # Add market flag to symbol display
            display_df['Stock'] = display_df.apply(
                lambda x: f"{'🇨🇦' if x.get('Market', 'US') == 'CA' else '🇺🇸'} {x['Stock Symbol']}", 
                axis=1
            )
            
            # Reorder columns
            cols_order = ['Stock', 'Company Name', 'Shares', 'Buy Price', 'Current Price', 
                         'Total Investment', 'Current Value', 'Unrealized P&L', 'Return %', 
                         'Dividend per Share', 'Total Dividends', 'Date Added']
            display_df = display_df[cols_order]
            
            # Format currency columns
            currency_cols = ['Buy Price', 'Current Price', 'Total Investment', 'Current Value', 
                           'Unrealized P&L', 'Dividend per Share', 'Total Dividends']
            for col in currency_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
            
            display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Edit/Update section
            st.subheader("🔄 Update Stock Prices")
            
            if len(st.session_state.portfolio) > 0:
                # Create stock selection with market indicator
                stock_options = st.session_state.portfolio.apply(
                    lambda x: f"{'🇨🇦' if x.get('Market', 'US') == 'CA' else '🇺🇸'} {x['Stock Symbol']}", 
                    axis=1
                ).tolist()
                selected_option = st.selectbox("Select stock to update:", stock_options)
                
                # Extract the actual symbol from the selection
                selected_stock = selected_option.split(' ', 1)[1] if selected_option else None
                
                col_update1, col_update2, col_update3 = st.columns(3)
                
                with col_update1:
                    new_current_price = st.number_input(
                        f"New current price for {selected_stock}:",
                        min_value=0.01,
                        step=0.01,
                        key="update_price"
                    )
                
                with col_update2:
                    new_dividend = st.number_input(
                        f"New dividend per share for {selected_stock}:",
                        min_value=0.0,
                        step=0.01,
                        key="update_dividend"
                    )
                
                with col_update3:
                    if st.button("Update Stock", type="primary"):
                        idx = st.session_state.portfolio[st.session_state.portfolio['Stock Symbol'] == selected_stock].index[0]
                        shares = st.session_state.portfolio.loc[idx, 'Shares']
                        total_investment = st.session_state.portfolio.loc[idx, 'Total Investment']
                        
                        # Update calculations
                        new_current_value = shares * new_current_price
                        new_unrealized_pnl = new_current_value - total_investment
                        new_return_pct = (new_unrealized_pnl / total_investment * 100) if total_investment > 0 else 0
                        new_total_dividends = shares * new_dividend
                        
                        # Update dataframe
                        st.session_state.portfolio.loc[idx, 'Current Price'] = new_current_price
                        st.session_state.portfolio.loc[idx, 'Current Value'] = new_current_value
                        st.session_state.portfolio.loc[idx, 'Unrealized P&L'] = new_unrealized_pnl
                        st.session_state.portfolio.loc[idx, 'Return %'] = new_return_pct
                        st.session_state.portfolio.loc[idx, 'Dividend per Share'] = new_dividend
                        st.session_state.portfolio.loc[idx, 'Total Dividends'] = new_total_dividends
                        
                        save_portfolio_to_db(st.session_state.portfolio)
                        st.success(f"Updated {selected_stock}!")
                        st.rerun()
            
            # Remove stock section
            st.subheader("🗑️ Remove Stock")
            if len(st.session_state.portfolio) > 0:
                col_remove1, col_remove2 = st.columns(2)
                with col_remove1:
                    remove_options = st.session_state.portfolio.apply(
                        lambda x: f"{'🇨🇦' if x.get('Market', 'US') == 'CA' else '🇺🇸'} {x['Stock Symbol']}", 
                        axis=1
                    ).tolist()
                    remove_selection = st.selectbox("Select stock to remove:", remove_options, key="remove_select")
                    stock_to_remove = remove_selection.split(' ', 1)[1] if remove_selection else None
                    
                with col_remove2:
                    if st.button("Remove Stock", type="secondary"):
                        st.session_state.portfolio = st.session_state.portfolio[
                            st.session_state.portfolio['Stock Symbol'] != stock_to_remove
                        ].reset_index(drop=True)
                        save_portfolio_to_db(st.session_state.portfolio)
                        st.success(f"Removed {stock_to_remove}!")
                        st.rerun()
        else:
            st.info("No stocks in portfolio yet. Add your first stock using the form on the left!")

    # Performance visualization
    if not st.session_state.portfolio.empty:
        st.markdown("---")
        st.header("📈 Performance Analysis")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # P&L bar chart
            fig_pnl = px.bar(
                st.session_state.portfolio,
                x='Stock Symbol',
                y='Unrealized P&L',
                color='Unrealized P&L',
                color_continuous_scale='RdYlGn',
                title="Unrealized P&L by Stock"
            )
            fig_pnl.update_layout(height=400)
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        with col_chart2:
            # Return percentage bar chart
            fig_return = px.bar(
                st.session_state.portfolio,
                x='Stock Symbol',
                y='Return %',
                color='Return %',
                color_continuous_scale='RdYlGn',
                title="Return % by Stock"
            )
            fig_return.update_layout(height=400)
            st.plotly_chart(fig_return, use_container_width=True)

def create_stock_chart(chart_symbol, time_period, show_annotations, market_type):
    """Create stock chart for given symbol and parameters"""
    try:
        with st.spinner(f"Loading {chart_symbol} data..."):
            # Get the correct ticker symbol
            ticker_symbol = get_ticker_symbol(chart_symbol, market_type)
            stock = yf.Ticker(ticker_symbol)
            
            # Get intraday data for short periods, daily for longer periods
            if time_period == "1d":
                hist = stock.history(period="1d", interval="5m")
            elif time_period == "5d":
                hist = stock.history(period="5d", interval="30m")
            else:
                hist = stock.history(period=time_period)
            
            info = stock.info
            
            if not hist.empty:
                # Stock info display
                col_info1, col_info2, col_info3 = st.columns(3)
                
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                
                with col_info1:
                    flag = "🇨🇦" if market_type == "CA" else "🇺🇸"
                    market_name = "TSX" if market_type == "CA" else "NYSE/NASDAQ"
                    st.metric(
                        f"{flag} {chart_symbol} ({market_name})",
                        f"${current_price:.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)"
                    )
                
                with col_info2:
                    high_24h = hist['High'].max()
                    low_24h = hist['Low'].min()
                    st.write(f"**Period High:** ${high_24h:.2f}")
                    st.write(f"**Period Low:** ${low_24h:.2f}")
                
                with col_info3:
                    if 'longName' in info and info['longName']:
                        st.write(f"**Company:** {info['longName'][:30]}...")
                    if 'Volume' in hist.columns:
                        total_volume = hist['Volume'].sum()
                        st.write(f"**Total Volume:** {total_volume:,.0f}")
                
                # Create line chart with peaks and valleys
                fig = go.Figure()
                
                # Main price line
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name=f'{chart_symbol} Price',
                    line=dict(color='#00d4ff', width=2),
                    hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
                ))
                
                # Add peaks and valleys if annotations are enabled
                if show_annotations and len(hist) > 10:
                    close_prices = hist['Close'].values
                    dates = hist.index
                    
                    # Adjust sensitivity based on data type
                    if time_period in ["1d", "5d"]:
                        min_distance = max(3, len(close_prices) // 30)
                        prominence = np.std(close_prices) * 0.4
                    else:
                        min_distance = max(2, len(close_prices) // 15)
                        prominence = np.std(close_prices) * 0.6
                    
                    # Find peaks
                    peaks, _ = find_peaks(
                        close_prices,
                        distance=min_distance,
                        prominence=prominence
                    )
                    
                    # Find valleys
                    valleys, _ = find_peaks(
                        -close_prices,
                        distance=min_distance,
                        prominence=prominence
                    )
                    
                    # Filter peaks and valleys for changes greater than $0.05
                    filtered_peaks = []
                    filtered_valleys = []
                    
                    # Filter peaks
                    for peak_idx in peaks:
                        peak_price = close_prices[peak_idx]
                        nearby_valleys = [v for v in valleys if abs(v - peak_idx) <= min_distance * 3]
                        if nearby_valleys:
                            min_nearby_valley = min([close_prices[v] for v in nearby_valleys])
                            if peak_price - min_nearby_valley >= 0.05:
                                filtered_peaks.append(peak_idx)
                        else:
                            if peak_price - np.min(close_prices) >= 0.05:
                                filtered_peaks.append(peak_idx)
                    
                    # Filter valleys
                    for valley_idx in valleys:
                        valley_price = close_prices[valley_idx]
                        nearby_peaks = [p for p in peaks if abs(p - valley_idx) <= min_distance * 3]
                        if nearby_peaks:
                            max_nearby_peak = max([close_prices[p] for p in nearby_peaks])
                            if max_nearby_peak - valley_price >= 0.05:
                                filtered_valleys.append(valley_idx)
                        else:
                            if np.max(close_prices) - valley_price >= 0.05:
                                filtered_valleys.append(valley_idx)
                    
                    peaks = np.array(filtered_peaks)
                    valleys = np.array(filtered_valleys)
                    
                    # Add peak markers and annotations
                    if len(peaks) > 0:
                        peak_texts = []
                        for i in peaks:
                            if time_period in ["1d", "5d"]:
                                time_label = dates[i].strftime('%H:%M')
                                text = f'${close_prices[i]:.2f}<br>{time_label}'
                            else:
                                date_label = dates[i].strftime('%m/%d')
                                text = f'${close_prices[i]:.2f}<br>{date_label}'
                            peak_texts.append(text)
                        
                        fig.add_trace(go.Scatter(
                            x=[dates[i] for i in peaks],
                            y=[close_prices[i] for i in peaks],
                            mode='markers+text',
                            marker=dict(color='#ff4444', size=12, symbol='triangle-up'),
                            text=peak_texts,
                            textposition="top center",
                            textfont=dict(color='#ff4444', size=10),
                            name='Peaks',
                            hovertemplate='<b>Peak</b><br>%{x}<br>Price: $%{y:.2f}<extra></extra>'
                        ))
                    
                    # Add valley markers and annotations
                    if len(valleys) > 0:
                        valley_texts = []
                        for i in valleys:
                            if time_period in ["1d", "5d"]:
                                time_label = dates[i].strftime('%H:%M')
                                text = f'${close_prices[i]:.2f}<br>{time_label}'
                            else:
                                date_label = dates[i].strftime('%m/%d')
                                text = f'${close_prices[i]:.2f}<br>{date_label}'
                            valley_texts.append(text)
                        
                        fig.add_trace(go.Scatter(
                            x=[dates[i] for i in valleys],
                            y=[close_prices[i] for i in valleys],
                            mode='markers+text',
                            marker=dict(color='#44ff44', size=12, symbol='triangle-down'),
                            text=valley_texts,
                            textposition="bottom center",
                            textfont=dict(color='#44ff44', size=10),
                            name='Valleys',
                            hovertemplate='<b>Valley</b><br>%{x}<br>Price: $%{y:.2f}<extra></extra>'
                        ))
                
                # Configure chart layout
                market_label = "Canadian" if market_type == "CA" else "US"
                if time_period in ["1d", "5d"]:
                    fig.update_layout(
                        xaxis=dict(
                            title="Time",
                            tickformat='%H:%M',
                            dtick=3600000 * 1 if time_period == "1d" else 3600000 * 3
                        )
                    )
                    chart_title = f"{chart_symbol} - {market_label} Intraday Price Chart ({time_period})"
                else:
                    fig.update_layout(
                        xaxis=dict(title="Date")
                    )
                    chart_title = f"{chart_symbol} - {market_label} Daily Price Chart ({time_period})"
                
                fig.update_layout(
                    title=chart_title,
                    yaxis_title="Price ($)",
                    height=600,
                    hovermode='x unified',
                    showlegend=True,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color='black',
                    xaxis=dict(gridcolor='#e0e0e0'),
                    yaxis=dict(gridcolor='#e0e0e0')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add price change bar chart
                st.subheader("📊 Price Changes Over Time")
                
                # Calculate price changes
                price_changes = hist['Close'].diff()
                
                # Create bar chart for price changes
                fig_changes = go.Figure()
                
                # Color based on positive or negative change
                colors = ['green' if x >= 0 else 'red' for x in price_changes]
                
                fig_changes.add_trace(go.Bar(
                    x=hist.index,
                    y=price_changes,
                    name='Price Change',
                    marker_color=colors,
                    hovertemplate='<b>%{x}</b><br>Change: $%{y:.2f}<extra></extra>'
                ))
                
                # Configure layout based on time period
                if time_period in ["1d", "5d"]:
                    interval_text = "5-min intervals" if time_period == "1d" else "30-min intervals"
                    fig_changes.update_layout(
                        title=f"Price Changes ({interval_text})",
                        xaxis=dict(
                            title="Time",
                            tickformat='%H:%M',
                            dtick=3600000 * 1 if time_period == "1d" else 3600000 * 3,
                            gridcolor='#e0e0e0'
                        )
                    )
                else:
                    fig_changes.update_layout(
                        title="Daily Price Changes",
                        xaxis=dict(
                            title="Date",
                            gridcolor='#e0e0e0'
                        )
                    )
                
                fig_changes.update_layout(
                    yaxis_title="Price Change ($)",
                    height=300,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font_color='black',
                    yaxis=dict(gridcolor='#e0e0e0', zeroline=True, zerolinecolor='black', zerolinewidth=1),
                    showlegend=False
                )
                
                st.plotly_chart(fig_changes, use_container_width=True)
                
                # Add statistics
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    avg_change = price_changes.mean()
                    st.metric("Avg Change", f"${avg_change:.2f}")
                
                with col_stat2:
                    max_increase = price_changes.max()
                    st.metric("Max Increase", f"${max_increase:.2f}")
                
                with col_stat3:
                    max_decrease = price_changes.min()
                    st.metric("Max Decrease", f"${max_decrease:.2f}")
                
                with col_stat4:
                    volatility = price_changes.std()
                    st.metric("Volatility (σ)", f"${volatility:.2f}")
                
                return True
            else:
                st.error(f"Could not fetch data for {chart_symbol}. Please check the symbol and try again.")
                return False
    except Exception as e:
        st.error(f"Error fetching data for {chart_symbol}: {str(e)}")
        return False

# TAB 2: Canadian Stocks Charts
with tab2:
    st.header("🇨🇦 Canadian Stock Charts")
    st.image("https://flagcdn.com/w40/ca.png", width=32)
    st.info("For Canadian stocks, enter the base symbol (e.g., TD, RY, CNR). The .TO suffix will be added automatically.")
    
    # Chart controls
    col_ca1, col_ca2, col_ca3 = st.columns([2, 1, 1])
    
    with col_ca1:
        # Get Canadian stocks from portfolio
        ca_portfolio_symbols = []
        if not st.session_state.portfolio.empty:
            ca_stocks = st.session_state.portfolio[st.session_state.portfolio.get('Market', 'US') == 'CA']
            if not ca_stocks.empty:
                ca_portfolio_symbols = ca_stocks['Stock Symbol'].tolist()
        
        ca_chart_symbol = st.text_input(
            "Canadian Stock Symbol:",
            placeholder="e.g., TD, RY, CNR, SHOP",
            key="ca_symbol_input"
        ).upper()
        
        if ca_portfolio_symbols:
            selected_ca_portfolio = st.selectbox(
                "Or select from your Canadian holdings:",
                [""] + ca_portfolio_symbols,
                key="ca_portfolio_select"
            )
            if selected_ca_portfolio:
                ca_chart_symbol = selected_ca_portfolio
    
    with col_ca2:
        ca_time_period = st.selectbox(
            "Time Period:",
            ["1d", "5d", "1mo", "3mo"],
            index=0,
            key="ca_time_period"
        )
    
    with col_ca3:
        ca_show_annotations = st.checkbox("Show Price Annotations", value=True, key="ca_annotations")
    
    # Popular Canadian stocks
    st.markdown("**Popular Canadian Stocks:** TD • RY • CNR • CP • SHOP • BMO • BNS • BCE • T • ENB")
    
    # Display chart if symbol is entered
    if ca_chart_symbol:
        create_stock_chart(ca_chart_symbol, ca_time_period, ca_show_annotations, "CA")
    else:
        st.info("Enter a Canadian stock symbol above to view charts and analysis.")

# TAB 3: US Stocks Charts
with tab3:
    st.header("🇺🇸 US Stock Charts")
    st.image("https://flagcdn.com/w40/us.png", width=32)

    
    # Chart controls
    col_us1, col_us2, col_us3 = st.columns([2, 1, 1])
    
    with col_us1:
        # Get US stocks from portfolio
        us_portfolio_symbols = []
        if not st.session_state.portfolio.empty:
            us_stocks = st.session_state.portfolio[st.session_state.portfolio.get('Market', 'US') == 'US']
            if not us_stocks.empty:
                us_portfolio_symbols = us_stocks['Stock Symbol'].tolist()
        
        us_chart_symbol = st.text_input(
            "US Stock Symbol:",
            placeholder="e.g., AAPL, TSLA, MSFT, GOOGL",
            key="us_symbol_input"
        ).upper()
        
        if us_portfolio_symbols:
            selected_us_portfolio = st.selectbox(
                "Or select from your US holdings:",
                [""] + us_portfolio_symbols,
                key="us_portfolio_select"
            )
            if selected_us_portfolio:
                us_chart_symbol = selected_us_portfolio
    
    with col_us2:
        us_time_period = st.selectbox(
            "Time Period:",
            ["1d", "5d", "1mo", "3mo"],
            index=0,
            key="us_time_period"
        )
    
    with col_us3:
        us_show_annotations = st.checkbox("Show Price Annotations", value=True, key="us_annotations")
    
    # Popular US stocks
    st.markdown("**Popular US Stocks:** AAPL • MSFT • GOOGL • AMZN • TSLA • META • NVDA • JPM • V • JNJ")
    
    # Display chart if symbol is entered
    if us_chart_symbol:
        create_stock_chart(us_chart_symbol, us_time_period, us_show_annotations, "US")
    else:
        st.info("Enter a US stock symbol above to view charts and analysis.")

# Export functionality (moved outside tabs)
if not st.session_state.portfolio.empty:
    st.markdown("---")
    st.header("💾 Export & Data Management")
    
    col_export1, col_export2, col_export3, col_export4 = st.columns(4)
    
    with col_export1:
        csv = st.session_state.portfolio.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    
    with col_export2:
        if st.button("🔄 Reload from DB", help="Reload portfolio data from database"):
            st.session_state.portfolio = load_portfolio_from_db()
            st.session_state.budget = load_budget_from_db()
            st.success("Data reloaded from database!")
            st.rerun()
    
    with col_export3:
        if st.button("💾 Force Save", help="Force save current data to database"):
            save_portfolio_to_db(st.session_state.portfolio)
            save_budget_to_db(st.session_state.budget)
            st.success("Data saved to database!")
    
    with col_export4:
        if st.button("🗑️ Clear All Data"):
            confirm_clear = st.checkbox("⚠️ I confirm I want to clear ALL data", key="confirm_clear")
            if confirm_clear:
                # Clear session state
                st.session_state.portfolio = pd.DataFrame(columns=[
                    'Stock Symbol', 'Company Name', 'Market', 'Shares', 'Buy Price',
                    'Current Price', 'Total Investment', 'Current Value',
                    'Unrealized P&L', 'Return %', 'Dividend per Share',
                    'Total Dividends', 'Date Added'
                ])
                st.session_state.budget = 0.0
                
                # Clear database
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM portfolio")
                    cursor.execute("DELETE FROM settings")
                    conn.commit()
                
                st.success("All data cleared from database!")
                st.rerun()

# Footer with database info
st.markdown("---")
st.markdown("### 🗄️ Database Information")
col_db1, col_db2, col_db3, col_db4 = st.columns(4)

with col_db1:
    if os.path.exists(DB_FILE):
        db_size = os.path.getsize(DB_FILE)
        st.write(f"**Database Size:** {db_size:,} bytes")

with col_db2:
    # Count records in database
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM portfolio")
            portfolio_count = cursor.fetchone()[0]
            st.write(f"**Stocks in DB:** {portfolio_count}")
    except:
        st.write("**Stocks in DB:** 0")

with col_db3:
    # Count by market
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM portfolio WHERE market = 'CA'")
            ca_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM portfolio WHERE market = 'US'")
            us_count = cursor.fetchone()[0]
            st.write(f"**🇨🇦 CA:** {ca_count} | **🇺🇸 US:** {us_count}")
    except:
        st.write("**Markets:** N/A")

with col_db4:
    st.write("**Database Type:** SQLite (Local)")

st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
    📈 Stock Portfolio Tracker with Canadian & US Market Support<br>
    Your data is automatically saved and will persist across sessions!
</div>
""", unsafe_allow_html=True)
