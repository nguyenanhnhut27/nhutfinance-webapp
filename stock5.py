import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks

# Set page configuration
st.set_page_config(
    page_title="Stock Portfolio Tracker",
    page_icon="📈",
    layout="wide"
)

# Initialize session state for portfolio data
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=[
        'Stock Symbol', 'Company Name', 'Shares', 'Buy Price', 
        'Current Price', 'Total Investment', 'Current Value', 
        'Unrealized P&L', 'Return %', 'Dividend per Share', 
        'Total Dividends', 'Date Added'
    ])

if 'budget' not in st.session_state:
    st.session_state.budget = 0.0

# Main title and header
st.title("📈 Stock Portfolio Tracker")
st.markdown("---")

# Create tabs for portfolio and charts
tab1, tab2 = st.tabs(["📊 Portfolio Management", "📈 Daily Charts"])

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
        st.session_state.budget = budget
        
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
            stock_symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL").upper()
            company_name = st.text_input("Company Name", placeholder="e.g., Apple Inc.")
            shares = st.number_input("Number of Shares", min_value=0.01, step=0.01)
            buy_price = st.number_input("Buy Price per Share ($)", min_value=0.01, step=0.01)
            current_price = st.number_input("Current Price per Share ($)", min_value=0.01, step=0.01)
            dividend_per_share = st.number_input("Dividend per Share ($)", min_value=0.0, step=0.01, value=0.0)
            
            submitted = st.form_submit_button("Add Stock")
            
            if submitted and stock_symbol and company_name:
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
                    st.success(f"Added {stock_symbol} to portfolio!")
                else:
                    st.error("Insufficient budget for this investment!")

    with col2:
        st.header("📊 Current Portfolio")
        
        if not st.session_state.portfolio.empty:
            # Display portfolio table
            display_df = st.session_state.portfolio.copy()
            
            # Format currency columns
            currency_cols = ['Buy Price', 'Current Price', 'Total Investment', 'Current Value', 'Unrealized P&L', 'Dividend per Share', 'Total Dividends']
            for col in currency_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
            
            display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Edit/Update section
            st.subheader("🔄 Update Stock Prices")
            
            if len(st.session_state.portfolio) > 0:
                selected_stock = st.selectbox(
                    "Select stock to update:",
                    st.session_state.portfolio['Stock Symbol'].tolist()
                )
                
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
                        buy_price = st.session_state.portfolio.loc[idx, 'Buy Price']
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
                        
                        st.success(f"Updated {selected_stock}!")
                        st.rerun()
            
            # Remove stock section
            st.subheader("🗑️ Remove Stock")
            if len(st.session_state.portfolio) > 0:
                col_remove1, col_remove2 = st.columns(2)
                with col_remove1:
                    stock_to_remove = st.selectbox(
                        "Select stock to remove:",
                        st.session_state.portfolio['Stock Symbol'].tolist(),
                        key="remove_select"
                    )
                with col_remove2:
                    if st.button("Remove Stock", type="secondary"):
                        st.session_state.portfolio = st.session_state.portfolio[
                            st.session_state.portfolio['Stock Symbol'] != stock_to_remove
                        ].reset_index(drop=True)
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

    # Export functionality
    if not st.session_state.portfolio.empty:
        st.markdown("---")
        st.header("💾 Export Portfolio")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            csv = st.session_state.portfolio.to_csv(index=False)
            st.download_button(
                label="Download Portfolio as CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        
        with col_export2:
            if st.button("Clear All Data"):
                if st.checkbox("I confirm I want to clear all portfolio data"):
                    st.session_state.portfolio = pd.DataFrame(columns=[
                        'Stock Symbol', 'Company Name', 'Shares', 'Buy Price', 
                        'Current Price', 'Total Investment', 'Current Value', 
                        'Unrealized P&L', 'Return %', 'Dividend per Share', 
                        'Total Dividends', 'Date Added'
                    ])
                    st.session_state.budget = 0.0
                    st.success("Portfolio cleared!")
                    st.rerun()

# TAB 2: Daily Charts with Hourly Detail
with tab2:
    st.header("📈 Daily Stock Charts with Hourly Analysis")
    
    # Chart controls
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    
    with col_input1:
        # Stock symbol input with portfolio selection
        portfolio_symbols = []
        if not st.session_state.portfolio.empty:
            portfolio_symbols = st.session_state.portfolio['Stock Symbol'].tolist()
        
        chart_symbol = st.text_input(
            "Stock Symbol:",
            placeholder="e.g., AAPL, TSLA, MSFT"
        ).upper()
        
        if portfolio_symbols:
            selected_from_portfolio = st.selectbox(
                "Or select from portfolio:",
                [""] + portfolio_symbols
            )
            if selected_from_portfolio:
                chart_symbol = selected_from_portfolio
    
    with col_input2:
        time_period = st.selectbox(
            "Time Period:",
            ["1d", "5d", "1mo", "3mo"],
            index=0,
            help="1d and 5d show intraday hourly data"
        )
    
    with col_input3:
        show_annotations = st.checkbox("Show Price Annotations", value=True)
    
    # Display chart if symbol is entered
    if chart_symbol:
        try:
            with st.spinner(f"Loading {chart_symbol} data..."):
                # Fetch stock data
                stock = yf.Ticker(chart_symbol)
                
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
                        st.metric(
                            f"{chart_symbol}",
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
                        
                        # Filter peaks and valleys for changes greater than $0.50
                        filtered_peaks = []
                        filtered_valleys = []
                        
                        # Filter peaks - compare with surrounding valleys
                        for peak_idx in peaks:
                            peak_price = close_prices[peak_idx]
                            # Find nearby valleys to compare
                            nearby_valleys = [v for v in valleys if abs(v - peak_idx) <= min_distance * 3]
                            if nearby_valleys:
                                min_nearby_valley = min([close_prices[v] for v in nearby_valleys])
                                if peak_price - min_nearby_valley >= 0.05:
                                    filtered_peaks.append(peak_idx)
                            else:
                                # If no nearby valleys, compare with overall minimum
                                if peak_price - np.min(close_prices) >= 0.05:
                                    filtered_peaks.append(peak_idx)
                        
                        # Filter valleys - compare with surrounding peaks
                        for valley_idx in valleys:
                            valley_price = close_prices[valley_idx]
                            # Find nearby peaks to compare
                            nearby_peaks = [p for p in peaks if abs(p - valley_idx) <= min_distance * 3]
                            if nearby_peaks:
                                max_nearby_peak = max([close_prices[p] for p in nearby_peaks])
                                if max_nearby_peak - valley_price >= 0.05:
                                    filtered_valleys.append(valley_idx)
                            else:
                                # If no nearby peaks, compare with overall maximum
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
                    if time_period in ["1d", "5d"]:
                        # For intraday data, format x-axis to show hours
                        fig.update_layout(
                            xaxis=dict(
                                title="Time",
                                tickformat='%H:%M',
                                dtick=3600000 * 1 if time_period == "1d" else 3600000 * 3  # Every hour for 1d, every 3 hours for 5d
                            )
                        )
                        chart_title = f"{chart_symbol} - Intraday Price Chart ({time_period})"
                    else:
                        # For daily data, show dates
                        fig.update_layout(
                            xaxis=dict(
                                title="Date"
                            )
                        )
                        chart_title = f"{chart_symbol} - Daily Price Chart ({time_period})"
                    
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
                    
                    # Display peak and valley summary
                    if show_annotations and len(hist) > 10:
                        col_peaks, col_valleys = st.columns(2)
                        
                        with col_peaks:
                            if 'peaks' in locals() and len(peaks) > 0:
                                st.subheader("🔺 Recent Peaks")
                                peak_data = []
                                for i in sorted(peaks[-5:], reverse=True):  # Last 5 peaks
                                    if time_period in ["1d", "5d"]:
                                        time_str = dates[i].strftime('%H:%M')
                                    else:
                                        time_str = dates[i].strftime('%m/%d/%Y')
                                    peak_data.append({
                                        'Time': time_str,
                                        'Price': f"${close_prices[i]:.2f}"
                                    })
                                if peak_data:
                                    st.dataframe(pd.DataFrame(peak_data), hide_index=True)
                        
                        with col_valleys:
                            if 'valleys' in locals() and len(valleys) > 0:
                                st.subheader("🔻 Recent Valleys")
                                valley_data = []
                                for i in sorted(valleys[-5:], reverse=True):  # Last 5 valleys
                                    if time_period in ["1d", "5d"]:
                                        time_str = dates[i].strftime('%H:%M')
                                    else:
                                        time_str = dates[i].strftime('%m/%d/%Y')
                                    valley_data.append({
                                        'Time': time_str,
                                        'Price': f"${close_prices[i]:.2f}"
                                    })
                                if valley_data:
                                    st.dataframe(pd.DataFrame(valley_data), hide_index=True)
                    
                    # Volume chart for intraday periods
                    if time_period in ["1d", "5d"] and 'Volume' in hist.columns:
                        fig_volume = go.Figure()
                        fig_volume.add_trace(go.Bar(
                            x=hist.index,
                            y=hist['Volume'],
                            name='Volume',
                            marker_color='rgba(100, 149, 237, 0.6)'
                        ))
                        
                        fig_volume.update_layout(
                            title=f"{chart_symbol} - Trading Volume",
                            xaxis_title="Time",
                            yaxis_title="Volume",
                            height=250,
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font_color='black',
                            xaxis=dict(
                                tickformat='%H:%M',
                                dtick=3600000 * 1 if time_period == "1d" else 3600000 * 3,
                                gridcolor='#e0e0e0'
                            ),
                            yaxis=dict(gridcolor='#e0e0e0')
                        )
                        st.plotly_chart(fig_volume, use_container_width=True)
                
                else:
                    st.error(f"No data found for {chart_symbol}. Please check the symbol.")
        
        except Exception as e:
            st.error(f"Error loading {chart_symbol}: {str(e)}")
            st.info("Please check your internet connection and verify the stock symbol is correct.")
    
    else:
        st.info("Enter a stock symbol above to view the daily chart with hourly price analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    📊 Stock Portfolio Tracker with Hourly Analysis | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)