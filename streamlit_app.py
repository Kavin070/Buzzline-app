# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import json
import os
import sys
from typing import List, Tuple, Optional
import threading
from queue import Queue

# Import your game class - adjust path based on your structure
try:
    from buzz_line.backend.game import StreamlitBuzzWireGame
except ImportError:
    # If running from inside buzz_line folder
    try:
        from backend.game import StreamlitBuzzWireGame
    except ImportError:
        # If game.py is in same directory
        from game import StreamlitBuzzWireGame

# -------------------- Streamlit App Configuration -------------------- #
st.set_page_config(
    page_title="Buzz Wire Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- Custom CSS -------------------- #
st.markdown("""
<style>
/* Global Styles */
.main-header {
    text-align: center;
    color: #ffcc00;
    font-size: 3rem;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.stApp {
    background: linear-gradient(135deg, #010101 0%, #575757 100%);
}

/* Player Info */
.player-info {
    background: linear-gradient(45deg, #ffcc00, #ff9900);
    color: #333;
    padding: 15px 20px;
    border-radius: 10px;
    margin: 10px 0;
    font-weight: bold;
    text-align: center;
}

/* Game Stats */
.game-stats {
    background: linear-gradient(135deg, rgba(0,0,0,0.3), rgba(255,255,255,0.1));
    padding: 20px;
    border-radius: 15px;
    margin: 10px 0;
    border-left: 4px solid #ffcc00;
}

/* Leaderboard Styles */
.leaderboard-entry {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 15px;
    margin: 8px 0;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    border-left: 3px solid transparent;
    transition: all 0.3s ease;
}

.leaderboard-entry:hover {
    background: rgba(255,255,255,0.1);
    transform: translateX(5px);
}

.leaderboard-entry.podium {
    background: rgba(255, 204, 0, 0.1);
    border-left-color: #ffcc00;
}

.rank {
    font-size: 1.3rem;
    font-weight: bold;
    width: 50px;
    text-align: center;
}

.player-name-lb {
    flex: 1;
    margin: 0 15px;
    color: #fff;
}

.player-date {
    font-size: 0.8rem;
    color: #ccc;
}

.time-display {
    font-family: 'Courier New', monospace;
    font-weight: bold;
    color: #00ff88;
    background: rgba(0, 255, 136, 0.1);
    padding: 6px 10px;
    border-radius: 5px;
    font-size: 0.9rem;
}

/* Instructions */
.instructions {
    background: rgba(0, 0, 0, 0.3);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #ffcc00;
    margin: 15px 0;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(45deg, #ffcc00, #ff9900);
    color: #333;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(255, 204, 0, 0.4);
}

/* Text Input */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.9);
    color: #333;
    border: none;
    border-radius: 8px;
}

.stTextInput > div > div > input:focus {
    box-shadow: 0 0 10px rgba(255, 204, 0, 0.5);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #999;
    font-size: 14px;
    padding: 20px;
}

/* Video Feed Styles */
.video-container {
    border: 2px solid #ffcc00;
    border-radius: 10px;
    overflow: hidden;
    background: #000;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Main App Function -------------------- #
def main():
    # Initialize game
    if 'buzz_game' not in st.session_state:
        st.session_state.buzz_game = StreamlitBuzzWireGame()
        st.session_state.camera_initialized = False
        st.session_state.last_frame_time = 0

    game = st.session_state.buzz_game

    # Main header
    st.markdown('<h1 class="main-header">🎮 Buzz Wire Game</h1>', unsafe_allow_html=True)

    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])

    # -------------------- Left Panel - Player Setup -------------------- #
    with col1:
        st.markdown("### 👤 Player Setup")
        
        # Player name input
        if not game.can_start_game():
            player_name = st.text_input("Enter your name:", max_chars=20, key="player_name_input")
            if st.button("Set Name", type="primary"):
                if player_name.strip():
                    game.set_player_name(player_name)
                    st.success(f"Welcome {player_name}!")
                    st.rerun()
                else:
                    st.error("Please enter a valid name!")
        else:
            current_name = st.session_state.game_state['current_player_name']
            st.markdown(f'<div class="player-info">Player: {current_name}</div>', unsafe_allow_html=True)
            if st.button("Change Player", type="secondary"):
                game.set_player_name(None)
                st.rerun()

        st.markdown("### 🎮 Game Controls")
        
        # Game controls
        if game.can_start_game():
            col_start, col_reset = st.columns(2)
            with col_start:
                if st.button("🚀 Start", type="primary", disabled=st.session_state.game_state['game_started']):
                    if game.start_game():
                        st.success("Game started!")
                        st.rerun()
            
            with col_reset:
                if st.button("🔄 Reset", type="secondary"):
                    game.reset_game()
                    st.success("Game reset!")
                    st.rerun()
        else:
            st.warning("Enter your name to enable game controls!")

        # Instructions
        st.markdown("""
        <div class="instructions">
        <h4>📋 How to Play:</h4>
        <ul>
        <li>▶ Enter your name to register</li>
        <li>▶ Click 'Start' to begin</li>
        <li>▶ Point your index finger at camera</li>
        <li>▶ Guide finger along white wire</li>
        <li>▶ Don't touch wire boundaries!</li>
        <li>▶ Reach red circle to win</li>
        <li>▶ Beat your best time!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # -------------------- Center Panel - Video Feed -------------------- #
    with col2:
        st.markdown("### 📹 Live Game Feed")
        
        # Initialize camera once
        if not st.session_state.camera_initialized:
            try:
                st.session_state.camera = cv2.VideoCapture(0)
                if not st.session_state.camera.isOpened():
                    st.error("❌ Cannot access camera. Please check camera permissions.")
                    return
                st.session_state.camera_initialized = True
            except Exception as e:
                st.error(f"❌ Camera error: {e}")
                return

        # Create camera feed container
        camera_container = st.container()
        
        with camera_container:
            # Game status display
            game_state = st.session_state.game_state
            
            if game_state['game_over']:
                if game_state['won']:
                    st.success(f"🎉 **Congratulations! You Won!** Time: {game.format_time(game_state['elapsed_time'])}")
                else:
                    st.error("💀 **GAME OVER!** You touched the wire!")
            elif game_state['game_started']:
                st.info(f"🎮 **Game in Progress...** Time: {game.format_time(game_state['elapsed_time'])}")
            elif game.can_start_game():
                st.success("✅ **Ready to Start!** Click the Start button above.")
            else:
                st.warning("⚠️ **Enter your name first** to begin playing.")

            # Process and display video frame
            if st.session_state.camera.isOpened():
                ret, frame = st.session_state.camera.read()
                if ret:
                    processed_frame = game.process_frame(frame)
                    # Fixed the deprecated parameter
                    st.image(processed_frame, channels="BGR", width="stretch")
                else:
                    st.error("Failed to read from camera")

    # -------------------- Right Panel - Leaderboard -------------------- #
    with col3:
        st.markdown("### 🏆 Leaderboard")
        
        # Load and display leaderboard
        leaderboard = game.load_leaderboard()
        
        if not leaderboard:
            st.info("No scores yet. Be the first!")
        else:
            for idx, entry in enumerate(leaderboard[:10]):
                medal = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else f"{idx + 1}."
                time_str = game.format_time(entry['time'])
                date_str = entry['date'][:10]  # Just the date part
                
                podium_class = "podium" if idx < 3 else ""
                st.markdown(f"""
                <div class="leaderboard-entry {podium_class}">
                    <span class="rank">{medal}</span>
                    <div class="player-name-lb">
                        <strong>{entry['name']}</strong><br>
                        <span class="player-date">{date_str}</span>
                    </div>
                    <span class="time-display">{time_str}</span>
                </div>
                """, unsafe_allow_html=True)

        # Current game stats
        if game_state['best_time'] is not None:
            st.markdown("### 📊 Your Stats")
            st.markdown(f"""
            <div class="game-stats">
                <strong>Personal Best:</strong> {game.format_time(game_state['best_time'])}<br>
                <strong>Current Time:</strong> {game.format_time(game_state['elapsed_time'])}
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>A KAGO PRODUCTION</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Continuous refresh during active gameplay
    current_time = time.time()
    # Reduced frame rate to minimize flickering while maintaining responsiveness
    if (current_time - st.session_state.last_frame_time > 0.05):  # 20 FPS
        st.session_state.last_frame_time = current_time
        st.rerun()


# -------------------- App Entry Point -------------------- #
if __name__ == "__main__":
    try:
        # Use the main function for continuous video (not the manual refresh version)
        main()
        
    except KeyboardInterrupt:
        # Clean up camera when app is stopped
        if 'camera' in st.session_state:
            st.session_state.camera.release()
        cv2.destroyAllWindows()
    except Exception as e:
        st.error(f"Application error: {e}")
        # Clean up
        if 'camera' in st.session_state:
            st.session_state.camera.release()
        cv2.destroyAllWindows()