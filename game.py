#game.py
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

# -------------------- Sound Handling -------------------- #
SOUND_AVAILABLE = False
beep_sounds = {}

try:
    import pygame
    try:
        pygame.mixer.init()
        SOUND_AVAILABLE = True

        def create_beep_sound(frequency: int, duration: float, volume: float):
            """Generate a beep sound with specified frequency, duration, and volume"""
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(2 * np.pi * frequency * t)
            wave = (wave * volume * 32767).astype(np.int16)

            stereo_wave = np.zeros((len(wave), 2), dtype=np.int16)
            stereo_wave[:, 0] = wave
            stereo_wave[:, 1] = wave

            return pygame.sndarray.make_sound(stereo_wave)

        # Pre-generate beep sounds
        beep_sounds = {
            'light': create_beep_sound(400, 0.1, 0.3),
            'medium': create_beep_sound(600, 0.15, 0.6),
            'high': create_beep_sound(1000, 0.3, 0.9),
        }

    except Exception as e:
        print(f"Audio disabled (pygame mixer error): {e}")
        SOUND_AVAILABLE = False

except ImportError:
    print("Pygame not available. Install with: pip install pygame")
    SOUND_AVAILABLE = False


def system_beep(intensity='high'):
    """Fallback system beep with different intensities"""
    if intensity == 'light':
        freq, duration = 400, 100
    elif intensity == 'medium':
        freq, duration = 600, 150
    else:  # high
        freq, duration = 1000, 300

    if sys.platform == "win32":
        try:
            import winsound
            winsound.Beep(freq, duration)
        except Exception:
            pass
    else:
        # Linux/macOS fallback
        os.system("echo -e '\\a'")


# -------------------- Streamlit Game Class -------------------- #
class StreamlitBuzzWireGame:
    """
    Buzz Wire Game adapted for Streamlit with session state management
    """

    def __init__(self):
        # Initialize session state if not exists
        if 'game_state' not in st.session_state:
            self.reset_session_state()
        
        # MediaPipe Hands - initialize once
        if 'mp_hands' not in st.session_state:
            st.session_state.mp_hands = mp.solutions.hands
            st.session_state.hands = st.session_state.mp_hands.Hands(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            st.session_state.mp_drawing = mp.solutions.drawing_utils

        # Game parameters
        self.wire_thickness = 4
        self.trace_thickness = 2
        self.tolerance = 8
        self.start_circle_radius = 15
        self.end_circle_radius = 15
        self.smoothing_factor = 0.3
        self.min_movement_threshold = 3
        self.max_trace_gap = 30

        # Sound parameters
        self.beep_cooldown = 0.1
        self.light_threshold = 0.4
        self.medium_threshold = 0.7

        # Leaderboard file
        self.leaderboard_file = "leaderboard.json"

    def reset_session_state(self):
        """Reset all game state variables"""
        st.session_state.game_state = {
            'game_started': False,
            'game_over': False,
            'won': False,
            'wire_points': [],
            'finger_trace': [],
            'start_time': None,
            'elapsed_time': 0.0,
            'best_time': None,
            'last_smooth_position': None,
            'frames_without_finger': 0,
            'finger_detected': False,
            'last_beep_time': 0,
            'last_beep_intensity': None,
            'current_player_name': None
        }

    # -------------------- Player & Leaderboard -------------------- #
    def set_player_name(self, name: str):
        st.session_state.game_state['current_player_name'] = name.strip() if name else None

    def can_start_game(self) -> bool:
        player_name = st.session_state.game_state.get('current_player_name')
        return player_name is not None and len(player_name) > 0

    def load_leaderboard(self) -> List[dict]:
        try:
            if os.path.exists(self.leaderboard_file):
                with open(self.leaderboard_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading leaderboard: {e}")
        return []

    def save_leaderboard(self, leaderboard):
        try:
            with open(self.leaderboard_file, 'w') as f:
                json.dump(leaderboard, f, indent=2)
        except Exception as e:
            print(f"Error saving leaderboard: {e}")

    def add_to_leaderboard(self, name: str, time_score: float):
        leaderboard = self.load_leaderboard()
        score_entry = {
            "name": name,
            "time": time_score,
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        leaderboard.append(score_entry)
        leaderboard.sort(key=lambda x: x['time'])
        leaderboard = leaderboard[:10]
        self.save_leaderboard(leaderboard)
        return leaderboard

    # -------------------- Sound -------------------- #
    def play_beep(self, intensity='high'):
        current_time = time.time()
        cooldown = {'light': 0.2, 'medium': 0.15, 'high': 0.1}.get(intensity, 0.1)
        
        game_state = st.session_state.game_state
        if (current_time - game_state['last_beep_time'] < cooldown and 
            game_state['last_beep_intensity'] == intensity):
            return

        game_state['last_beep_time'] = current_time
        game_state['last_beep_intensity'] = intensity

        if SOUND_AVAILABLE:
            try:
                beep_sounds[intensity].play()
            except Exception:
                system_beep(intensity)
        else:
            system_beep(intensity)

    # -------------------- Core Game Logic -------------------- #
    def get_proximity_to_wire(self, point, wire_points):
        if not wire_points or len(wire_points) < 2:
            return float('inf')
        min_distance = float('inf')
        px, py = point
        for i in range(len(wire_points) - 1):
            distance = self.point_line_distance(px, py, *wire_points[i], *wire_points[i + 1])
            min_distance = min(min_distance, distance)
        return min_distance / self.tolerance if self.tolerance > 0 else float('inf')

    def check_proximity_and_play_sound(self, point, wire_points):
        proximity_ratio = self.get_proximity_to_wire(point, wire_points)
        if proximity_ratio >= 1.0:
            self.play_beep('high')
            return True
        elif proximity_ratio >= self.medium_threshold:
            self.play_beep('medium')
        elif proximity_ratio >= self.light_threshold:
            self.play_beep('light')
        return False

    def start_game(self):
        if not self.can_start_game():
            return False
        
        game_state = st.session_state.game_state
        if not game_state['game_started']:
            game_state.update({
                'game_started': True,
                'game_over': False,
                'won': False,
                'finger_trace': [],
                'start_time': time.time(),
                'elapsed_time': 0.0,
                'last_smooth_position': None,
                'frames_without_finger': 0,
                'finger_detected': False,
                'last_beep_time': 0,
                'last_beep_intensity': None
            })
        return True

    def reset_game(self):
        game_state = st.session_state.game_state
        game_state.update({
            'game_started': False,
            'game_over': False,
            'won': False,
            'finger_trace': [],
            'start_time': None,
            'elapsed_time': 0.0,
            'last_smooth_position': None,
            'frames_without_finger': 0,
            'finger_detected': False,
            'last_beep_time': 0,
            'last_beep_intensity': None,
            'wire_points': []
        })

    # -------------------- Helper Functions -------------------- #
    def generate_random_wire(self, width: int, height: int, num_points: int = 6) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        step = width // (num_points - 1)
        for i in range(num_points):
            x = i * step + 50
            y = random.randint(100, height - 100)
            points.append((x, y))
        return points

    @staticmethod
    def smooth_position(new_pos: Tuple[int, int], last_pos: Optional[Tuple[int, int]], factor: float) -> Tuple[int, int]:
        if last_pos is None:
            return new_pos
        smooth_x = int(last_pos[0] + factor * (new_pos[0] - last_pos[0]))
        smooth_y = int(last_pos[1] + factor * (new_pos[1] - last_pos[1]))
        return (smooth_x, smooth_y)

    @staticmethod
    def should_add_point(new_pos: Tuple[int, int], last_pos: Optional[Tuple[int, int]], threshold: float) -> bool:
        if last_pos is None:
            return True
        dx = new_pos[0] - last_pos[0]
        dy = new_pos[1] - last_pos[1]
        distance = (dx * dx + dy * dy) ** 0.5
        return distance >= threshold

    def draw_wire(self, frame: np.ndarray, points: List[Tuple[int, int]]):
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (255, 255, 255), self.wire_thickness)
        cv2.circle(frame, points[0], self.start_circle_radius, (0, 255, 0), -1)
        cv2.circle(frame, points[-1], self.end_circle_radius, (0, 0, 255), -1)

    @staticmethod
    def point_line_distance(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1.0
        if len_sq != 0:
            param = dot / len_sq
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx, yy = x1 + param * C, y1 + param * D
        dx = px - xx
        dy = py - yy
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:05.2f}"

    def draw_timer(self, frame: np.ndarray, elapsed_time: float, best_time: Optional[float], w: int, h: int):
        timer_text = f"Time: {self.format_time(elapsed_time)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(timer_text, font, font_scale, thickness)
        x = w - text_width - 20
        y = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, timer_text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        if best_time is not None:
            best_text = f"Best: {self.format_time(best_time)}"
            (best_width, best_height), _ = cv2.getTextSize(best_text, font, font_scale - 0.1, thickness - 1)
            best_x = w - best_width - 20
            best_y = y + 35
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (best_x - 10, best_y - best_height - 10), (best_x + best_width + 10, best_y + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, best_text, (best_x, best_y), font, font_scale - 0.1, (0, 255, 255), thickness - 1)

    def process_frame(self, frame):
        """Process a single frame and return the processed frame"""
        game_state = st.session_state.game_state
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Generate wire once
        if not game_state['wire_points']:
            game_state['wire_points'] = self.generate_random_wire(w - 100, h)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = st.session_state.hands.process(rgb)

        # Draw wire
        self.draw_wire(frame, game_state['wire_points'])

        # Update timer
        if game_state['game_started'] and not game_state['game_over'] and game_state['start_time'] is not None:
            game_state['elapsed_time'] = time.time() - game_state['start_time']

        # Draw timer
        self.draw_timer(frame, game_state['elapsed_time'], game_state['best_time'], w, h)

        # Reset detection flag
        game_state['finger_detected'] = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                index_tip = landmarks[8]
                raw_x, raw_y = int(index_tip.x * w), int(index_tip.y * h)

                # Check if index finger is pointing (up and extended)
                index_up = landmarks[8].y < landmarks[6].y
                if index_up and (landmarks[8].y < landmarks[12].y):
                    game_state['finger_detected'] = True
                    game_state['frames_without_finger'] = 0

                    # Smooth the position
                    smooth_pos = self.smooth_position((raw_x, raw_y), game_state['last_smooth_position'], self.smoothing_factor)
                    game_state['last_smooth_position'] = smooth_pos
                    index_x, index_y = smooth_pos

                    if game_state['game_started'] and not game_state['game_over']:
                        # Add point to trace
                        if self.should_add_point((index_x, index_y), 
                                               game_state['finger_trace'][-1] if game_state['finger_trace'] else None, 
                                               self.min_movement_threshold):
                            game_state['finger_trace'].append((index_x, index_y))
                            if len(game_state['finger_trace']) > 1000:
                                game_state['finger_trace'] = game_state['finger_trace'][-1000:]

                        # Check proximity and play appropriate sound
                        collision = self.check_proximity_and_play_sound((index_x, index_y), game_state['wire_points'])
                        
                        if collision:
                            game_state['game_over'] = True
                            game_state['won'] = False

                        # Check win condition (reach end point)
                        end_point = game_state['wire_points'][-1]
                        distance_to_end = ((index_x - end_point[0]) ** 2 + (index_y - end_point[1]) ** 2) ** 0.5
                        if distance_to_end < (self.end_circle_radius + 10):
                            if (game_state['best_time'] is None) or (game_state['elapsed_time'] < game_state['best_time']):
                                game_state['best_time'] = game_state['elapsed_time']
                            
                            # Add to leaderboard
                            if game_state['current_player_name']:
                                self.add_to_leaderboard(game_state['current_player_name'], game_state['elapsed_time'])
                            
                            game_state['game_over'] = True
                            game_state['won'] = True

                    # Draw finger position
                    cv2.circle(frame, (index_x, index_y), 8, (255, 0, 255), -1)

        # Handle finger detection loss
        if not game_state['finger_detected']:
            game_state['frames_without_finger'] += 1
            if game_state['frames_without_finger'] > self.max_trace_gap:
                game_state['last_smooth_position'] = None

        # Draw finger trace
        if game_state['game_started'] and len(game_state['finger_trace']) > 1:
            for i in range(1, len(game_state['finger_trace'])):
                cv2.line(frame, game_state['finger_trace'][i - 1], game_state['finger_trace'][i], (0, 0, 0), self.trace_thickness)

        # Display game messages on frame
        if game_state['game_over']:
            if game_state['won']:
                cv2.putText(frame, "Congratulations! You Won!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, "Score added to leaderboard!", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "GAME OVER!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif not game_state['game_started']:
            if not self.can_start_game():
                cv2.putText(frame, "Enter your name first!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            else:
                cv2.putText(frame, f"Player: {game_state['current_player_name']}", (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Ready to Start!", (50, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                cv2.putText(frame, "Point with index finger only!", (50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame