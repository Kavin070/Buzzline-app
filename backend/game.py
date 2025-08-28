# Updated game.py with name input and leaderboard functionality
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import json
import os
from typing import List, Tuple, Optional

# Option 1: Using pygame for sound (recommended)
try:
    import pygame
    pygame.mixer.init()
    SOUND_AVAILABLE = True
    
    def create_beep_sound(frequency: int, duration: float, volume: float):
        """Generate a beep sound with specified frequency, duration, and volume"""
        import numpy as np
        sample_rate = 44100
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Apply volume
        wave = wave * volume
        
        # Convert to 16-bit integers
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_wave = np.zeros((len(wave), 2), dtype=np.int16)
        stereo_wave[:, 0] = wave
        stereo_wave[:, 1] = wave
        
        # Create pygame sound
        sound = pygame.sndarray.make_sound(stereo_wave)
        return sound
    
    # Create different intensity beep sounds
    beep_sounds = {
        'light': create_beep_sound(400, 0.1, 0.3),    # Low freq, short, quiet
        'medium': create_beep_sound(600, 0.15, 0.6),  # Medium freq, medium, moderate
        'high': create_beep_sound(1000, 0.3, 0.9)     # High freq, long, loud
    }
    
except ImportError:
    SOUND_AVAILABLE = False
    print("Pygame not available. Install with: pip install pygame")

# Option 2: Using system beep (fallback)
import os
import sys

def system_beep(intensity='high'):
    """Fallback system beep with different intensities"""
    if intensity == 'light':
        freq, duration = 400, 100
    elif intensity == 'medium':
        freq, duration = 600, 150
    else:  # high
        freq, duration = 1000, 300
    
    if sys.platform == "win32":
        import winsound
        winsound.Beep(freq, duration)
    elif sys.platform == "darwin":  # macOS
        os.system("afplay /System/Library/Sounds/Beep.aiff")
    else:  # Linux
        os.system(f"beep -f {freq} -l {duration} 2>/dev/null || echo -e '\\a'")


class BuzzWireGame:
    """
    Buzz Wire Game with MediaPipe Hands + OpenCV + Sound Effects + Leaderboard.
    - Enter player name before starting
    - Press 's' on the browser window to start (only if name is entered)
    - Press 'r' on the browser window to restart
    - Leaderboard tracks best scores
    """

    def __init__(self, camera_index: int = 0):
        # Video capture
        self.cap = cv2.VideoCapture(camera_index)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # ---- Tunable Parameters ---- #
        self.wire_thickness = 4
        self.trace_thickness = 2
        self.tolerance = 8
        self.start_circle_radius = 15
        self.end_circle_radius = 15

        # Smoothing parameters
        self.smoothing_factor = 0.3
        self.min_movement_threshold = 3
        self.max_trace_gap = 30

        # Sound parameters
        self.beep_cooldown = 0.1
        self.last_beep_time = 0
        self.last_beep_intensity = None
        
        # Proximity thresholds for different beep intensities
        self.light_threshold = 0.4
        self.medium_threshold = 0.7

        # ---- Game State ---- #
        self.game_started = False
        self.game_over = False
        self.won = False
        self.wire_points: List[Tuple[int, int]] = []
        self.finger_trace: List[Tuple[int, int]] = []
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        self.best_time: Optional[float] = None
        self.last_smooth_position: Optional[Tuple[int, int]] = None
        self.frames_without_finger: int = 0
        self.finger_detected: bool = False

        # ---- Player and Leaderboard ---- #
        self.current_player_name: Optional[str] = None
        self.leaderboard_file = "leaderboard.json"
        self.leaderboard = self.load_leaderboard()

    def set_player_name(self, name: str):
        """Set the current player name"""
        self.current_player_name = name.strip() if name else None

    def can_start_game(self) -> bool:
        """Check if game can start (name must be entered)"""
        return self.current_player_name is not None and len(self.current_player_name) > 0

    def load_leaderboard(self) -> List[dict]:
        """Load leaderboard from JSON file"""
        try:
            if os.path.exists(self.leaderboard_file):
                with open(self.leaderboard_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading leaderboard: {e}")
        return []

    def save_leaderboard(self):
        """Save leaderboard to JSON file"""
        try:
            with open(self.leaderboard_file, 'w') as f:
                json.dump(self.leaderboard, f, indent=2)
        except Exception as e:
            print(f"Error saving leaderboard: {e}")

    def add_to_leaderboard(self, name: str, time_score: float):
        """Add a new score to leaderboard and sort by time"""
        score_entry = {
            "name": name,
            "time": time_score,
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.leaderboard.append(score_entry)
        # Sort by time (ascending - faster times first)
        self.leaderboard.sort(key=lambda x: x['time'])
        
        # Keep only top 10 scores
        self.leaderboard = self.leaderboard[:10]
        
        self.save_leaderboard()

    def get_leaderboard(self) -> List[dict]:
        """Get current leaderboard"""
        return self.leaderboard

    def play_beep(self, intensity='high'):
        """Play beep sound with specified intensity and cooldown"""
        current_time = time.time()
        
        # Adjust cooldown based on intensity
        cooldown = {
            'light': 0.2,
            'medium': 0.15,
            'high': 0.1
        }.get(intensity, 0.1)
        
        # Check cooldown and avoid repeating same intensity too quickly
        if (current_time - self.last_beep_time < cooldown and 
            self.last_beep_intensity == intensity):
            return
            
        self.last_beep_time = current_time
        self.last_beep_intensity = intensity
        
        if SOUND_AVAILABLE:
            try:
                beep_sounds[intensity].play()
            except Exception as e:
                print(f"Error playing {intensity} sound: {e}")
                system_beep(intensity)
        else:
            system_beep(intensity)

    def get_proximity_to_wire(self, point: Tuple[int, int], wire_points: List[Tuple[int, int]]) -> float:
        """
        Get the minimum distance from a point to the wire path.
        Returns distance as a ratio of the tolerance (0 = on wire, 1 = at tolerance boundary)
        """
        if not wire_points or len(wire_points) < 2:
            return float('inf')
        
        min_distance = float('inf')
        px, py = point
        
        for i in range(len(wire_points) - 1):
            distance = self.point_line_distance(px, py, *wire_points[i], *wire_points[i + 1])
            min_distance = min(min_distance, distance)
        
        return min_distance / self.tolerance if self.tolerance > 0 else float('inf')

    def check_proximity_and_play_sound(self, point: Tuple[int, int], wire_points: List[Tuple[int, int]]):
        """Check proximity to wire and play appropriate sound based on distance"""
        proximity_ratio = self.get_proximity_to_wire(point, wire_points)
        
        if proximity_ratio >= 1.0:
            self.play_beep('high')
            return True
        elif proximity_ratio >= self.medium_threshold:
            self.play_beep('medium')
        elif proximity_ratio >= self.light_threshold:
            self.play_beep('light')
        
        return False

    # -------------------- Controls -------------------- #
    def start_game(self):
        """Start the game (only if player name is set)"""
        if not self.can_start_game():
            return False  # Cannot start without name
            
        if not self.game_started:
            self.game_started = True
            self.game_over = False
            self.won = False
            self.finger_trace = []
            self.start_time = time.time()
            self.elapsed_time = 0.0
            self.last_smooth_position = None
            self.frames_without_finger = 0
            self.finger_detected = False
            self.last_beep_time = 0
            self.last_beep_intensity = None
        return True

    def reset_game(self):
        """Reset the game"""
        self.game_started = False
        self.game_over = False
        self.won = False
        self.finger_trace = []
        self.start_time = None
        self.elapsed_time = 0.0
        self.last_smooth_position = None
        self.frames_without_finger = 0
        self.finger_detected = False
        self.last_beep_time = 0
        self.last_beep_intensity = None

        # Generate a fresh wire
        ret, frame = self.cap.read()
        if ret:
            h, w, _ = frame.shape
            self.wire_points = self.generate_random_wire(w - 100, h)
        else:
            self.wire_points = []

    def stop(self):
        """Release resources"""
        try:
            if self.hands:
                self.hands.close()
        except Exception:
            pass
        self.cap.release()
        cv2.destroyAllWindows()
        
        if SOUND_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    # -------------------- Frame Generator -------------------- #
    def get_frame(self, key: Optional[str] = None) -> Optional[bytes]:
        """Capture, process, draw, and return the current frame as JPEG bytes"""
        if key:
            key = key.lower()
            if key == 's':
                self.start_game()  # Will only start if name is set
            elif key == 'r':
                self.reset_game()

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Generate wire once
        if not self.wire_points:
            self.wire_points = self.generate_random_wire(w - 100, h)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        # Draw wire
        self.draw_wire(frame, self.wire_points)

        # Update timer
        if self.game_started and not self.game_over and self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time

        # Draw timer
        self.draw_timer(frame, self.elapsed_time, self.best_time, w, h)

        # Reset detection flag
        self.finger_detected = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                index_tip = landmarks[8]
                raw_x, raw_y = int(index_tip.x * w), int(index_tip.y * h)

                index_up = landmarks[8].y < landmarks[6].y
                if index_up and (landmarks[8].y < landmarks[12].y):
                    self.finger_detected = True
                    self.frames_without_finger = 0

                    smooth_pos = self.smooth_position((raw_x, raw_y), self.last_smooth_position, self.smoothing_factor)
                    self.last_smooth_position = smooth_pos
                    index_x, index_y = smooth_pos

                    if self.game_started and not self.game_over:
                        # Add point to trace
                        if self.should_add_point((index_x, index_y), self.finger_trace[-1] if self.finger_trace else None, self.min_movement_threshold):
                            self.finger_trace.append((index_x, index_y))
                            if len(self.finger_trace) > 1000:
                                self.finger_trace = self.finger_trace[-1000:]

                        # Check proximity and play appropriate sound
                        collision = self.check_proximity_and_play_sound((index_x, index_y), self.wire_points)
                        
                        if collision:
                            self.game_over = True
                            self.won = False

                        # Check win condition
                        if (index_x - self.wire_points[-1][0]) ** 2 + (index_y - self.wire_points[-1][1]) ** 2 < (self.end_circle_radius + 10) ** 2:
                            if (self.best_time is None) or (self.elapsed_time < self.best_time):
                                self.best_time = self.elapsed_time
                            
                            # Add to leaderboard
                            if self.current_player_name:
                                self.add_to_leaderboard(self.current_player_name, self.elapsed_time)
                            
                            self.game_over = True
                            self.won = True

                    cv2.circle(frame, (index_x, index_y), 8, (255, 0, 255), -1)

        if not self.finger_detected:
            self.frames_without_finger += 1
            if self.frames_without_finger > self.max_trace_gap:
                self.last_smooth_position = None

        # Draw finger trace
        if self.game_started and len(self.finger_trace) > 1:
            for i in range(1, len(self.finger_trace)):
                cv2.line(frame, self.finger_trace[i - 1], self.finger_trace[i], (0, 0, 0), self.trace_thickness)

        # Display messages
        if self.game_over:
            if self.won:
                cv2.putText(frame, "🎉 Congratulations! You Won!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, "Score added to leaderboard!", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'r' to Start New Game", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "💀 GAME OVER!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, "Press 'r' to Restart", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif not self.game_started:
            if not self.can_start_game():
                cv2.putText(frame, "Enter your name first!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                cv2.putText(frame, "Name required to start game", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, f"Player: {self.current_player_name}", (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Press 's' to Start", (50, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                cv2.putText(frame, "Point with index finger only!", (50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            return None
        return buffer.tobytes()

    # -------------------- Helpers -------------------- #
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

    @staticmethod
    def put_center_text(frame: np.ndarray, text: str, h: int, color=(0, 255, 0)):
        cv2.putText(frame, text, (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    @staticmethod
    def put_sub_text(frame: np.ndarray, text: str, h: int, color=(0, 255, 0)):
        cv2.putText(frame, text, (50, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)