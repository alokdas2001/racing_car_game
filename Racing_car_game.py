import cv2
import mediapipe as mp
import pygame
import pygame.mixer
import random
import time
import numpy as np
import json
import os
from datetime import datetime
import math

# Initialize pygame
pygame.init()

# Get screen dimensions for fullscreen
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h
LANE_WIDTH = SCREEN_WIDTH // 3
CAR_WIDTH = 80
CAR_HEIGHT = 120
OBSTACLE_WIDTH = 70
OBSTACLE_HEIGHT = 100
FPS = 60

# Camera feed circle (moved to bottom-left corner)
CAMERA_FEED_RADIUS = 60
CAMERA_FEED_POS = (CAMERA_FEED_RADIUS + 20, SCREEN_HEIGHT - CAMERA_FEED_RADIUS - 20)

# Enhanced Gaming Color Palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)
DARK_RED = (139, 0, 0)
PURPLE = (128, 0, 128)
DARK_BLUE = (0, 0, 139)
LIME = (50, 205, 50)
NAVY = (25, 25, 112)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)

# Gaming-inspired colors
NEON_BLUE = (0, 191, 255)
NEON_GREEN = (57, 255, 20)
NEON_PINK = (255, 20, 147)
ELECTRIC_BLUE = (125, 249, 255)
GAMING_PURPLE = (148, 0, 211)
CYBER_ORANGE = (255, 140, 0)
MATRIX_GREEN = (0, 255, 65)
PLASMA_PINK = (255, 105, 180)
TECH_GRAY = (47, 79, 79)
STEEL_BLUE = (70, 130, 180)

# Score file
SCORES_FILE = "highway_rush_scores.json"

class SoundManager:
    def __init__(self):
        """UPDATED: Enabled sound manager with engine startup sound"""
        try:
            # Initialize pygame mixer[14][15][16]
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()
            
            self.audio_enabled = True
            self.engine_sound = None
            self.engine_channel = None
            
            # Load engine startup sound from specified path
            self.engine_sound_path = r"C:\project\sounds\engine_startup.wav"
            self.load_sounds()
            
        except Exception as e:
            print(f"Sound initialization failed: {e}")
            self.audio_enabled = False
            self.engine_sound = None
            self.engine_channel = None
    
    def load_sounds(self):
        """Load engine startup sound"""
        try:
            if self.audio_enabled and os.path.exists(self.engine_sound_path):
                # Load the engine startup sound[17][20][44]
                self.engine_sound = pygame.mixer.Sound(self.engine_sound_path)
                print(f"Engine sound loaded successfully from {self.engine_sound_path}")
            else:
                print(f"Engine sound file not found at {self.engine_sound_path}")
                self.audio_enabled = False
        except Exception as e:
            print(f"Failed to load engine sound: {e}")
            self.audio_enabled = False
            self.engine_sound = None
    
    def play_engine_startup(self):
        """Play engine startup sound continuously"""
        try:
            if self.audio_enabled and self.engine_sound:
                # Stop any existing engine sound
                self.stop_engine()
                # Play engine sound in loop (-1 means infinite loop)[16][24]
                self.engine_channel = self.engine_sound.play(-1)
                print("Engine startup sound started playing")
        except Exception as e:
            print(f"Failed to play engine sound: {e}")
    
    def stop_engine(self):
        """Stop engine sound"""
        try:
            if self.engine_channel:
                self.engine_channel.stop()
                self.engine_channel = None
                print("Engine sound stopped")
        except Exception as e:
            print(f"Failed to stop engine sound: {e}")
    
    def set_engine_volume(self, volume):
        """Set engine sound volume (0.0 to 1.0)"""
        try:
            if self.audio_enabled and self.engine_sound:
                self.engine_sound.set_volume(max(0.0, min(1.0, volume)))
        except Exception as e:
            print(f"Failed to set engine volume: {e}")
    
    def is_engine_playing(self):
        """Check if engine sound is currently playing"""
        try:
            return self.engine_channel and self.engine_channel.get_busy()
        except:
            return False
    
    # Keep these methods for compatibility but they won't do anything
    def play_sound(self, sound_name, volume=None):
        """Disabled - no sound playback except engine"""
        pass
    
    def start_engine_startup_loop(self):
        """Use play_engine_startup instead"""
        self.play_engine_startup()
    
    def stop_engine_startup_loop(self):
        """Use stop_engine instead"""
        self.stop_engine()
    
    def play_engine_loop(self, speed_multiplier):
        """Engine sound with speed-based volume adjustment"""
        if not self.is_engine_playing():
            self.play_engine_startup()
        
        # Adjust volume based on speed (0.3 to 1.0 range)
        try:
            volume = 0.3 + (speed_multiplier - 0.5) * 0.35
            volume = max(0.3, min(1.0, volume))
            self.set_engine_volume(volume)
        except:
            pass
    
    def stop_all_engine_sounds(self):
        """Stop all engine sounds"""
        self.stop_engine()
    
    def play_background_music(self, music_file, loop=-1):
        """Disabled - no background music"""
        pass
    
    def stop_music(self):
        """Disabled - no music"""
        pass
    
    def test_audio(self):
        """Test if audio is working"""
        return self.audio_enabled

class ScoreManager:
    def __init__(self):
        self.scores = self.load_scores()
    
    def load_scores(self):
        """Load scores from file with error handling and data validation"""
        try:
            if os.path.exists(SCORES_FILE):
                with open(SCORES_FILE, 'r') as f:
                    scores = json.load(f)
                    
                # Validate and fix score entries
                validated_scores = []
                for score in scores:
                    try:
                        # Ensure all required fields exist with defaults
                        validated_score = {
                            "player": str(score.get("player", "Unknown")),
                            "score": int(score.get("score", 0)),
                            "survival_time": int(score.get("survival_time", 0)),
                            "max_speed": float(score.get("max_speed", 1.0)),
                            "date": str(score.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        }
                        validated_scores.append(validated_score)
                    except (ValueError, TypeError) as e:
                        continue
                
                return validated_scores
            return []
        except Exception as e:
            return []
    
    def save_scores(self):
        """Save scores to file with proper error handling"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(SCORES_FILE)), exist_ok=True)
            
            with open(SCORES_FILE, 'w') as f:
                json.dump(self.scores, f, indent=2)
        except Exception as e:
            pass
    
    def add_score(self, player_name, score, survival_time, max_speed):
        """Add a new score entry with data validation"""
        try:
            score_entry = {
                "player": str(player_name) if player_name else "Unknown",
                "score": int(score) if score is not None else 0,
                "survival_time": int(survival_time) if survival_time is not None else 0,
                "max_speed": float(max_speed) if max_speed is not None else 1.0,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.scores.append(score_entry)
            self.scores.sort(key=lambda x: x["score"], reverse=True)
            self.scores = self.scores[:100]  # Keep top 100 scores
            self.save_scores()
            
        except Exception as e:
            pass
    
    def get_top_scores(self, limit=100):
        """Get top scores with validation"""
        try:
            return self.scores[:limit]
        except Exception as e:
            return []

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def get_hand_position(self, frame):
        """Returns the x-coordinate of the hand center (0-1 range)"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    return wrist.x, wrist.y
            return None, None
        except Exception as e:
            return None, None

class Car:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2 - CAR_WIDTH // 2
        self.y = SCREEN_HEIGHT - CAR_HEIGHT - 50
        self.target_x = self.x
        self.smoothing_factor = 0.35
        
    def update_position(self, hand_x):
        """Continuously track hand position and move car accordingly"""
        if hand_x is not None:
            padding = CAR_WIDTH // 2
            self.target_x = hand_x * (SCREEN_WIDTH - 2 * padding) + padding
        
        self.x += (self.target_x - self.x) * self.smoothing_factor
        self.x = max(0, min(SCREEN_WIDTH - CAR_WIDTH, self.x))
    
    def get_current_lane(self):
        """Get which lane the car is currently in"""
        car_center = self.x + CAR_WIDTH // 2
        if car_center < LANE_WIDTH:
            return 0
        elif car_center < 2 * LANE_WIDTH:
            return 1
        else:
            return 2
    
    def draw(self, screen, is_blinking=False, blink_timer=0):
        """Draw car with optional blinking effect for invincibility"""
        if is_blinking and (blink_timer // 10) % 2 == 1:
            return
            
        # Car body
        pygame.draw.rect(screen, BLUE, (int(self.x), self.y, CAR_WIDTH, CAR_HEIGHT))
        
        # Car windshield
        pygame.draw.rect(screen, WHITE, (int(self.x) + 10, self.y + 10, CAR_WIDTH - 20, 25))
        
        # Headlights
        pygame.draw.circle(screen, YELLOW, (int(self.x) + 15, self.y + 5), 6)
        pygame.draw.circle(screen, YELLOW, (int(self.x) + CAR_WIDTH - 15, self.y + 5), 6)
        
        # Wheels
        pygame.draw.circle(screen, BLACK, (int(self.x) + 15, self.y + CAR_HEIGHT - 10), 10)
        pygame.draw.circle(screen, BLACK, (int(self.x) + CAR_WIDTH - 15, self.y + CAR_HEIGHT - 10), 10)
        pygame.draw.circle(screen, GRAY, (int(self.x) + 15, self.y + CAR_HEIGHT - 10), 6)
        pygame.draw.circle(screen, GRAY, (int(self.x) + CAR_WIDTH - 15, self.y + CAR_HEIGHT - 10), 6)
        
        # Invincibility indicator
        if is_blinking:
            pygame.draw.rect(screen, ELECTRIC_BLUE, (int(self.x) - 5, self.y - 5, CAR_WIDTH + 10, CAR_HEIGHT + 10), 3)

class Obstacle:
    def __init__(self, x_position, speed_multiplier=1.0, vehicle_type="truck"):
        self.x = max(10, min(SCREEN_WIDTH - OBSTACLE_WIDTH - 10, x_position - OBSTACLE_WIDTH // 2))
        self.y = -OBSTACLE_HEIGHT
        self.base_speed = 10  # Increased base speed
        self.speed = int(self.base_speed * speed_multiplier)
        self.vehicle_type = vehicle_type
        self.color = self.get_vehicle_color()
        
    def get_vehicle_color(self):
        """Return random color for vehicle variety"""
        colors = [RED, PURPLE, ORANGE, DARK_RED, GREEN, DARK_BLUE, YELLOW, CYAN, MAGENTA]
        return random.choice(colors)
        
    def update(self):
        self.y += self.speed
        
    def draw(self, screen):
        if self.vehicle_type == "truck":
            self.draw_truck(screen)
        elif self.vehicle_type == "car":
            self.draw_car(screen)
        elif self.vehicle_type == "bus":
            self.draw_bus(screen)
            
    def draw_truck(self, screen):
        # Truck body
        body_rect = pygame.Rect(self.x, self.y + 25, OBSTACLE_WIDTH, OBSTACLE_HEIGHT - 25)
        pygame.draw.rect(screen, self.color, body_rect)
        
        # Truck cabin
        cabin_width = 30
        cabin_height = 35
        cabin_rect = pygame.Rect(self.x + OBSTACLE_WIDTH - cabin_width, self.y, cabin_width, cabin_height)
        pygame.draw.rect(screen, YELLOW, cabin_rect)
        
        # Windshield
        windshield_rect = pygame.Rect(self.x + OBSTACLE_WIDTH - cabin_width + 5, self.y + 5, cabin_width - 10, 15)
        pygame.draw.rect(screen, WHITE, windshield_rect)
        
        # Wheels
        wheel_radius = 8
        wheel_y = self.y + OBSTACLE_HEIGHT - wheel_radius
        pygame.draw.circle(screen, BLACK, (self.x + OBSTACLE_WIDTH - 15, wheel_y), wheel_radius)
        pygame.draw.circle(screen, BLACK, (self.x + 15, wheel_y), wheel_radius)
        pygame.draw.circle(screen, GRAY, (self.x + OBSTACLE_WIDTH - 15, wheel_y), wheel_radius - 2)
        pygame.draw.circle(screen, GRAY, (self.x + 15, wheel_y), wheel_radius - 2)
        
    def draw_car(self, screen):
        # Car body
        car_rect = pygame.Rect(self.x + 10, self.y + 20, OBSTACLE_WIDTH - 20, OBSTACLE_HEIGHT - 20)
        pygame.draw.rect(screen, self.color, car_rect)
        
        # Windshield
        windshield_rect = pygame.Rect(self.x + 15, self.y + 25, OBSTACLE_WIDTH - 30, 20)
        pygame.draw.rect(screen, WHITE, windshield_rect)
        
        # Wheels
        wheel_radius = 6
        wheel_y = self.y + OBSTACLE_HEIGHT - 10
        pygame.draw.circle(screen, BLACK, (self.x + 20, wheel_y), wheel_radius)
        pygame.draw.circle(screen, BLACK, (self.x + OBSTACLE_WIDTH - 20, wheel_y), wheel_radius)
        
    def draw_bus(self, screen):
        # Bus body
        bus_rect = pygame.Rect(self.x, self.y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
        pygame.draw.rect(screen, YELLOW, bus_rect)
        
        # Windows
        for i in range(3):
            window_rect = pygame.Rect(self.x + 8 + i * 18, self.y + 10, 15, 20)
            pygame.draw.rect(screen, WHITE, window_rect)
        
        # Wheels
        wheel_radius = 8
        wheel_y = self.y + OBSTACLE_HEIGHT - wheel_radius
        pygame.draw.circle(screen, BLACK, (self.x + 15, wheel_y), wheel_radius)
        pygame.draw.circle(screen, BLACK, (self.x + OBSTACLE_WIDTH - 15, wheel_y), wheel_radius)
        
    def is_off_screen(self):
        return self.y > SCREEN_HEIGHT
        
    def collides_with_car(self, car):
        car_rect = pygame.Rect(int(car.x), car.y, CAR_WIDTH, CAR_HEIGHT)
        obstacle_rect = pygame.Rect(self.x, self.y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
        return car_rect.colliderect(obstacle_rect)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("🚗 Highway Rush - Hand Control Challenge 🚛")
        self.clock = pygame.time.Clock()
        
        # UPDATED: Enabled sound manager with engine sound
        self.sound_manager = SoundManager()
        
        # Enhanced Gaming Fonts
        self.font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.022))
        self.small_font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.016))
        self.big_font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.055))
        self.title_font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.035))
        self.huge_font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.085))
        self.mega_font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.1))
        
        # Animation timers
        self.animation_timer = 0
        self.pulse_timer = 0
        
        # Enhanced scrollable leaderboard system
        self.scroll_offset = 0
        self.max_visible_scores = 8  # Number of scores visible at once
        self.scroll_speed = 2  # How many entries to scroll per action
        self.scroll_animation_timer = 0
        self.scroll_target = 0
        self.smooth_scroll_speed = 0.2
        
        # Game objects
        self.car = Car()
        self.obstacles = []
        self.hand_tracker = HandTracker()
        self.score_manager = ScoreManager()
        
        # Game state
        self.current_state = "name_input"
        self.player_name = ""
        self.score = 0
        self.total_score = 0
        self.combined_survival_time = 0
        self.max_speed_achieved = 0
        self.lives_remaining = 2
        self.game_start_time = 0
        self.current_life_start_time = 0
        self.obstacle_spawn_timer = 0
        self.spawn_delay = 40  # Increased for better spacing
        
        # New crash and respawn system
        self.respawn_countdown = 0
        self.respawn_countdown_timer = 0
        self.invincible_until = 0
        self.blink_timer = 0
        self.crash_message_timer = 0
        
        # Input handling
        self.input_active = True
        self.cursor_visible = True
        self.cursor_timer = 0
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.current_frame = None

    def get_total_game_time(self):
        """Get total time since the entire game started"""
        try:
            if self.game_start_time == 0:
                return 0
            return time.time() - self.game_start_time
        except Exception as e:
            return 0
        
    def get_speed_multiplier(self):
        """Gradual speed progression system"""
        try:
            total_time = self.get_total_game_time()
            
            # First 15 seconds: 0.5x speed
            if total_time <= 15:
                factor = 0.5
            # At 20 seconds (15 + 5): 1.0x speed  
            elif total_time <= 20:
                factor = 1.0
            # After 20 seconds: increase by 0.2x every 5 seconds
            else:
                # Calculate how many 5-second intervals have passed since 20 seconds
                intervals_since_20 = (total_time - 20) // 5
                # Start at 1.0x and add 0.2x for each interval: 1.2x, 1.4x, 1.6x, etc.
                factor = 1.0 + (intervals_since_20 + 1) * 0.2
            
            # Maximum speed cap to prevent excessive difficulty
            max_speed = 10.0
            if factor > max_speed:
                factor = max_speed
                
            return factor
        except Exception as e:
            return 1.0
        
    def get_current_survival_time(self):
        """Get survival time for current life"""
        try:
            if self.current_life_start_time == 0:
                return 0
            return int(time.time() - self.current_life_start_time)
        except Exception as e:
            return 0

    # Enhanced scroll management methods
    def handle_scroll_up(self):
        """Scroll up in the leaderboard"""
        try:
            self.scroll_target = max(0, self.scroll_target - self.scroll_speed)
        except Exception as e:
            pass

    def handle_scroll_down(self):
        """Scroll down in the leaderboard"""
        try:
            total_scores = len(self.score_manager.get_top_scores())
            max_scroll = max(0, total_scores - self.max_visible_scores)
            self.scroll_target = min(max_scroll, self.scroll_target + self.scroll_speed)
        except Exception as e:
            pass

    def update_scroll_animation(self):
        """Smooth scrolling animation"""
        try:
            if abs(self.scroll_offset - self.scroll_target) > 0.1:
                self.scroll_offset += (self.scroll_target - self.scroll_offset) * self.smooth_scroll_speed
            else:
                self.scroll_offset = self.scroll_target
        except Exception as e:
            pass

    def reset_scroll(self):
        """Reset scroll position"""
        try:
            self.scroll_offset = 0
            self.scroll_target = 0
        except Exception as e:
            pass

    def handle_crash(self):
        """Handle car crash with new respawn system"""
        try:
            # UPDATED: Stop engine sound on crash
            self.sound_manager.stop_engine()
            
            if self.lives_remaining > 1:
                self.lives_remaining -= 1
                self.current_state = "respawning"
                self.respawn_countdown = 5
                self.respawn_countdown_timer = 0
                self.crash_message_timer = 0
                
                # Clear obstacles to give player breathing room
                self.obstacles = []
                
            else:
                self.end_game()
                
        except Exception as e:
            pass

    def update_respawn_countdown(self):
        """Update respawn countdown with faster timing"""
        try:
            self.respawn_countdown_timer += 1
            
            countdown_frame_threshold = 20
            
            if self.respawn_countdown_timer >= countdown_frame_threshold:
                self.respawn_countdown -= 1
                self.respawn_countdown_timer = 0
                
                if self.respawn_countdown <= 0:
                    self.car = Car()
                    self.invincible_until = time.time() + 5
                    self.current_state = "playing"
                    
                    # UPDATED: Restart engine sound when respawning
                    self.sound_manager.play_engine_startup()
                    
        except Exception as e:
            pass

    def end_game(self):
        """End the game and calculate final stats"""
        try:
            self.current_state = "final_game_over"
            
            # UPDATED: Stop engine sound when game ends
            self.sound_manager.stop_engine()
            
            current_survival = self.get_current_survival_time()
            self.combined_survival_time += current_survival
            
            current_speed = self.get_speed_multiplier()
            if current_speed > self.max_speed_achieved:
                self.max_speed_achieved = current_speed
                
            self.total_score += self.score
            
            # Save the final score
            self.score_manager.add_score(
                str(self.player_name) if self.player_name else "Unknown",
                int(self.total_score) if self.total_score is not None else 0,
                int(self.combined_survival_time) if self.combined_survival_time is not None else 0,
                float(self.max_speed_achieved) if self.max_speed_achieved is not None else 1.0
            )
            
            # Reset scroll when ending game
            self.reset_scroll()
            
        except Exception as e:
            pass

    def start_new_game(self):
        """UPDATED: Start a brand new game with engine sound"""
        try:
            # Initialize game state silently
            self.car = Car()
            self.obstacles = []
            self.score = 0
            self.total_score = 0
            self.combined_survival_time = 0
            self.lives_remaining = 2
            self.game_start_time = time.time()
            self.current_life_start_time = time.time()
            self.obstacle_spawn_timer = 0
            self.spawn_delay = 40  # Increased for better spacing
            self.respawn_countdown = 0
            self.invincible_until = 0
            self.blink_timer = 0
            self.max_speed_achieved = 0
            self.current_state = "playing"
            self.reset_scroll()
            
            # UPDATED: Start engine sound when game begins
            self.sound_manager.play_engine_startup()
            
        except Exception as e:
            pass

    def restart_with_new_name(self):
        """Restart game and go back to name input screen"""
        try:
            # UPDATED: Stop engine sound when restarting
            self.sound_manager.stop_engine()
            
            # Reset all game state
            self.current_state = "name_input"
            self.player_name = ""
            self.score = 0
            self.total_score = 0
            self.combined_survival_time = 0
            self.lives_remaining = 2
            self.game_start_time = 0
            self.current_life_start_time = 0
            self.obstacle_spawn_timer = 0
            self.spawn_delay = 40  # Increased for better spacing
            self.respawn_countdown = 0
            self.invincible_until = 0
            self.blink_timer = 0
            self.max_speed_achieved = 0
            self.reset_scroll()
            
            # Reset objects
            self.car = Car()
            self.obstacles = []
            
        except Exception as e:
            pass

    def spawn_obstacle_wave(self):
        """Limited car spawning with guaranteed navigable spaces"""
        try:
            current_speed_multiplier = self.get_speed_multiplier()
            
            # REDUCED: Maximum 5 cars at once with better progression
            if current_speed_multiplier < 1:
                num_vehicles = random.randint(1, 2)  # Start very easy
            elif current_speed_multiplier < 2:
                num_vehicles = random.randint(2, 3)  # Gradual increase
            elif current_speed_multiplier < 4:
                num_vehicles = random.randint(3, 4)  # Medium difficulty
            else:
                num_vehicles = random.randint(3, 5)  # Max 5 cars for highest speeds
            
            vehicle_types = ["truck", "car", "bus"]
            new_obstacles = []
            
            # IMPROVED: Ensure navigable spaces by dividing screen into sections
            screen_sections = 5  # Divide screen into 5 equal sections
            section_width = SCREEN_WIDTH // screen_sections
            occupied_sections = []
            
            # Always ensure at least 2 sections remain free for navigation
            max_occupied = min(num_vehicles, screen_sections - 2)
            
            for _ in range(max_occupied):
                attempts = 0
                max_attempts = 20
                
                while attempts < max_attempts:
                    # Choose a random section
                    section = random.randint(0, screen_sections - 1)
                    
                    if section not in occupied_sections:
                        # Calculate position within the section
                        section_start = section * section_width
                        section_end = (section + 1) * section_width
                        
                        # Ensure obstacle fits within screen bounds
                        min_x = max(section_start + OBSTACLE_WIDTH // 2, OBSTACLE_WIDTH)
                        max_x = min(section_end - OBSTACLE_WIDTH // 2, SCREEN_WIDTH - OBSTACLE_WIDTH)
                        
                        if min_x < max_x:
                            random_x = random.randint(min_x, max_x)
                            
                            # Check spacing with existing obstacles
                            valid_position = True
                            for existing_obstacle in self.obstacles:
                                if existing_obstacle.y > -OBSTACLE_HEIGHT - 150:  # Check recently spawned
                                    distance = abs(existing_obstacle.x + OBSTACLE_WIDTH//2 - random_x)
                                    if distance < 150:  # Minimum safe distance
                                        valid_position = False
                                        break
                            
                            if valid_position:
                                occupied_sections.append(section)
                                vehicle_type = random.choice(vehicle_types)
                                obstacle = Obstacle(random_x, current_speed_multiplier, vehicle_type)
                                new_obstacles.append(obstacle)
                                break
                    
                    attempts += 1
            
            # Add new obstacles to game
            self.obstacles.extend(new_obstacles)
            
        except Exception as e:
            pass

    def update_obstacles(self):
        """Update all obstacles and check for collisions"""
        try:
            for obstacle in self.obstacles[:]:
                obstacle.update()
                
                # Check collision only if not invincible
                if time.time() > self.invincible_until and obstacle.collides_with_car(self.car):
                    self.handle_crash()
                    return
                    
                if obstacle.is_off_screen():
                    self.obstacles.remove(obstacle)
                    base_points = 25
                    speed_bonus = int(self.get_speed_multiplier() * 10)
                    self.score += base_points + speed_bonus
        except Exception as e:
            pass

    def draw_road(self):
        """Draw clean road"""
        self.screen.fill(GRAY)
        
        # Draw lane dividers
        for i in range(1, 3):
            x = i * LANE_WIDTH
            for y in range(0, SCREEN_HEIGHT, 50):
                pygame.draw.rect(self.screen, WHITE, (x - 2, y, 4, 25))
                
        # Road edges
        pygame.draw.line(self.screen, WHITE, (0, 0), (0, SCREEN_HEIGHT), 6)
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), 6)

    def draw_animated_background(self, base_color1, base_color2):
        """Draw animated gaming background"""
        self.animation_timer += 0.02
        
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            wave_offset = math.sin(self.animation_timer + y * 0.005) * 0.1
            ratio = max(0, min(1, ratio + wave_offset))
            
            r = int(base_color1[0] * (1 - ratio) + base_color2[0] * ratio)
            g = int(base_color1[1] * (1 - ratio) + base_color2[1] * ratio)
            b = int(base_color1[2] * (1 - ratio) + base_color2[2] * ratio)
            
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

    def draw_neon_text(self, text, font, color, position, glow_color=None):
        """Draw text with neon glow effect"""
        if glow_color is None:
            glow_color = color
            
        glow_offsets = [(2, 2), (-2, -2), (2, -2), (-2, 2), (0, 3), (0, -3), (3, 0), (-3, 0)]
        
        for offset in glow_offsets:
            glow_surface = font.render(text, True, glow_color)
            glow_surface.set_alpha(50)
            glow_pos = (position[0] + offset[0], position[1] + offset[1])
            self.screen.blit(glow_surface, glow_pos)
        
        main_surface = font.render(text, True, color)
        self.screen.blit(main_surface, position)
        
        return main_surface.get_rect(topleft=position)

    def draw_gaming_panel(self, rect, primary_color, secondary_color, border_width=4):
        """Draw enhanced gaming-style panel"""
        for i in range(rect.height):
            ratio = i / rect.height
            r = int(primary_color[0] * (1 - ratio * 0.3) + secondary_color[0] * (ratio * 0.3))
            g = int(primary_color[1] * (1 - ratio * 0.3) + secondary_color[1] * (ratio * 0.3))
            b = int(primary_color[2] * (1 - ratio * 0.3) + secondary_color[2] * (ratio * 0.3))
            pygame.draw.line(self.screen, (r, g, b), (rect.x, rect.y + i), (rect.right, rect.y + i))
        
        pulse = math.sin(self.animation_timer * 3) * 0.3 + 0.7
        border_color = (
            int(secondary_color[0] * pulse),
            int(secondary_color[1] * pulse),
            int(secondary_color[2] * pulse)
        )
        pygame.draw.rect(self.screen, border_color, rect, border_width)

    def draw_ui(self):
        """Compact UI panel with only essential information"""
        # REDUCED panel size significantly
        ui_width = int(SCREEN_WIDTH * 0.18)  # Reduced from 0.32 to 0.18
        ui_height = 160  # Reduced from 300 to 160
        ui_rect = pygame.Rect(10, 10, ui_width, ui_height)
        
        self.draw_gaming_panel(ui_rect, TECH_GRAY, ELECTRIC_BLUE, 5)
        
        # Player name - smaller font and positioning
        player_y = 15
        player_font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.024))  # Smaller font
        self.draw_neon_text(f"👑 {self.player_name.upper()}", player_font, NEON_GREEN,
                           (15, player_y), MATRIX_GREEN)
        
        # ONLY essential stats with compact spacing
        stats = [
            (f"💖 Lives: {self.lives_remaining}", RED if self.lives_remaining == 1 else NEON_GREEN, 45),
            (f"⏱️ Time: {self.combined_survival_time + self.get_current_survival_time()}s", ELECTRIC_BLUE, 70),
        ]
        
        # Use smaller font for stats
        stat_font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.018))
        
        for text, color, y_pos in stats:
            stat_surface = stat_font.render(text, True, color)
            self.screen.blit(stat_surface, (15, y_pos))
        
        # Speed indicator with compact design
        speed_multiplier = self.get_speed_multiplier()
        total_time = self.get_total_game_time()
        
        # Determine speed status and color
        if total_time <= 15:
            speed_color = NEON_BLUE
            speed_status = "START"
        elif total_time <= 20:
            speed_color = NEON_GREEN
            speed_status = "NORM" 
        elif speed_multiplier < 2:
            speed_color = CYBER_ORANGE
            speed_status = "FAST"
        elif speed_multiplier < 4:
            speed_color = PLASMA_PINK
            speed_status = "RAPID"
        elif speed_multiplier < 6:
            speed_color = RED
            speed_status = "EXTREME"
        else:
            speed_color = MAGENTA
            speed_status = "INSANE"
            
        speed_text = f"🚀 {speed_multiplier:.1f}x ({speed_status})"
        speed_surface = stat_font.render(speed_text, True, speed_color)
        self.screen.blit(speed_surface, (15, 95))
        
        # Compact score display
        score_text = f"💰 {self.total_score + self.score:,}"
        score_surface = stat_font.render(score_text, True, CYBER_ORANGE)
        self.screen.blit(score_surface, (15, 120))
        
        # Enhanced invincibility indicator (moved to not overlap with compact UI)
        if time.time() < self.invincible_until:
            remaining_invincibility = self.invincible_until - time.time()
            
            shield_alpha = int(150 + 105 * math.sin(self.animation_timer * 10))
            shield_width = 400  # Reduced width
            shield_height = 60   # Reduced height
            shield_x = SCREEN_WIDTH // 2 - shield_width // 2
            shield_y = 20  # Moved down slightly
            
            shield_surface = pygame.Surface((shield_width, shield_height))
            shield_surface.set_alpha(shield_alpha)
            shield_surface.fill(ELECTRIC_BLUE)
            self.screen.blit(shield_surface, (shield_x, shield_y))
            
            pygame.draw.rect(self.screen, NEON_BLUE, (shield_x, shield_y, shield_width, shield_height), 3)
            
            invincible_text = f"🛡️ SHIELD: {remaining_invincibility:.1f}s"
            invincible_font = pygame.font.Font(None, int(SCREEN_WIDTH * 0.025))
            self.draw_neon_text(invincible_text, invincible_font, ELECTRIC_BLUE,
                               (shield_x + 60, shield_y + 20), NEON_BLUE)

    def draw_name_input_screen(self):
        """Draw enhanced gaming-style name input screen"""
        self.draw_animated_background(TECH_GRAY, NAVY)
        
        title_y = SCREEN_HEIGHT // 8
        self.draw_neon_text("HIGHWAY RUSH", self.mega_font, NEON_GREEN, 
                           (SCREEN_WIDTH // 2 - 300, title_y), ELECTRIC_BLUE)
        
        input_section_width = 600
        input_section_height = 250
        input_section_x = (SCREEN_WIDTH - input_section_width) // 2
        input_section_y = SCREEN_HEIGHT // 2 - 125
        
        input_rect = pygame.Rect(input_section_x, input_section_y, input_section_width, input_section_height)
        self.draw_gaming_panel(input_rect, GAMING_PURPLE, ELECTRIC_BLUE, 5)
        
        prompt_y = input_section_y + 40
        self.draw_neon_text("ENTER GAMER TAG", self.title_font, NEON_GREEN,
                           (SCREEN_WIDTH // 2 - 120, prompt_y), MATRIX_GREEN)
        
        input_box_width = 450
        input_box_height = 60
        input_box_x = (SCREEN_WIDTH - input_box_width) // 2
        input_box_y = input_section_y + 100
        
        input_box = pygame.Rect(input_box_x, input_box_y, input_box_width, input_box_height)
        
        border_pulse = math.sin(self.animation_timer * 4) * 0.5 + 0.5
        border_color = (
            int(ELECTRIC_BLUE[0] * border_pulse + NEON_BLUE[0] * (1 - border_pulse)),
            int(ELECTRIC_BLUE[1] * border_pulse + NEON_BLUE[1] * (1 - border_pulse)),
            int(ELECTRIC_BLUE[2] * border_pulse + NEON_BLUE[2] * (1 - border_pulse))
        )
        
        pygame.draw.rect(self.screen, BLACK, input_box)
        pygame.draw.rect(self.screen, border_color, input_box, 4)
        
        name_text = self.title_font.render(self.player_name, True, NEON_GREEN)
        name_rect = name_text.get_rect(center=(SCREEN_WIDTH // 2, input_box.centery))
        self.screen.blit(name_text, name_rect)
        
        if self.cursor_visible and len(self.player_name) < 20:
            cursor_x = name_rect.right + 10
            cursor_y = name_rect.centery
            cursor_height = 25
            cursor_alpha = int(255 * (math.sin(self.animation_timer * 6) * 0.5 + 0.5))
            cursor_surface = pygame.Surface((3, cursor_height))
            cursor_surface.set_alpha(cursor_alpha)
            cursor_surface.fill(ELECTRIC_BLUE)
            self.screen.blit(cursor_surface, (cursor_x, cursor_y - cursor_height // 2))
        
       

    def draw_instructions_screen(self):
        """Draw instructions screen"""
        self.draw_animated_background(TECH_GRAY, STEEL_BLUE)
        
        welcome_y = 40
        welcome_text = f"WELCOME {self.player_name.upper()}!"
        welcome_surface = self.big_font.render(welcome_text, True, NEON_GREEN)
        welcome_rect = welcome_surface.get_rect()
        title_x = (SCREEN_WIDTH - welcome_rect.width) // 2
        self.draw_neon_text(welcome_text, self.big_font, NEON_GREEN,
                           (title_x, welcome_y), MATRIX_GREEN)
        
        panel_margin = 60
        panel_width = min(int(SCREEN_WIDTH * 0.85), SCREEN_WIDTH - 2 * panel_margin)
        panel_height = min(int(SCREEN_HEIGHT * 0.65), SCREEN_HEIGHT - welcome_y - 180)
        panel_x = (SCREEN_WIDTH - panel_width) // 2
        panel_y = welcome_y + 100
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        self.draw_gaming_panel(panel_rect, GAMING_PURPLE, ELECTRIC_BLUE, 5)
        
        content_margin = 40
        content_width = panel_width - 2 * content_margin
        col_width = (content_width - 40) // 2
        
        col1_x = panel_x + content_margin
        col2_x = panel_x + content_margin + col_width + 40
        content_y_start = panel_y + 30
        line_height = 28
        
        # Left column - Game Rules
        rules_title_y = content_y_start
        rules_title = "🎮 GAME RULES"
        rules_title_surface = self.title_font.render(rules_title, True, CYBER_ORANGE)
        rules_title_rect = rules_title_surface.get_rect()
        rules_title_x = col1_x + (col_width - rules_title_rect.width) // 2
        self.draw_neon_text(rules_title, self.title_font, CYBER_ORANGE,
                           (rules_title_x, rules_title_y), YELLOW)
        
        rules = [
            "• Hand controls car movement",
            "• Avoid obstacles to score points", 
            "• 2 lives total - survive as long as possible",
            "• Speed: 0.5x→1.0x→1.2x→1.4x...",
            "• Max 5 cars at once with safe gaps",
            "• Compact UI for better visibility + Engine sound!"  # UPDATED
        ]
        
        for i, rule in enumerate(rules):
            rule_y = rules_title_y + 40 + i * line_height
            if rule_y < panel_y + panel_height - 40:
                rule_surface = self.small_font.render(rule, True, WHITE)
                rule_rect = rule_surface.get_rect()
                rule_x = col1_x + (col_width - rule_rect.width) // 2
                self.screen.blit(rule_surface, (rule_x, rule_y))
        
        # Right column - Controls
        controls_title_y = content_y_start
        controls_title = "🕹️ CONTROLS"
        controls_title_surface = self.title_font.render(controls_title, True, CYBER_ORANGE)
        controls_title_rect = controls_title_surface.get_rect()
        controls_title_x = col2_x + (col_width - controls_title_rect.width) // 2
        self.draw_neon_text(controls_title, self.title_font, CYBER_ORANGE,
                           (controls_title_x, controls_title_y), YELLOW)
        
        controls = [
            "• Move hand LEFT/RIGHT in camera view",
            "• Keep hand steady to center car", 
            "• Camera feed shows in bottom-left corner",
            "• ESC key exits game anytime",
            "• Blue dot shows hand tracking status"
        ]
        
        for i, control in enumerate(controls):
            control_y = controls_title_y + 40 + i * line_height
            if control_y < panel_y + panel_height - 40:
                control_surface = self.small_font.render(control, True, WHITE)
                control_rect = control_surface.get_rect()
                control_x = col2_x + (col_width - control_rect.width) // 2
                self.screen.blit(control_surface, (control_x, control_y))
        
        # Tips section - UPDATED
        tips_y = content_y_start + 200
        if tips_y < panel_y + panel_height - 100:
            tips_title = "💡 AUDIO ENHANCED"
            tips_title_surface = self.title_font.render(tips_title, True, NEON_GREEN)
            tips_title_rect = tips_title_surface.get_rect()
            tips_title_x = panel_x + (panel_width - tips_title_rect.width) // 2
            self.draw_neon_text(tips_title, self.title_font, NEON_GREEN,
                               (tips_title_x, tips_y), MATRIX_GREEN)
            
            tip_text = "🔊 Engine sound plays when driving • 🎵 Volume adjusts with speed • 🚗 Immersive experience"
            tip_y = tips_y + 40
            tip_surface = self.small_font.render(tip_text, True, ELECTRIC_BLUE)
            tip_rect = tip_surface.get_rect()
            tip_x = panel_x + (panel_width - tip_rect.width) // 2
            if tip_x + tip_rect.width > panel_x + panel_width - content_margin:
                tip_x = panel_x + content_margin
            self.screen.blit(tip_surface, (tip_x, tip_y))
        
        start_button_y = min(panel_y + panel_height + 40, SCREEN_HEIGHT - 120)
        start_text = "🚀 PRESS SPACE TO START RACING!"
        start_surface = self.big_font.render(start_text, True, NEON_GREEN)
        start_rect = start_surface.get_rect()
        start_x = (SCREEN_WIDTH - start_rect.width) // 2
        self.draw_neon_text(start_text, self.big_font, NEON_GREEN,
                           (start_x, start_button_y), MATRIX_GREEN)

    def draw_crash_and_respawn_ui(self):
        """Draw crash message with countdown"""
        try:
            overlay_alpha = int(150 + 50 * math.sin(self.animation_timer * 5))
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(overlay_alpha)
            overlay.fill(DARK_RED)
            self.screen.blit(overlay, (0, 0))
            
            panel_width = int(SCREEN_WIDTH * 0.7)
            panel_height = int(SCREEN_HEIGHT * 0.5)
            panel_x = (SCREEN_WIDTH - panel_width) // 2
            panel_y = (SCREEN_HEIGHT - panel_height) // 2
            
            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            self.draw_gaming_panel(panel_rect, DARK_RED, RED, 6)
            
            crash_pulse = math.sin(self.animation_timer * 6) * 0.4 + 0.6
            crash_color = (int(255 * crash_pulse), 0, 0)
            self.draw_neon_text("💥 VEHICLE COLLISION! 💥", self.big_font, crash_color,
                               (SCREEN_WIDTH // 2 - 250, panel_y + panel_height * 0.2), RED)
            
            lives_text = f"❤️ {self.lives_remaining} LIFE REMAINING - STAY STRONG!"
            self.draw_neon_text(lives_text, self.title_font, NEON_GREEN,
                               (SCREEN_WIDTH // 2 - 200, panel_y + panel_height * 0.4), MATRIX_GREEN)
            
            if self.respawn_countdown > 0:
                countdown_text = f"RESPAWN IN {self.respawn_countdown}"
                countdown_color = ELECTRIC_BLUE if self.respawn_countdown > 2 else CYBER_ORANGE
                
                self.draw_neon_text(countdown_text, self.big_font, countdown_color,
                                   (SCREEN_WIDTH // 2 - 150, panel_y + panel_height * 0.7), NEON_BLUE)
            
        except Exception as e:
            pass

    def draw_final_game_over(self):
        """Draw final game over screen with scrollable leaderboard"""
        try:
            self.draw_animated_background(BLACK, TECH_GRAY)
            self.update_scroll_animation()
            
            panel_width = int(SCREEN_WIDTH * 0.95)
            panel_height = int(SCREEN_HEIGHT * 0.95)
            panel_x = (SCREEN_WIDTH - panel_width) // 2
            panel_y = (SCREEN_HEIGHT - panel_height) // 2
            
            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            self.draw_gaming_panel(panel_rect, TECH_GRAY, NEON_GREEN, 8)
            
            title_y = panel_y + 30
            self.draw_neon_text("🏁 RACE COMPLETE! 🏁", self.big_font, NEON_GREEN,
                               (SCREEN_WIDTH // 2 - 250, title_y), MATRIX_GREEN)
            
            stats_y = title_y + 80
            player_title = f"🏆 {self.player_name.upper()}'S FINAL PERFORMANCE 🏆"
            self.draw_neon_text(player_title, self.title_font, CYBER_ORANGE,
                               (SCREEN_WIDTH // 2 - len(player_title) * 6, stats_y), PLASMA_PINK)
            
            final_stats_y = stats_y + 50
            stat_spacing = panel_width // 3
            
            stats_data = [
                f"FINAL SCORE: {self.total_score:,}",
                f"TOTAL SURVIVAL: {self.combined_survival_time}s", 
                f"MAX SPEED: {self.max_speed_achieved:.1f}x"
            ]
            
            for i, stat in enumerate(stats_data):
                stat_x = panel_x + 50 + i * stat_spacing
                stat_surface = self.font.render(stat, True, NEON_GREEN)
                self.screen.blit(stat_surface, (stat_x, final_stats_y))
            
            # ENHANCED SCROLLABLE LEADERBOARD SECTION
            leaderboard_y = final_stats_y + 80
            
            header_rect = pygame.Rect(panel_x + 50, leaderboard_y, panel_width - 100, 50)
            pygame.draw.rect(self.screen, GAMING_PURPLE, header_rect)
            pygame.draw.rect(self.screen, ELECTRIC_BLUE, header_rect, 3)
            
            self.draw_neon_text("🏆 HALL OF FAME LEADERBOARD 🏆", self.title_font, NEON_GREEN,
                               (SCREEN_WIDTH // 2 - 180, leaderboard_y + 15), MATRIX_GREEN)
            
            # Scroll instructions
            scroll_instructions_y = leaderboard_y + 55
            if len(self.score_manager.get_top_scores()) > self.max_visible_scores:
                scroll_text = "⬆️⬇️ Use UP/DOWN Arrow Keys to Scroll • Mouse Wheel Supported"
                scroll_surface = self.small_font.render(scroll_text, True, CYBER_ORANGE)
                scroll_rect = scroll_surface.get_rect()
                scroll_x = SCREEN_WIDTH // 2 - scroll_rect.width // 2
                self.screen.blit(scroll_surface, (scroll_x, scroll_instructions_y))
            
            table_start_y = leaderboard_y + 85
            header_height = 40
            table_header_rect = pygame.Rect(panel_x + 50, table_start_y, panel_width - 100, header_height)
            
            pygame.draw.rect(self.screen, STEEL_BLUE, table_header_rect)
            pygame.draw.rect(self.screen, NEON_BLUE, table_header_rect, 2)
            
            col_widths = [80, 200, 150, 120]
            col_x_positions = []
            current_x = panel_x + 60
            for width in col_widths:
                col_x_positions.append(current_x)
                current_x += width
            
            headers = ["RANK", "GAMER", "SCORE", "DATE"]
            for i, header in enumerate(headers):
                header_surface = self.font.render(header, True, WHITE)
                self.screen.blit(header_surface, (col_x_positions[i], table_start_y + 10))
            
            # SCROLLABLE SCORES IMPLEMENTATION
            top_scores = self.score_manager.get_top_scores(100)  # Get up to 100 scores
            row_height = 35
            
            # Calculate visible scores based on scroll position
            visible_start_index = int(self.scroll_offset)
            visible_end_index = min(visible_start_index + self.max_visible_scores, len(top_scores))
            
            # Draw scroll indicators if needed
            if len(top_scores) > self.max_visible_scores:
                # Draw scrollbar background
                scrollbar_x = panel_x + panel_width - 25
                scrollbar_y = table_start_y + header_height + 5
                scrollbar_height = self.max_visible_scores * row_height
                scrollbar_rect = pygame.Rect(scrollbar_x, scrollbar_y, 15, scrollbar_height)
                pygame.draw.rect(self.screen, TECH_GRAY, scrollbar_rect)
                pygame.draw.rect(self.screen, ELECTRIC_BLUE, scrollbar_rect, 2)
                
                # Draw scroll thumb
                if len(top_scores) > 0:
                    thumb_height = max(20, scrollbar_height * self.max_visible_scores // len(top_scores))
                    thumb_y = scrollbar_y + (scrollbar_height - thumb_height) * self.scroll_offset / max(1, len(top_scores) - self.max_visible_scores)
                    thumb_rect = pygame.Rect(scrollbar_x + 2, int(thumb_y), 11, int(thumb_height))
                    pygame.draw.rect(self.screen, NEON_GREEN, thumb_rect)
                
                # Scroll arrows/indicators
                up_arrow_alpha = 180 if self.scroll_offset > 0 else 80
                down_arrow_alpha = 180 if self.scroll_offset < len(top_scores) - self.max_visible_scores else 80
                
                # Up arrow
                up_arrow_surface = pygame.Surface((20, 20))
                up_arrow_surface.set_alpha(up_arrow_alpha)
                up_arrow_surface.fill(CYBER_ORANGE)
                up_arrow_text = self.small_font.render("⬆", True, WHITE)
                up_arrow_rect = up_arrow_text.get_rect(center=(10, 10))
                up_arrow_surface.blit(up_arrow_text, up_arrow_rect)
                self.screen.blit(up_arrow_surface, (scrollbar_x - 25, scrollbar_y - 25))
                
                # Down arrow
                down_arrow_surface = pygame.Surface((20, 20))
                down_arrow_surface.set_alpha(down_arrow_alpha)
                down_arrow_surface.fill(CYBER_ORANGE)
                down_arrow_text = self.small_font.render("⬇", True, WHITE)
                down_arrow_rect = down_arrow_text.get_rect(center=(10, 10))
                down_arrow_surface.blit(down_arrow_text, down_arrow_rect)
                self.screen.blit(down_arrow_surface, (scrollbar_x - 25, scrollbar_y + scrollbar_height + 5))
            
            # Draw visible score entries
            for i in range(visible_start_index, visible_end_index):
                if i >= len(top_scores):
                    break
                    
                score_entry = top_scores[i]
                display_index = i - visible_start_index
                row_y = table_start_y + header_height + 5 + display_index * row_height
                
                # Alternate row colors
                if i % 2 == 0:
                    row_rect = pygame.Rect(panel_x + 50, row_y, panel_width - 100, row_height)
                    pygame.draw.rect(self.screen, TECH_GRAY, row_rect)
                
                # Highlight current player's score
                row_color = NEON_GREEN if score_entry['player'] == self.player_name else WHITE
                if score_entry['player'] == self.player_name:
                    highlight_rect = pygame.Rect(panel_x + 50, row_y, panel_width - 100, row_height)
                    pygame.draw.rect(self.screen, GAMING_PURPLE, highlight_rect, 2)
                
                # Format and display score data
                rank_text = f"#{i+1}"
                gamer_text = score_entry['player'][:15]
                score_text = f"{score_entry['score']:,}"
                date_text = score_entry['date'][:10]  # Show only date part
                
                row_data = [rank_text, gamer_text, score_text, date_text]
                
                for j, data in enumerate(row_data):
                    if j < len(col_x_positions):
                        data_surface = self.font.render(str(data), True, row_color)
                        self.screen.blit(data_surface, (col_x_positions[j], row_y + 8))
            
            # Show scroll position indicator
            if len(top_scores) > self.max_visible_scores:
                scroll_info_y = table_start_y + header_height + self.max_visible_scores * row_height + 15
                total_scores = len(top_scores)
                showing_start = visible_start_index + 1
                showing_end = min(visible_end_index, total_scores)
                
                scroll_info_text = f"Showing {showing_start}-{showing_end} of {total_scores} scores"
                scroll_info_surface = self.small_font.render(scroll_info_text, True, ELECTRIC_BLUE)
                scroll_info_rect = scroll_info_surface.get_rect()
                scroll_info_x = SCREEN_WIDTH // 2 - scroll_info_rect.width // 2
                self.screen.blit(scroll_info_surface, (scroll_info_x, scroll_info_y))
            
            # Controls section
            controls_y = table_start_y + header_height + self.max_visible_scores * row_height + 50
            if controls_y < SCREEN_HEIGHT - 120:
                controls_width = 800
                controls_height = 80
                controls_x = (SCREEN_WIDTH - controls_width) // 2
                controls_rect = pygame.Rect(controls_x, controls_y, controls_width, controls_height)
                self.draw_gaming_panel(controls_rect, GAMING_PURPLE, ELECTRIC_BLUE, 4)
                
                button_y = controls_y + 20
                
                option1_text = "🔄 Press SPACE to Race Again"
                option1_surface = self.font.render(option1_text, True, NEON_GREEN)
                option1_x = controls_x + 50
                self.screen.blit(option1_surface, (option1_x, button_y))
                
                option2_text = "🆔 Press R to Restart with New Name"
                option2_surface = self.font.render(option2_text, True, CYBER_ORANGE)
                option2_x = controls_x + 50
                option2_y = button_y + 30
                self.screen.blit(option2_surface, (option2_x, option2_y))
                
                option3_text = "🚪 Press ESC to Exit Game"
                option3_surface = self.font.render(option3_text, True, RED)
                option3_x = controls_x + 450
                option3_y = button_y + 15
                self.screen.blit(option3_surface, (option3_x, option3_y))
                
        except Exception as e:
            pass

    def draw_camera_feed(self):
        """Draw camera feed in corner"""
        if self.current_frame is not None:
            try:
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (CAMERA_FEED_RADIUS * 2, CAMERA_FEED_RADIUS * 2))
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                
                feed_rect = pygame.Rect(
                    CAMERA_FEED_POS[0] - CAMERA_FEED_RADIUS - 8,
                    CAMERA_FEED_POS[1] - CAMERA_FEED_RADIUS - 8,
                    CAMERA_FEED_RADIUS * 2 + 16,
                    CAMERA_FEED_RADIUS * 2 + 16
                )
                
                self.draw_gaming_panel(feed_rect, TECH_GRAY, NEON_GREEN, 6)
                
                feed_pos = (CAMERA_FEED_POS[0] - CAMERA_FEED_RADIUS, CAMERA_FEED_POS[1] - CAMERA_FEED_RADIUS)
                self.screen.blit(frame_surface, feed_pos)
                
                hand_x, hand_y = self.hand_tracker.get_hand_position(self.current_frame)
                if hand_x is not None and hand_y is not None:
                    indicator_x = int(CAMERA_FEED_POS[0] - CAMERA_FEED_RADIUS + hand_x * CAMERA_FEED_RADIUS * 2)
                    indicator_y = int(CAMERA_FEED_POS[1] - CAMERA_FEED_RADIUS + hand_y * CAMERA_FEED_RADIUS * 2)
                    
                    indicator_pulse = math.sin(self.animation_timer * 6) * 3 + 8
                    pygame.draw.circle(self.screen, ELECTRIC_BLUE, (indicator_x, indicator_y), int(indicator_pulse))
                    pygame.draw.circle(self.screen, NEON_GREEN, (indicator_x, indicator_y), 3)
                
            except Exception as e:
                error_rect = pygame.Rect(
                    CAMERA_FEED_POS[0] - CAMERA_FEED_RADIUS - 8,
                    CAMERA_FEED_POS[1] - CAMERA_FEED_RADIUS - 8,
                    CAMERA_FEED_RADIUS * 2 + 16,
                    CAMERA_FEED_RADIUS * 2 + 16
                )
                
                self.draw_gaming_panel(error_rect, DARK_RED, RED, 6)
                
                error_text = self.small_font.render("📹 CAM", True, WHITE)
                error_rect_center = error_text.get_rect(center=CAMERA_FEED_POS)
                self.screen.blit(error_text, error_rect_center)

    def handle_events(self):
        """Handle all pygame events including scroll controls"""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # UPDATED: Stop engine sound on exit
                        self.sound_manager.stop_engine()
                        return False
                    
                    elif self.current_state == "name_input":
                        if event.key == pygame.K_BACKSPACE:
                            if self.player_name:
                                self.player_name = self.player_name[:-1]
                        elif event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                            if self.player_name.strip():
                                self.current_state = "instructions"
                        elif event.unicode.isprintable() and len(self.player_name) < 20:
                            self.player_name += event.unicode.upper()
                    
                    elif self.current_state == "instructions":
                        if event.key == pygame.K_SPACE:
                            self.start_new_game()
                    
                    elif self.current_state == "final_game_over":
                        if event.key == pygame.K_SPACE:
                            self.start_new_game()
                        elif event.key == pygame.K_r:
                            self.restart_with_new_name()
                        
                        # Enhanced: Scroll controls for leaderboard
                        elif event.key == pygame.K_UP:
                            self.handle_scroll_up()
                        elif event.key == pygame.K_DOWN:
                            self.handle_scroll_down()
                        elif event.key == pygame.K_PAGEUP:
                            # Scroll up by more entries
                            for _ in range(3):
                                self.handle_scroll_up()
                        elif event.key == pygame.K_PAGEDOWN:
                            # Scroll down by more entries
                            for _ in range(3):
                                self.handle_scroll_down()
                        elif event.key == pygame.K_HOME:
                            # Scroll to top
                            self.scroll_target = 0
                        elif event.key == pygame.K_END:
                            # Scroll to bottom
                            total_scores = len(self.score_manager.get_top_scores())
                            self.scroll_target = max(0, total_scores - self.max_visible_scores)
                
                # Enhanced: Mouse wheel support for scrolling
                elif event.type == pygame.MOUSEWHEEL:
                    if self.current_state == "final_game_over":
                        if event.y > 0:  # Scroll up
                            self.handle_scroll_up()
                        elif event.y < 0:  # Scroll down
                            self.handle_scroll_down()
                
            return True
        except Exception as e:
            return True

    def update_camera(self):
        """Update camera feed"""
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.flip(frame, 1)
        except Exception as e:
            self.current_frame = None

    def update_timers(self):
        """Update all animation timers"""
        try:
            self.animation_timer += 0.05
            self.pulse_timer += 1
            self.cursor_timer += 1
            self.scroll_animation_timer += 1
            
            if self.cursor_timer >= 30:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0
                
            if self.current_state == "playing":
                self.blink_timer += 1
                
        except Exception as e:
            pass

    def run(self):
        """UPDATED: Main game loop with engine sound support"""
        try:
            running = True
            
            while running:
                running = self.handle_events()
                if not running:
                    break
                
                self.update_camera()
                self.update_timers()
                
                if self.current_state == "playing":
                    # UPDATED: Update engine sound based on speed
                    current_speed = self.get_speed_multiplier()
                    self.sound_manager.play_engine_loop(current_speed)
                    
                    if self.current_frame is not None:
                        hand_x, _ = self.hand_tracker.get_hand_position(self.current_frame)
                        self.car.update_position(hand_x)
                    
                    # Better paced obstacle spawning
                    self.obstacle_spawn_timer += 1
                    current_spawn_delay = max(15, self.spawn_delay - int(self.get_total_game_time() // 8))
                    
                    if self.obstacle_spawn_timer >= current_spawn_delay:
                        self.spawn_obstacle_wave()
                        self.obstacle_spawn_timer = 0
                    
                    self.update_obstacles()
                
                elif self.current_state == "respawning":
                    self.update_respawn_countdown()
                
                # Rendering
                if self.current_state in ["playing", "respawning"]:
                    self.draw_road()
                    
                    for obstacle in self.obstacles:
                        obstacle.draw(self.screen)
                    
                    is_invincible = time.time() < self.invincible_until
                    self.car.draw(self.screen, is_invincible, self.blink_timer)
                    
                    self.draw_ui()
                    
                    if self.current_state == "respawning":
                        self.draw_crash_and_respawn_ui()
                    
                    self.draw_camera_feed()
                
                elif self.current_state == "name_input":
                    self.draw_name_input_screen()
                
                elif self.current_state == "instructions":
                    self.draw_instructions_screen()
                
                elif self.current_state == "final_game_over":
                    self.draw_final_game_over()
                
                pygame.display.flip()
                self.clock.tick(FPS)
                
        except Exception as e:
            pass
        finally:
            try:
                # UPDATED: Clean up sound and resources
                self.sound_manager.stop_engine()
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()
                pygame.quit()
            except Exception as cleanup_error:
                pass

def test_score_manager():
    """Test score manager functionality"""
    try:
        manager = ScoreManager()
        return True
    except Exception as e:
        return False

# Main execution
if __name__ == "__main__":
    try:
        if test_score_manager():
            game = Game()
            game.run()
        else:
            print("Component tests failed - exiting")
            
    except Exception as e:
        print(f"Failed to start game: {e}")
        print("Make sure you have all required packages installed:")
        print("pip install pygame opencv-python mediapipe numpy")
        
    finally:
        print("Thanks for playing Highway Rush! 🏁")
