"""
MIT License

Copyright (c) 2023 zhangjiedev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

gui.py: A graphical user interface for Reeds-Shepp and Hybrid A* planning.
This file handles the user interface, drawing, and interaction. It imports
logic from `reeds_shepp.py` and configuration from `settings.ini`.

Note: The imported `reeds_shepp` and `hybrid_astar` modules are licensed
under the MIT license from the zhm-real/MotionPlanning repository.
"""

import sys
import math
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QComboBox, QPushButton, QMessageBox,
                             QInputDialog)  # Added QInputDialog
from PyQt5.QtGui import (QPainter, QBrush, QPen, QColor, QFont, QPolygonF)
from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF, QTimer
import configparser

import hybrid_astar

# --- START OF CONFIG LOADING ---
config = configparser.ConfigParser()
config.read('settings.ini')


def get_float(section, key):
    return config.getfloat(section, key)


# Define the scaling factor based on the prompt: 1 meter = 20 pixels
SCALE_FACTOR = 20.0

# Extract necessary config values (converted to float/int)
# Vehicle Dimensions
C_RF = get_float('VehicleDimensions', 'RF')
C_RB = get_float('VehicleDimensions', 'RB')
C_W = get_float('VehicleDimensions', 'W')

# Planner Resolutions
C_XY_RESO = get_float('Constants', 'XY_RESO')
C_YAW_RESO_DEG = get_float('Constants', 'YAW_RESO_DEG')

# GUI Settings
C_GRID_SIZE = config.getint('GUI', 'GRID_SIZE')
C_MIN_WINDOW_WIDTH = config.getint('GUI', 'MIN_WINDOW_WIDTH')
C_MIN_WINDOW_HEIGHT = config.getint('GUI', 'MIN_WINDOW_HEIGHT')
C_INITIAL_WINDOW_WIDTH = config.getint('GUI', 'INITIAL_WINDOW_WIDTH')
C_INITIAL_WINDOW_HEIGHT = config.getint('GUI', 'INITIAL_WINDOW_HEIGHT')

# Obstacle Settings
if 'Obstacle' in config and 'OBSTACLE_MAP_FILE' in config['Obstacle']:
    C_OBSTACLE_MAP_FILE = config.get('Obstacle', 'OBSTACLE_MAP_FILE')
else:
    C_OBSTACLE_MAP_FILE = "default_map.ini"

# Default car positions (used on startup or if load fails)
DEFAULT_CAR_STATES = [
    {
        'x': 150.0,
        'y': 100.0,
        'angle': 0.0
    },  # Car 1 (Start)
    {
        'x': 450.0,
        'y': 300.0,
        'angle': 90.0
    }  # Car 2 (Goal)
]
# --- END OF CONFIG LOADING ---


class Car:
    """Represents a car with a position, angle, and dimensions."""

    def __init__(self, x, y, angle, color):
        self.x = x
        self.y = y
        self.angle = angle
        self.color = color

        # Get dimensions from settings.ini (in meters)
        car_rf = C_RF
        car_rb = C_RB
        car_w = C_W

        # Calculate scaled dimensions (in pixels)
        total_length_m = car_rf + car_rb
        self.width = total_length_m * SCALE_FACTOR  # Scaled Length (in pixels)
        self.height = car_w * SCALE_FACTOR  # Scaled Width (in pixels)

        # Store the front and back lengths for drawing
        self.rf_scaled = car_rf * SCALE_FACTOR
        self.rb_scaled = car_rb * SCALE_FACTOR

    def contains(self, point):
        """Checks if a given point is inside the car's bounding rectangle."""
        car_rect = QRectF(self.x - self.width / 2, self.y - self.height / 2,
                          self.width, self.height)
        return car_rect.contains(point)


class CarCanvas(QWidget):
    """A custom QWidget to display and interact with the cars and path."""

    def __init__(self, cars):
        super().__init__()
        self.cars = cars
        self.animated_car = None
        self.path = None
        self.paths = []
        self.obstacles = []
        self.selected_car = None
        self.last_mouse_pos = QPoint()
        self.setMinimumSize(C_MIN_WINDOW_WIDTH, C_MIN_WINDOW_HEIGHT)
        self.setMouseTracking(True)
        self.update_gui_callback = None
        self.recalculate_path_callback = None
        self.path_info = ""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.setBrush(QBrush(QColor(230, 230, 230)))
        painter.drawRect(self.rect())

        self.draw_grid(painter)
        self.draw_obstacles(painter)

        if self.path:
            pen = QPen(QColor(30, 30, 30), 2, Qt.DashLine)
            painter.setPen(pen)
            points = [QPointF(x, y) for x, y in zip(self.path.x, self.path.y)]
            poly = QPolygonF(points)
            painter.drawPolyline(poly)
        colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255)]
        c = 0
        if self.path:
            for p in self.paths:
                pen = QPen(colors[c], 2, Qt.DashLine)
                c += 1
                painter.setPen(pen)
                points = [QPointF(x, y) for x, y in zip(p.x, p.y)]
                poly = QPolygonF(points)
                painter.drawPolyline(poly)

        all_cars_to_draw = self.cars[:]
        if self.animated_car:
            all_cars_to_draw.append(self.animated_car)

        for car in all_cars_to_draw:
            painter.save()
            painter.translate(car.x, car.y)
            painter.rotate(car.angle)

            car_body_rect = QRectF(-car.rb_scaled, -car.height / 2, car.width,
                                   car.height)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(car.color, 2))
            painter.drawRect(car_body_rect)

            painter.setBrush(QBrush(QColor(255, 255, 102)))
            light_width, light_height = 5, 8

            painter.drawRect(
                QRectF(car.rf_scaled - light_width, -car.height / 2,
                       light_width, light_height))
            painter.drawRect(
                QRectF(car.rf_scaled - light_width,
                       car.height / 2 - light_height, light_width,
                       light_height))

            painter.restore()

        if self.selected_car:
            painter.setPen(QPen(Qt.black))
            painter.setFont(QFont("Arial", 10))
            text = f"X: {self.selected_car.x:.1f}, Y: {self.selected_car.y:.1f}, Angle: {self.selected_car.angle:.1f}°"
            painter.drawText(10, 20, text)

        self.draw_path_info(painter)

    def draw_grid(self, painter):
        """Draws a grid on the canvas."""
        painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.SolidLine))
        width, height = self.width(), self.height()
        for x in range(0, width, C_GRID_SIZE):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, C_GRID_SIZE):
            painter.drawLine(0, y, width, y)

    def draw_obstacles(self, painter):
        """Draws the filled obstacle squares."""
        painter.setBrush(QBrush(QColor(100, 100, 100)))
        painter.setPen(Qt.NoPen)
        for x, y in self.obstacles:
            painter.drawRect(x, y, C_GRID_SIZE, C_GRID_SIZE)

    def draw_path_info(self, painter):
        """Draws the path information on the bottom right of the canvas."""
        if self.path_info:
            painter.setPen(QPen(Qt.black))
            painter.setFont(QFont("Arial", 10))
            x_pos = self.width() - 190
            y_pos = self.height() - 60
            text_rect = QRectF(x_pos, y_pos, 180, 50)
            painter.drawText(text_rect, Qt.AlignBottom | Qt.AlignRight,
                             self.path_info)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selected_car = None
            for car in reversed(self.cars):
                if car.contains(event.pos()):
                    self.selected_car = car
                    self.last_mouse_pos = event.pos()
                    if self.update_gui_callback:
                        self.update_gui_callback(self.cars.index(car))
                    break

            if not self.selected_car:
                grid_x = event.pos().x() // C_GRID_SIZE * C_GRID_SIZE
                grid_y = event.pos().y() // C_GRID_SIZE * C_GRID_SIZE
                obstacle_pos = (grid_x, grid_y)

                is_border = self.is_border_obstacle(obstacle_pos)

                if not is_border:
                    if obstacle_pos in self.obstacles:
                        self.obstacles.remove(obstacle_pos)
                    else:
                        self.obstacles.append(obstacle_pos)

            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.selected_car:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            self.selected_car.x += dx
            self.selected_car.y += dy
            self.last_mouse_pos = event.pos()
            if self.update_gui_callback:
                self.update_gui_callback(self.cars.index(self.selected_car))
            self.update()

    def mouseReleaseEvent(self, event):
        self.selected_car = None

    def is_border_obstacle(self, pos):
        """Checks if a given (x, y) tuple is a position reserved for the border."""
        x, y = pos
        width, height = self.width(), self.height()
        grid = C_GRID_SIZE

        x_squares = width // grid
        y_squares = height // grid

        is_x_border = x == 0 or x == x_squares * grid
        is_y_border = y == 0 or y == y_squares * grid

        return is_x_border or is_y_border

    def initialize_border_obstacles(self):
        """Initializes a permanent border of obstacles around the canvas edge."""
        original_obstacles = self.obstacles[:]

        # Filter out existing border obstacles before clearing
        non_border_obstacles = [
            obs for obs in original_obstacles
            if not self.is_border_obstacle(obs)
        ]

        self.obstacles.clear()

        width, height = self.width(), self.height()
        grid = C_GRID_SIZE

        x_squares = width // grid
        y_squares = height // grid

        for i in range(x_squares + 1):
            x_pos = i * grid
            self.obstacles.append((x_pos, 0))
            self.obstacles.append((x_pos, y_squares * grid))

        for j in range(1, y_squares):
            y_pos = j * grid
            self.obstacles.append((0, y_pos))
            self.obstacles.append((x_squares * grid, y_pos))

        self.obstacles.extend(non_border_obstacles)

        self.obstacles = sorted(list(set(self.obstacles)))


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reeds Shepp Parking Planner")

        # Initial car positions loaded from defaults
        self.cars = [
            Car(DEFAULT_CAR_STATES[0]['x'],
                DEFAULT_CAR_STATES[0]['y'], DEFAULT_CAR_STATES[0]['angle'],
                QColor(0, 102, 204)),  # Car 1 (Start)
            Car(DEFAULT_CAR_STATES[1]['x'],
                DEFAULT_CAR_STATES[1]['y'], DEFAULT_CAR_STATES[1]['angle'],
                QColor(204, 0, 0))  # Car 2 (Goal)
        ]
        self.current_car_index = 0

        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.car_canvas = CarCanvas(self.cars)
        self.car_canvas.update_gui_callback = self.update_control_gui
        self.car_canvas.recalculate_path_callback = self.calculate_path
        main_layout.addWidget(self.car_canvas, 1)

        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)
        control_group.setMaximumWidth(200)

        # Car selector
        self.car_selector = QComboBox()
        self.car_selector.addItem("Car 1 (Start)")
        self.car_selector.addItem("Car 2 (Goal)")
        self.car_selector.currentIndexChanged.connect(self.select_car)
        control_layout.addWidget(self.car_selector)

        # Position and Angle controls
        self.x_spinbox = QSpinBox(self,
                                  minimum=0,
                                  maximum=2000,
                                  singleStep=5,
                                  prefix="X: ")
        self.x_spinbox.valueChanged.connect(self.set_x_pos)
        self.y_spinbox = QSpinBox(self,
                                  minimum=0,
                                  maximum=2000,
                                  singleStep=5,
                                  prefix="Y: ")
        self.y_spinbox.valueChanged.connect(self.set_y_pos)
        self.angle_spinbox = QDoubleSpinBox(self,
                                            minimum=0,
                                            maximum=360,
                                            singleStep=5,
                                            decimals=1,
                                            suffix=" °")
        self.angle_spinbox.setWrapping(True)
        self.angle_spinbox.valueChanged.connect(self.set_angle)

        control_layout.addWidget(self.x_spinbox)
        control_layout.addWidget(self.y_spinbox)
        control_layout.addWidget(self.angle_spinbox)

        # Path calculation and animation controls
        self.calc_button = QPushButton("Calculate Path")
        self.calc_button.clicked.connect(self.calculate_path)
        control_layout.addWidget(self.calc_button)

        self.animate_button = QPushButton("Animate Path")
        self.animate_button.clicked.connect(self.start_animation)
        control_layout.addWidget(self.animate_button)

        self.calc_k_button = QPushButton("Multiple Paths")
        self.calc_k_button.clicked.connect(self.calculate_k_paths)
        control_layout.addWidget(self.calc_k_button)

        self.clear_button = QPushButton("Clear Path")
        self.clear_button.clicked.connect(self.clear_path)
        control_layout.addWidget(self.clear_button)

        # Obstacle Map Save/Load Controls
        map_group = QGroupBox("Obstacle Map & Car State")
        map_layout = QVBoxLayout(map_group)

        self.save_map_button = QPushButton("Save Map As...")
        self.save_map_button.clicked.connect(self.save_obstacle_map)
        map_layout.addWidget(self.save_map_button)

        self.load_map_button = QPushButton("Load Map From...")
        self.load_map_button.clicked.connect(
            lambda: self.load_obstacle_map(initial_load=False))
        map_layout.addWidget(self.load_map_button)

        control_layout.addWidget(map_group)

        # Animation speed control
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed (ms):")
        self.speed_spinbox = QSpinBox(self,
                                      minimum=1,
                                      maximum=100,
                                      singleStep=5,
                                      value=20)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_spinbox)
        control_layout.addLayout(speed_layout)

        # New section for path information
        path_info_group = QGroupBox("Path Information")
        path_info_layout = QVBoxLayout(path_info_group)

        # Static Path Info (Best Path)
        self.path_cost_static_label = QLabel(
            "Total Cost: N/A")  # Renamed/repurposed old label
        self.path_length_static_label = QLabel(
            "Length: N/A")  # Renamed/repurposed old label
        path_info_layout.addWidget(QLabel("--- Total Path Info ---"))
        path_info_layout.addWidget(self.path_cost_static_label)
        path_info_layout.addWidget(self.path_length_static_label)

        # Real-time Animation Info
        self.path_x_label = QLabel("X: N/A")
        self.path_y_label = QLabel("Y: N/A")
        self.path_yaw_label = QLabel("Yaw: N/A")
        self.path_direction_label = QLabel("Dir: N/A")

        path_info_layout.addWidget(QLabel("--- Animated State ---"))
        path_info_layout.addWidget(self.path_x_label)
        path_info_layout.addWidget(self.path_y_label)
        path_info_layout.addWidget(self.path_yaw_label)
        path_info_layout.addWidget(self.path_direction_label)

        control_layout.addWidget(path_info_group)

        control_layout.addStretch()
        main_layout.addWidget(control_group)
        self.update_control_gui(self.current_car_index)

        # Animation timer and state
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.current_step = 0
        self.total_steps = 0

        # Initial load using the default file name from config
        self.load_obstacle_map(initial_load=True, map_name=C_OBSTACLE_MAP_FILE)

    def ensure_maps_directory(self):
        """Creates the 'maps' directory if it does not exist."""
        maps_dir = "maps"
        if not os.path.exists(maps_dir):
            try:
                os.makedirs(maps_dir)
            except OSError as e:
                # Handle potential permission or other OS errors
                QMessageBox.critical(
                    self, "Directory Error",
                    f"Failed to create directory 'maps/': {e}")

    def prompt_map_name(self, action):
        """Prompts the user for a map name (max 8 chars) and returns the full filename."""
        default_name = C_OBSTACLE_MAP_FILE.split(
            '.')[0] if '.' in C_OBSTACLE_MAP_FILE else C_OBSTACLE_MAP_FILE

        text, ok = QInputDialog.getText(self,
                                        f"{action} Map",
                                        f"Enter map name (max 8 chars):",
                                        text=default_name)

        if ok and text:
            # Enforce max 8 characters and append .ini extension
            safe_name = text[:8].strip()
            if not safe_name:
                QMessageBox.warning(self, "Invalid Name",
                                    "Map name cannot be empty.")
                return None
            return f"{safe_name}.ini"
        return None

    def load_obstacle_map(self, initial_load=False, map_name=None):
        """Loads obstacle and car data from the specified map file."""

        # 1. Get map name from user or use default/startup name
        if not initial_load:
            map_file = self.prompt_map_name("Load")
            if not map_file:
                return  # User cancelled
        elif map_name:
            map_file = map_name
        else:
            map_file = C_OBSTACLE_MAP_FILE  # Fallback

        map_file = os.path.join("maps", map_file)

        map_config = configparser.ConfigParser()
        loaded_car_states = None

        # 2. Try to load the map file
        if os.path.exists(map_file):
            try:
                map_config.read(map_file)
                self.car_canvas.obstacles = []  # Clear current

                # Load Obstacles
                if 'Obstacles' in map_config:
                    for key, val in map_config.items('Obstacles'):
                        try:
                            x_str, y_str = key.split('_')
                            x = int(x_str) * C_GRID_SIZE
                            y = int(y_str) * C_GRID_SIZE
                            self.car_canvas.obstacles.append((x, y))
                        except ValueError:
                            pass  # Skip malformed entries

                # Load Car States
                if 'CarStates' in map_config:
                    try:
                        loaded_car_states = [{
                            'x':
                                map_config.getfloat('CarStates', 'Car1_X'),
                            'y':
                                map_config.getfloat('CarStates', 'Car1_Y'),
                            'angle':
                                map_config.getfloat('CarStates', 'Car1_Angle')
                        }, {
                            'x':
                                map_config.getfloat('CarStates', 'Car2_X'),
                            'y':
                                map_config.getfloat('CarStates', 'Car2_Y'),
                            'angle':
                                map_config.getfloat('CarStates', 'Car2_Angle')
                        }]
                    except configparser.NoOptionError:
                        pass  # Car data missing

                if not initial_load:
                    QMessageBox.information(
                        self, "Map Loaded",
                        f"Map and car states loaded from {map_file}.")

            except Exception as e:
                QMessageBox.critical(
                    self, "Load Error",
                    f"Failed to read map file {map_file}. Using defaults: {e}")
        else:
            if not initial_load:
                QMessageBox.warning(
                    self, "Load Warning",
                    f"Map file {map_file} not found. Using current or default car positions and border obstacles."
                )

        # 3. Apply Car States
        if loaded_car_states:
            self.cars[0].x = loaded_car_states[0]['x']
            self.cars[0].y = loaded_car_states[0]['y']
            self.cars[0].angle = loaded_car_states[0]['angle']

            self.cars[1].x = loaded_car_states[1]['x']
            self.cars[1].y = loaded_car_states[1]['y']
            self.cars[1].angle = loaded_car_states[1]['angle']
        else:
            # If load failed or no car data, set to hardcoded defaults (useful on startup if file doesn't exist)
            self.cars[0].x = DEFAULT_CAR_STATES[0]['x']
            self.cars[0].y = DEFAULT_CAR_STATES[0]['y']
            self.cars[0].angle = DEFAULT_CAR_STATES[0]['angle']
            self.cars[1].x = DEFAULT_CAR_STATES[1]['x']
            self.cars[1].y = DEFAULT_CAR_STATES[1]['y']
            self.cars[1].angle = DEFAULT_CAR_STATES[1]['angle']

        # 4. Initialize border and update GUI
        self.car_canvas.initialize_border_obstacles()
        self.update_control_gui(self.current_car_index)
        self.clear_path()  # Clear any old path
        self.car_canvas.update()

    def save_obstacle_map(self):
        """Saves current non-border obstacle and car data to a user-specified INI file."""
        map_file = self.prompt_map_name("Save")
        if not map_file:
            return

        map_file = os.path.join("maps", map_file)

        map_config = configparser.ConfigParser()

        map_config.add_section('Obstacles')
        for x, y in self.car_canvas.obstacles:
            if not self.car_canvas.is_border_obstacle((x, y)):
                x_grid_idx = x // C_GRID_SIZE
                y_grid_idx = y // C_GRID_SIZE
                key = f"{x_grid_idx}_{y_grid_idx}"
                map_config.set('Obstacles', key, '1')

        map_config.add_section('CarStates')
        # Car 1
        map_config.set('CarStates', 'Car1_X', str(self.cars[0].x))
        map_config.set('CarStates', 'Car1_Y', str(self.cars[0].y))
        map_config.set('CarStates', 'Car1_Angle', str(self.cars[0].angle))
        # Car 2
        map_config.set('CarStates', 'Car2_X', str(self.cars[1].x))
        map_config.set('CarStates', 'Car2_Y', str(self.cars[1].y))
        map_config.set('CarStates', 'Car2_Angle', str(self.cars[1].angle))

        try:
            with open(map_file, 'w') as configfile:
                map_config.write(configfile)
            QMessageBox.information(
                self, "Map Saved",
                f"Map and car states saved successfully to {map_file}.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error",
                                 f"Failed to write map file {map_file}: {e}")

    def get_planning_params(self):
        self.clear_path()
        start_car, goal_car = self.cars[0], self.cars[1]

        # Convert car states for the path planner
        sx, sy = start_car.x / SCALE_FACTOR, start_car.y / SCALE_FACTOR
        syaw = math.radians(start_car.angle)
        gx, gy = goal_car.x / SCALE_FACTOR, goal_car.y / SCALE_FACTOR
        gyaw = math.radians(goal_car.angle)

        # Extract obstacle coordinates
        ox_pixels = [obs[0] for obs in self.car_canvas.obstacles]
        oy_pixels = [obs[1] for obs in self.car_canvas.obstacles]

        # Obstacle coordinates in meters (using the center of the grid cell)
        grid_half_size_m = (C_GRID_SIZE / 2) / SCALE_FACTOR
        ox = [(p / SCALE_FACTOR) + grid_half_size_m for p in ox_pixels]
        oy = [(p / SCALE_FACTOR) + grid_half_size_m for p in oy_pixels]

        # Define resolutions
        XY_RESO = C_XY_RESO
        YAW_RESO = math.radians(C_YAW_RESO_DEG)

        return sx, sy, syaw, gx, gy, gyaw, ox, oy, XY_RESO, YAW_RESO

    def calculate_path(self):
        """Gets car states, calculates the path, and updates the canvas."""
        sx, sy, syaw, gx, gy, gyaw, ox, oy, XY_RESO, YAW_RESO = self.get_planning_params(
        )

        # The function call
        path = hybrid_astar.hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw,
                                                  ox, oy, XY_RESO, YAW_RESO)

        if path:
            # Convert path points back to pixels for drawing: pixels = meters * SCALE_FACTOR
            path.x = [val * SCALE_FACTOR for val in path.x]
            path.y = [val * SCALE_FACTOR for val in path.y]

            self.car_canvas.path = path
            self.update_path_info()
        else:
            print("Path search failed!")

        self.car_canvas.update()

    def calculate_k_paths(self):
        """Calculates up to 3 paths using hybrid_astar_planning_k_paths."""
        self.clear_path()
        sx, sy, syaw, gx, gy, gyaw, ox, oy, XY_RESO, YAW_RESO = self.get_planning_params(
        )

        K_MAX = 3  # Maximum number of paths to calculate

        paths = hybrid_astar.hybrid_astar_planning_k_paths(sx,
                                                           sy,
                                                           syaw,
                                                           gx,
                                                           gy,
                                                           gyaw,
                                                           ox,
                                                           oy,
                                                           XY_RESO,
                                                           YAW_RESO,
                                                           k_max=K_MAX)

        if paths:
            # Convert path points back to pixels for drawing for ALL paths
            for path in paths:
                path.x = [val * SCALE_FACTOR for val in path.x]
                path.y = [val * SCALE_FACTOR for val in path.y]

            self.car_canvas.paths = paths
            self.car_canvas.path = paths[-1]

            self.update_path_info()
        else:
            print("Path search failed!")

        self.car_canvas.update()

    def start_animation(self):
        """Starts the animation of the car along the calculated path."""
        if not self.car_canvas.path:
            return

        start_car = self.cars[0]
        self.car_canvas.animated_car = Car(start_car.x, start_car.y,
                                           start_car.angle, start_car.color)

        self.animation_timer.stop()
        self.current_step = 0
        self.total_steps = len(self.car_canvas.path.x)
        self.animation_timer.start(self.speed_spinbox.value())
        self.update_path_info()

    def update_animation(self):
        """Updates the animated car's position to the next point on the path."""
        if self.current_step < self.total_steps and self.car_canvas.animated_car:

            # Block timer signals to prevent re-entrancy during long updates (good practice)
            self.animation_timer.blockSignals(True)

            path = self.car_canvas.path
            animated_car = self.car_canvas.animated_car

            animated_car.x = path.x[self.current_step]
            animated_car.y = path.y[self.current_step]

            animated_car.angle = math.degrees(path.yaw[self.current_step])

            # MODIFICATION START: Update real-time path info
            # Convert back to meters for display (pixels / SCALE_FACTOR)
            self.path_x_label.setText(
                f"X: {animated_car.x / SCALE_FACTOR:.2f} m")
            self.path_y_label.setText(
                f"Y: {animated_car.y / SCALE_FACTOR:.2f} m")
            self.path_yaw_label.setText(f"Yaw: {animated_car.angle:.1f}°")

            # Note: The Path object stores direction for *each point* in path.direction
            current_direction = path.direction[self.current_step]
            self.path_direction_label.setText(
                f"Dir: {'Fwd' if current_direction > 0 else 'Bwd'}")

            self.car_canvas.update()

            self.current_step += 1

            # Unblock timer signals
            self.animation_timer.blockSignals(False)
        else:
            self.animation_timer.stop()
            self.car_canvas.animated_car = None
            self.car_canvas.update()

    def clear_path(self):
        """Clears the path and resets the animation."""
        self.animation_timer.stop()
        self.car_canvas.path = None
        self.car_canvas.paths = []
        self.car_canvas.animated_car = None
        self.car_canvas.path_info = ""
        self.update_path_info()
        self.car_canvas.update()

    def update_path_info(self):
        """Updates the path info labels and the canvas text."""
        path = self.car_canvas.path

        # Update Static (Total Path) Info
        if path:
            L_approx = sum(
                math.hypot(path.x[i + 1] - path.x[i], path.y[i + 1] - path.y[i])
                for i in range(len(path.x) - 1))
            L_approx_m = L_approx / SCALE_FACTOR

            self.path_length_static_label.setText(
                f"Length: {L_approx_m:.2f} m (Approx)")
            self.path_cost_static_label.setText(f"Total Cost: {path.cost:.2f}")

            self.car_canvas.path_info = (
                f"Length: {L_approx_m:.2f} m (Approx)\n"
                f"Cost: {path.cost:.2f}")
        else:
            self.path_length_static_label.setText("Length: N/A")
            self.path_cost_static_label.setText("Total Cost: N/A")
            self.car_canvas.path_info = ""

        # Clear Dynamic (Animated State) Info
        self.path_x_label.setText("X: N/A")
        self.path_y_label.setText("Y: N/A")
        self.path_yaw_label.setText("Yaw: N/A")
        self.path_direction_label.setText("Dir: N/A")

    def select_car(self, index):
        self.current_car_index = index
        self.car_canvas.selected_car = self.cars[index]
        self.update_control_gui(index)
        self.car_canvas.update()

    def set_x_pos(self, value):
        self.cars[self.current_car_index].x = value
        self.clear_path()
        self.car_canvas.update()

    def set_y_pos(self, value):
        self.cars[self.current_car_index].y = value
        self.clear_path()
        self.car_canvas.update()

    def set_angle(self, value):
        self.cars[self.current_car_index].angle = value
        self.clear_path()
        self.car_canvas.update()

    def update_control_gui(self, car_index):
        """Updates the control panel to reflect the selected car's state."""
        car = self.cars[car_index]
        self.car_selector.setCurrentIndex(car_index)

        self.x_spinbox.blockSignals(True)
        self.x_spinbox.setValue(int(car.x))
        self.x_spinbox.blockSignals(False)

        self.y_spinbox.blockSignals(True)
        self.y_spinbox.setValue(int(car.y))
        self.y_spinbox.blockSignals(False)

        self.angle_spinbox.blockSignals(True)
        self.angle_spinbox.setValue(car.angle)
        self.angle_spinbox.blockSignals(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, C_INITIAL_WINDOW_WIDTH,
                       C_INITIAL_WINDOW_HEIGHT)
    window.show()
    sys.exit(app.exec_())
