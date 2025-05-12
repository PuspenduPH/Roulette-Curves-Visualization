import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import patches
import os
import matplotlib.pyplot as plt



class Trochoids:
    """
    A class to generate and animate trochoids: common, curtate, and prolate.

    A trochoid is the curve traced by a point attached to a circle as it rolls
    along a straight line.
    - Common Cycloid: The point is on the circumference (distance d = r).
    - Curtate Cycloid: The point is inside the circle (distance a < r).
    - Prolate Cycloid: The point is outside the circle (distance b > r).

    Attributes:
    -----------
    r : float
        Radius of the rolling circle.
    a : float
        Distance of the tracing point from the center for a curtate cycloid (a < r).
    b : float
        Distance of the tracing point from the center for a prolate cycloid (b > r).
    num_revolutions : int
        Number of complete revolutions for the animation.
    save_anim : bool
        Flag to indicate whether to save the animation to a file.
    filename : str or None
        The name of the file to save the animation (if save_anim is True).
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    """

    def __init__(self, r=1.0, a=0.5, b=1.5, num_revolutions=2, save_anim=False, filename=None):
        """
        Initializes the Trochoids class with parameters.

        Parameters:
        -----------
        r : float, optional
            Radius of the rolling circle (default: 1.0).
        a : float, optional
            Distance parameter for curtate cycloid (default: 0.5). Must be < r.
        b : float, optional
            Distance parameter for prolate cycloid (default: 1.5). Must be > r.
        num_revolutions : int, optional
            Number of complete revolutions (default: 2).
        save_anim : bool, optional
            Whether to save the animation (default: False).
        filename : str, optional
            Name of the file to save the animation (default: None). If saving,
            a default name will be generated based on the trochoid type if None.
        """
        if not isinstance(r, (int, float)) or r <= 0:
            raise ValueError("Radius 'r' must be a positive number.")
        if not isinstance(a, (int, float)) or a <= 0:
            raise ValueError("Parameter 'a' must be a positive number.")
        if not isinstance(b, (int, float)) or b <= 0:
            raise ValueError("Parameter 'b' must be a positive number.")
        # Note: Validation a < r and b > r is done in the animate method
        #       as it depends on the type being animated.

        self.r = float(r)
        self.a = float(a)
        self.b = float(b)
        self.num_revolutions = int(num_revolutions)
        self.save_anim = bool(save_anim)
        self.filename = filename

        # Placeholders for animation elements initialized in animate()
        self.fig = None
        self.ax = None
        self.path_line = None
        self.circle_patch = None
        self.point = None
        self.radius_line = None
        self.center_point = None
        self.anim = None
        self._trochoid_type = None
        self._distance_param = None # Will be r, a, or b depending on type
        self._color = None
        self._theta_vals = None
        self._total_frames = None
        self._interval_ms = 25 # Milliseconds between frames

    def _trochoid_coordinates(self, theta, d):
        """
        Calculates the x and y coordinates of the trochoid point.
        Uses the standard parameterization:
        x = r*theta - d*sin(theta)
        y = r - d*cos(theta)

        Parameters:
        -----------
        theta : float or np.ndarray
            The angle(s) of rotation in radians.
        d : float
            The distance of the tracing point from the circle's center.

        Returns:
        --------
        tuple (np.ndarray, np.ndarray)
            x and y coordinates.
        """
        x = self.r * theta - d * np.sin(theta)
        y = self.r - d * np.cos(theta)
        return x, y

    def _init_animation(self):
        """Initializes the plot elements for the animation."""
        self.path_line.set_data([], [])
        self.circle_patch.center = (0, self.r)
        self.point.set_data([], [])
        self.radius_line.set_data([], [])
        self.center_point.set_data([], [])
        return [artist for artist in (self.path_line, self.circle_patch, self.point,
                                      self.radius_line, self.center_point) if artist is not None]

    def _update_frame(self, frame):
        """Updates the plot elements for each animation frame."""
        # Calculate points up to the current frame
        theta_current = self._theta_vals[:frame + 1]
        x_trochoid, y_trochoid = self._trochoid_coordinates(theta_current, self._distance_param)

        # Current position of the tracing point
        current_x = x_trochoid[-1]
        current_y = y_trochoid[-1]

        # Current position of the circle center
        center_x = self.r * self._theta_vals[frame]
        center_y = self.r

        # Update plot elements
        self.path_line.set_data(x_trochoid, y_trochoid)
        self.circle_patch.center = (center_x, center_y)
        self.point.set_data([current_x], [current_y])
        self.radius_line.set_data([center_x, current_x], [center_y, current_y])
        self.center_point.set_data([center_x], [center_y])

        return [artist for artist in (self.path_line, self.circle_patch, self.point,
                                      self.radius_line, self.center_point) if artist is not None]

    def animate(self, trochoid_type='common'):
        """
        Generates and displays or saves the animation for a specified trochoid type.

        Parameters:
        -----------
        trochoid_type : str, optional
            The type of trochoid to animate. Must be one of 'common',
            'curtate', or 'prolate' (case-insensitive, default: 'common').

        Returns:
        --------
        matplotlib.animation.FuncAnimation
            The animation object.

        Raises:
        -------
        ValueError
            If trochoid_type is invalid or if parameters a/b do not meet the
            requirements for the selected type (a < r for curtate, b > r for prolate).
        """
        self._trochoid_type = trochoid_type.lower()
        valid_types = ['common', 'curtate', 'prolate']
        if self._trochoid_type not in valid_types:
            raise ValueError(f"trochoid_type must be one of {valid_types}")

        # --- Parameter Setup based on type ---
        if self._trochoid_type == 'common':
            self._distance_param = self.r
            self._color = 'lime' # Changed color for distinction
            param_label = 'd=r'
        elif self._trochoid_type == 'curtate':
            if not self.a < self.r:
                raise ValueError(f"For curtate trochoid, 'a' ({self.a}) must be less than 'r' ({self.r}).")
            self._distance_param = self.a
            self._color = 'yellow'
            param_label = f'a={self.a:.2f}'
        else: # 'prolate'
            if not self.b > self.r:
                 raise ValueError(f"For prolate trochoid, 'b' ({self.b}) must be greater than 'r' ({self.r}).")
            self._distance_param = self.b
            self._color = 'cyan'
            param_label = f'b={self.b:.2f}'

        # --- Plot Setup ---
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 4)) 
        plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.1) # Adjust top/bottom for title/labels

        # Set title
        title = (f"{self._trochoid_type.capitalize()} Cycloid ({param_label})\n"
                 f"Radius r={self.r:.2f}, Revolutions={self.num_revolutions}")
        self.ax.set_title(title, fontsize=14, pad=15)

        # --- Animation Parameters ---
        frames_per_revolution = 100
        self._total_frames = int(self.num_revolutions * frames_per_revolution)
        # self._interval_ms is set in __init__

        theta_max = self.num_revolutions * 2 * np.pi
        self._theta_vals = np.linspace(0, theta_max, self._total_frames)

        # --- Calculate Plot Limits ---
        # y coordinates range from r - d to r + d
        y_min_coord = abs(self.r - self._distance_param)  # Distance from baseline to lowest point
        y_max_coord = self.r + self._distance_param       # Distance from baseline to highest point
        y_padding_up = 1.5 * max(self.r, self._distance_param)  # Increased padding for upper limit
        y_padding_down = 0.25 * self.r                    # Padding for lower limit
        x_max_coord = self.r * theta_max + max(self.r, self._distance_param) # Furthest x point approx
        x_min_coord = max(self.r, self._distance_param) # Start from -r or -d

        self.ax.set_xlim(-x_min_coord, x_max_coord)
        self.ax.set_ylim(-y_min_coord - y_padding_down, y_max_coord + y_padding_up)
        self.ax.set_aspect('equal', adjustable='box')

        # --- Axis Properties ---
        xticks = np.linspace(0, theta_max * self.r, int(2 * self.num_revolutions + 1))
        xtick_labels = []
        for x in xticks:
            val = x / (np.pi * self.r)
            if np.isclose(val, 0):
                xtick_labels.append('$0$')
            elif np.isclose(val, 1):
                 xtick_labels.append(r"$\pi r$") # Show r explicitly for clarity
            elif np.isclose(val % 1, 0):
                 xtick_labels.append(f"${int(val)}\\pi r$")
            else:
                 xtick_labels.append(f"${val:.1f}\\pi r$") # Fallback for non-integers

        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xtick_labels)
        self.ax.set_xlabel("Distance Rolled (proportional to $\\theta$)") # Use LaTeX
        self.ax.set_ylabel("Y (in units of r)")
        self.ax.grid(False) # Keep grid off for cleaner look

        # Base line
        self.ax.axhline(0, color='gray', lw=1.5, ls='--', alpha=0.6)
        # Vertical  Line
        self.ax.axvline(0, color='gray', lw=1.5, ls='--', alpha=0.6)

        # --- Initialize Plot Elements ---
        self.path_line, = self.ax.plot([], [], color=self._color, lw=2, label='Trochoid Path')
        self.circle_patch = patches.Circle((0, self.r), self.r, fill=False, color='white', lw=1.5, alpha=0.7)
        self.ax.add_patch(self.circle_patch)
        # Draw a smaller circle marker at the tracing point's distance if not common
        if self._trochoid_type != 'common':
             distance_marker = patches.Circle((0, self.r), self._distance_param, fill=False, color=self._color, lw=1, ls=':', alpha=0.5)
             self.ax.add_patch(distance_marker) # Add this marker too

        self.point, = self.ax.plot([], [], 'o', color=self._color, ms=8, label='Tracing Point')
        self.radius_line, = self.ax.plot([], [], '--', color=self._color, lw=1, alpha=0.8)
        self.center_point, = self.ax.plot([], [], 'o', color='white', ms=5, mec='black')

        self.ax.legend(loc='upper right', fontsize=8)

        # --- Create Animation ---
        self.anim = FuncAnimation(self.fig, self._update_frame, frames=self._total_frames,
                                  init_func=self._init_animation, interval=self._interval_ms,
                                  repeat=False, blit=False) 

        # --- Save or Show Animation ---
        if self.save_anim:
            # Determine filename
            fname = self.filename
            if fname is None:
                 # Generate default filename if none provided
                 fname = f"{self._trochoid_type}_trochoid_r{self.r}_d{self._distance_param:.2f}".replace('.', '_') + ".gif"
            elif not fname.lower().endswith(('.gif', '.mp4', '.mov')):
                 fname += ".gif" # Default to gif if extension missing


            save_dir = "ANIMATIONS/TROCHOIDS" # Changed directory name
            os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist
            full_path = os.path.join(save_dir, fname)

            print(f"Saving animation to {os.path.abspath(full_path)}...")
            try:
                # Use pillow for GIF, ffmpeg needed for mp4/mov (ensure installed)
                writer = 'pillow' if full_path.lower().endswith('.gif') else 'ffmpeg'
                self.anim.save(full_path, writer=writer, fps=int(1000 / self._interval_ms), dpi=150) # Adjusted dpi
                print("Animation saved successfully!")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("If saving as MP4/MOV, ensure ffmpeg is installed and in your system's PATH.")
                print("Showing animation instead...")
                plt.show() # Fallback to showing if save fails
            finally:
                 plt.close(self.fig) # Close the figure after saving or error

        else:
            plt.show() # Display the animation

        return self.anim


# --- Example Usage ---

if __name__ == "__main__":
    # Example 1: Common Cycloid (d=r)
    try:
        trochoid_common = Trochoids(r=1.0, num_revolutions=2, 
                                    save_anim=False
                                    )
        anim1 = trochoid_common.animate(trochoid_type='common') # Show
    except ValueError as e:
        print(f"Error creating common trochoid: {e}")

    # Example 2: Curtate Cycloid (a=0.5 < r=1.0), Save as GIF
    try:
        trochoid_curtate = Trochoids(r=1.0, a=0.5, num_revolutions=2, 
                                     save_anim=True, 
                                     filename="curtate_cycloid_animation.gif")
        # anim2 = trochoid_curtate.animate(trochoid_type='curtate') # Save
    except ValueError as e:
        print(f"Error creating curtate trochoid: {e}")

    # Example 3: Prolate Cycloid (b=1.5 > r=1.0), Save using default name
    try:
        trochoid_prolate = Trochoids(r=1.0, b=1.5, num_revolutions=3, 
                                     save_anim=True
                                     ) # No filename provided
        # anim3 = trochoid_prolate.animate(trochoid_type='prolate') # Save with default name
    except ValueError as e:
        print(f"Error creating prolate trochoid: {e}")

    # Example 4: Invalid parameters (a > r for curtate)
    try:
        trochoid_invalid = Trochoids(r=1.0, a=1.2)
        # anim4 = trochoid_invalid.animate(trochoid_type='curtate')
    except ValueError as e:
        print(f"\nSuccessfully caught expected error: {e}")

    # Example 5: Invalid type
    try:
        trochoid_invalid_type = Trochoids(r=1.0)
        # anim5 = trochoid_invalid_type.animate(trochoid_type='hyper')
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")