import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from pathlib import Path
import warnings
import math
from tqdm import tqdm
from fractions import Fraction
import matplotlib.pyplot as plt



class Epitrochoids:
    """
    Generates and animates multiple epitrochoids formed by circles rolling
    around the *outside* of a stationary circle.

    Handles integer, rational, and irrational ratios n = R/r, adjusting
    animation duration accordingly. Also handles special cases like epicycloids,
    limaçons, roses, and circles (degenerate case h=0).
    """
    def __init__(self, R=4, d=None, n_values=None, r_values=None,
                 type='random', save_anim=False, filename=None,
                 frames_per_rev=100, fps=25,
                 max_revolutions_irrational=5):
        """
        Initializes the Epitrochoids animation setup.

        Parameters:
        -----------
        R : float
            Radius of the stationary circle (default: 4). a = R.
        d : float or list[float], optional
            Distance(s) of the tracing point from the center of the rolling circle (h = d).
            If None, defaults are chosen based on type.
            If float, applied to all rolling circles.
            If list, must match the number of rolling circles.
        n_values : int, float, Fraction, or list, optional
            Determines rolling circle radii as r = R/n. Used if r_values is None.
            Accepts integers, floats, Fractions, or lists containing these.
            Affects animation duration:
             - Integer n: Closes in 1 revolution of the center.
             - Rational n=p/q: Closes in q revolutions of the center.
             - Irrational n: Animates for max_revolutions_irrational.
            (default: [3, 2, 1] for random type)
        r_values : int, float, or list, optional
            Explicit radii for the rolling circles (b = r). Overrides n_values if provided.
            If int/float, converted to a list.
            Corresponding n values (R/r) will be calculated and used for animation duration.
        type : str, optional
            Type of curve to generate ('random', 'epicycloid', 'limacon', 'circle', 'rose').
            Defaults to 'random'. Affects default parameters and calculations.
        save_anim : bool, optional
            Whether to save the animation to a file (default: False).
        filename : str, optional
            Name of the file to save the animation (required if save_anim is True).
            Default filenames are generated based on type if None.
        frames_per_rev : int, optional
            Number of frames per 2*pi revolution of the rolling circle's center (default: 100).
            Controls animation smoothness.
        fps : int, optional
            Frames per second for the animation (default: 25). Controls speed.
        max_revolutions_irrational : float or int, optional
            Number of 2*pi revolutions of the center to simulate for irrational n values.
            (default: 5)
        """
        self.R = float(R)
        self.a = self.R  # Use 'a' for radius of fixed circle in formulas
        if self.a <= 0:
            raise ValueError("Fixed circle radius 'R' must be positive.")
        self.type = type.lower()
        self.save_anim = save_anim
        self.filename = filename
        self.frames_per_revolution = int(frames_per_rev)
        self.FPS = fps
        self.max_revolutions_irrational = float(max_revolutions_irrational)

        # Lists to store calculated parameters for each curve
        self.n_values_input_type = [] # Store original input type for n (esp Fraction)
        self.n_values_numeric = [] # The numeric ratio a/b for each curve for calcs
        self.b_values = []  # Radius of rolling circle (r)
        self.h_values = []  # Distance of tracing point (d)
        self.n_types = []   # Type of n ('integer', 'rational', 'irrational')
        self.n_revs = []    # Revolutions needed for each curve to close/complete

        self._process_parameters(d, n_values, r_values)
        self._calculate_animation_parameters()

        # Plotting attributes to be initialized later
        self.fig = None
        self.ax = None
        self.circles_rolling = []
        self.path_lines = []
        self.radius_lines = []
        self.tracing_points = []
        self.center_points = []
        self.fixed_circle = None
        self.anim = None
        self.rev_text = None 

    @staticmethod
    def _is_irrational(num, denom_threshold=1000): 
        """
        Approximate check for irrational numbers or complex rationals based on Fraction representation.
        Returns True if number seems irrational or is a rational with a large denominator.
        """
        # Handle obvious non-numeric types if they somehow sneak in
        if not isinstance(num, (int, float, Fraction)):
            return True # Treat unexpected types as irrational
        if isinstance(num, int): # Integers are rational
            return False
        if isinstance(num, Fraction): # Already a fraction
            # Check if denominator is large enough to be treated as complex/irrational for animation
            return num.denominator > denom_threshold

        # Handle Floats
        if not np.isfinite(num): # infinity or NaN
            return True

        # Check via Fraction conversion
        try:
            # Use a large limit_denominator to better represent the float
            f = Fraction(num).limit_denominator(100000) # Increased precision limit
            # Check if denominator is large after limiting (suggests irrational or complex rational)
            if f.denominator > denom_threshold:
                return True
        except (OverflowError, ValueError):
            return True # Treat conversion errors as irrational
        return False # Assume rational otherwise

    def _process_parameters(self, d, n_values_in, r_values_in):
        """
        Determines rolling circle radii (b), tracing distances (h), n values,
        and n types based on inputs and curve type.
        """
        default_n = [3, 2, 1] # Default for random if nothing else specified
        processed_n_values = [] # Store n as calculated or input
        processed_r_values = []
        r_was_irrational_flags = [] # Track if input r was likely irrational

        # --- Step 1: Determine the primary list of n or r values ---
        if r_values_in is not None:
            # Convert r_values to a list of floats
            if isinstance(r_values_in, (int, float)):
                input_r_list = [float(r_values_in)]
            else:
                input_r_list = [float(r) for r in r_values_in]

            # Calculate corresponding n_values and check r's nature
            temp_n_values = []
            valid_r_values = []
            for r in input_r_list:
                if r <= 0:
                    warnings.warn(f"Rolling radius r={r} must be positive. Skipping.")
                    continue
                if not np.isfinite(r):
                    warnings.warn(f"Rolling radius r={r} is not finite. Skipping.")
                    continue

                valid_r_values.append(r)
                n_numeric = self.a / r
                temp_n_values.append(n_numeric)
                # Check if the input r itself seems irrational/complex
                r_was_irrational_flags.append(self._is_irrational(r, denom_threshold=1000))

            processed_n_values = temp_n_values
            processed_r_values = valid_r_values
            if not processed_n_values:
                raise ValueError("No valid rolling circles could be determined from input r_values.")

        else: # Use n_values_in
            input_n_list = []
            if n_values_in is None:
                # Use defaults based on type if n_values not given
                if self.type == 'rose':
                    input_n_list = [2, 3/2, 5/3] # Example values for rose patterns
                elif self.type == 'limacon':
                    input_n_list = [1] # Limaçon requires n=1 (a=b)
                    if d is None: d = [self.R * 0.5, self.R * 1.0, self.R * 1.5] # Example distances for limaçon
                elif self.type == 'epicycloid':
                    input_n_list = default_n
                elif self.type == 'circle':
                    input_n_list = default_n
                    if d is not None: warnings.warn(f"Type is 'circle', ignoring provided 'd'. Using d=0.")
                    d = 0 # Force d=0 for circle type
                else: # Random
                    input_n_list = default_n
            elif isinstance(n_values_in, (int, float, Fraction)):
                input_n_list = [n_values_in]
            else: # It's a list
                input_n_list = list(n_values_in)

            # Validate n_values and calculate corresponding r (b) values
            temp_r_values = []
            valid_n_input_type = [] # Keep original type (like Fraction)
            valid_n_numeric = []

            for n_raw in input_n_list:
                n_numeric = None
                try:
                    # Try converting to float first for numeric checks
                    n_numeric = float(n_raw)
                except (TypeError, ValueError):
                    warnings.warn(f"Invalid type for n value: {n_raw}. Skipping.")
                    continue

                if n_numeric == 0:
                    warnings.warn(f"n=0 detected, results in infinite radius 'b'. Skipping.")
                    continue
                if not np.isfinite(n_numeric):
                    warnings.warn(f"n={n_numeric} is not finite. Skipping.")
                    continue

                if self.type == 'limacon' and abs(n_numeric - 1.0) > 1e-6:
                    warnings.warn(f"Type is 'limacon', requires n=R/r=1. Adjusting input n={n_numeric} to 1.0.")
                    n_numeric = 1.0
                    n_raw = 1.0 # Update raw value as well

                r_calc = self.a / n_numeric
                temp_r_values.append(r_calc)
                valid_n_input_type.append(n_raw) # Store original input form
                valid_n_numeric.append(n_numeric)# Store numeric form for calculations

            if not valid_n_numeric:
                raise ValueError("No valid rolling circles could be determined from input n_values.")

            processed_r_values = temp_r_values
            processed_n_values = valid_n_numeric # Store numeric values
            self.n_values_input_type = valid_n_input_type # Store input types separately

        num_circles = len(processed_r_values)
        self.b_values = processed_r_values
        self.n_values_numeric = processed_n_values # Store numeric n for calcs/type checking


        # --- Step 2: Determine n_types and revolutions needed ---
        self.n_types = []
        self.n_revs = []
        for i, n_numeric in enumerate(self.n_values_numeric): # Iterate using numeric n
            determined_n_type = "irrational" # Default assumption
            determined_revolutions = self.max_revolutions_irrational
            is_forced_irrational = False
            n_input_val = self.n_values_input_type[i] if i < len(self.n_values_input_type) else n_numeric # Get original input if available

            # Check if r was flagged as irrational (only if r_values were the input source)
            if r_values_in is not None and i < len(r_was_irrational_flags) and r_was_irrational_flags[i]:
                 print(f"Input r={self.b_values[i]:.4f} flagged as irrational/complex. Forcing n={n_numeric:.4f} to be treated as irrational.")
                 determined_n_type = "irrational"
                 determined_revolutions = self.max_revolutions_irrational
                 is_forced_irrational = True

            # Also check if n was input directly as a float likely representing irrational
            elif r_values_in is None and isinstance(n_input_val, float) and self._is_irrational(n_input_val):
                 print(f"Input n={n_input_val:.4f} flagged as irrational/complex float. Forcing treatment as irrational.")
                 determined_n_type = "irrational"
                 determined_revolutions = self.max_revolutions_irrational
                 is_forced_irrational = True

            # Only proceed with int/rational checks if not forced to irrational
            if not is_forced_irrational:
                try:
                    # Check if it's essentially an integer
                    if np.isclose(n_numeric, round(n_numeric)):
                        n_int = round(n_numeric)
                        if n_int >= 1: # Valid integer n
                            determined_n_type = "integer"
                            determined_revolutions = 1
                        else:
                           # Handle cases like n=0.5 treated as integer 0 or 1 incorrectly
                           pass # Keep default irrational if rounded n < 1
                    else:
                        # Check if it's truly rational (not too complex)
                        # Use the original input Fraction if available, otherwise check the float
                        is_rational_simple = False
                        if isinstance(n_input_val, Fraction):
                            # If input was Fraction, respect it unless denominator is huge
                            if n_input_val.denominator <= 1000: # Threshold for 'simple' rational
                                determined_n_type = "rational"
                                determined_revolutions = n_input_val.denominator
                                is_rational_simple = True
                        # If not a simple input Fraction, check the float n_numeric
                        if not is_rational_simple and not self._is_irrational(n_numeric, denom_threshold=1000):
                             # It's representable by a fraction with a reasonably small denominator
                            frac_n = Fraction(n_numeric).limit_denominator(1000) # Find that fraction
                            p = frac_n.numerator
                            q = frac_n.denominator
                            if q > 1: # Ensure it's not simplified back to integer
                                determined_n_type = "rational"
                                determined_revolutions = q
                        # else: keep default of irrational / max_revolutions

                except (ValueError, OverflowError):
                    # Error converting, treat as irrational 
                    pass

            self.n_types.append(determined_n_type)
            

            if self.type == 'circle':
                # Force revolutions to 1 for circle type
                self.n_revs.append(1)
                print(f"Type is 'circle', using n={n_numeric:.4f} (classified as {determined_n_type}) "
                      f"but forcing completion in 1 revolution.")
            else:
                # Not a circle, store the naturally determined revolutions
                self.n_revs.append(determined_revolutions)

                # Print info about n interpretation (only for non-circle types now)
                n_display = n_input_val if r_values_in is None else n_numeric
                if determined_n_type=='irrational':
                    print(
                        f"n={n_numeric:.4f} "
                        f"(Input: ~{n_display:.4f}) "
                        f"interpreted as {determined_n_type}, "
                        f"setting max revolutions to {determined_revolutions:.2f}."
                    )
                elif determined_n_type=='rational':
                    # Display fraction from input if Fraction, else format calculated fraction
                    frac_display = (n_input_val
                                    if isinstance(n_input_val, Fraction)
                                    else Fraction(n_numeric).limit_denominator(1000))
                    print(
                        f"n={float(n_numeric):.4f} "
                        f"(Input: {float(n_display):.4f}, Approx Fraction: {frac_display}) "
                        f"interpreted as {determined_n_type}, "
                        f"setting max revolutions to {determined_revolutions:.2f}."
                    )
                else: # integer
                    print(
                        f"n={n_numeric:.1f} "
                        f"interpreted as {determined_n_type}, "
                        f"requires {determined_revolutions} revolution(s)."
                    )

        # --- Step 3: Determine d_values (tracing distances 'h') ---
        final_d_values = []
        if self.type == 'epicycloid':
            final_d_values = list(self.b_values)
            if d is not None: warnings.warn(f"Type is 'epicycloid', ignoring provided 'd'. Using d=r.")
        elif self.type == 'rose':
            final_d_values = [self.a + b for b in self.b_values]
            if d is not None: warnings.warn(f"Type is 'rose', ignoring provided 'd'. Using d=R+r (a+b).")
        elif self.type == 'limacon':
            # Ensure b reflects n=1 if needed (already handled during n processing)
            # Use provided d for limacon, otherwise default
            if d is None:
                final_d_values = [self.a] * num_circles # Default to h=a (Cardioid)
                warnings.warn(f"Type is 'limacon' but 'd' not provided. Using default d=R (h=a) -> Cardioid.")
            elif isinstance(d, (int, float)):
                final_d_values = [float(d)] * num_circles
            else: # d is list
                if len(d) != num_circles:
                     # If n_values wasn't specified, maybe adjust num_circles based on d
                    if n_values_in is None and r_values_in is None:
                        num_circles = len(d)
                        # Need to resize n_values, b_values etc. assuming n=1
                        self.n_values_numeric = [1.0] * num_circles
                        self.n_values_input_type = [1.0] * num_circles
                        self.b_values = [self.a] * num_circles
                        self.n_types = ["integer"] * num_circles
                        self.n_revs = [1] * num_circles
                        # warnings.warn(f"Type is 'limacon' and 'd' is a list. Neither n_values nor r_values provided. 
                        # f"Adjusted number of curves to match length of 'd' ({num_circles}).")
                        final_d_values = [float(val) for val in d]
                    else:
                        raise ValueError(f"Length of 'd' ({len(d)}) must match the number of rolling circles ({num_circles}) "
                                         "for limacon type when 'd' is a list and n/r values were provided."
                                         )
                else:
                     final_d_values = [float(val) for val in d]

        elif self.type == 'circle':
            final_d_values = [0.0] * num_circles
        else: # Random type
            if d is None:
                final_d_values = [b * 0.5 for b in self.b_values]
                warnings.warn(f"Parameter 'd' not provided for random type. Using default d=r/2.")
            elif isinstance(d, (int, float)):
                final_d_values = [float(d)] * num_circles
            else: # d is a list
                d_float = [float(val) for val in d]
                if len(d_float) != num_circles:
                   # Adjust d list to match num_circles (repeat last or truncate)
                   if len(d_float) < num_circles:
                       warnings.warn(f"Length of 'd' ({len(d_float)}) < number of circles ({num_circles}). Repeating last 'd' value.")
                       final_d_values = d_float + [d_float[-1]] * (num_circles - len(d_float))
                   else: # len > num_circles
                       warnings.warn(f"Length of 'd' ({len(d_float)}) > number of circles ({num_circles}). Truncating 'd' list.")
                       final_d_values = d_float[:num_circles]
                else:
                   final_d_values = d_float

        # Ensure h (d) is not negative, use absolute value.
        self.h_values = [abs(val) for val in final_d_values] # Use h in formulas

        # Set default filename if needed
        if self.save_anim and self.filename is None:
            # Use numeric n for filename consistency
            n_info = "_".join([f"n{n:.3f}".replace('.','p') for n in self.n_values_numeric])
            h_info = "_".join([f"d{h:.2f}".replace('.', 'p') for h in self.h_values])
            self.filename = f"{self.type}_R{self.R:.1f}_{n_info}_{h_info}.gif"


    def _calculate_animation_parameters(self):
        """
        Calculate padding, theta range, and frame count based on n_types.
        The animation runs long enough for the longest closing pattern or
        the max duration for any irrational patterns.
        """
        # Calculate the maximum extent needed for the plot limits
        max_dist_needed = self.a # Start with the fixed circle radius

        if not self.b_values: # Handle case with no rolling circles
            self.max_padding = self.a * 1.1 if self.a > 0 else 1.0 # Default padding
        else:
            for b, h in zip(self.b_values, self.h_values):
                trace_max = self.a + b + h
                circle_edge_max = self.a + 2 * b
                current_max = max(trace_max, circle_edge_max)
                if current_max > max_dist_needed:
                    max_dist_needed = current_max
            # Add a buffer for better visualization
            self.max_padding = max_dist_needed * 1.1

        # Ensure padding is at least a small positive value
        if self.max_padding <= 0:
            self.max_padding = 1.0 # Default minimum padding

        # Determine total angle needed based on the maximum revolutions required
        if not self.n_revs: # Handle case of no valid circles
            self.max_total_revolutions = 1
        else:
            self.max_total_revolutions = max(self.n_revs) # Use the calculated revs

        print(f"Animation requires max {self.max_total_revolutions:.2f} revolutions of the center.")

        self.theta_max_pattern = 2 * np.pi * self.max_total_revolutions
        # Ensure total_frames is integer
        self.total_frames = int(np.ceil(self.max_total_revolutions * self.frames_per_revolution))
        if self.total_frames <= 0: self.total_frames = self.frames_per_revolution 

        self.theta_vals = np.linspace(0, self.theta_max_pattern, self.total_frames)

    @staticmethod
    def _epitrochoid_points(theta, a, b, h):
        """Calculate epitrochoid points for a given angle theta (or array of thetas)."""
        # Handle potential division by zero if b is zero
        if np.isclose(b, 0): # Use isclose for float comparison
            # If b=0, the "rolling" circle is a point. Its center is on the fixed circle.
            # Path of center IS the fixed circle.
            center_x = a * np.cos(theta)
            center_y = a * np.sin(theta)
            # The tracing point 'h' is relative to this center point, rotating with theta.
            # The "trace angle" for h relative to the center's position vector is 0.
            # So, h is just added radially outward/inward.
            trace_x = (a + h) * np.cos(theta) 
            trace_y = (a + h) * np.sin(theta)
            return center_x, center_y, trace_x, trace_y

        # Center of rolling circle (outside)
        center_x = (a + b) * np.cos(theta)
        center_y = (a + b) * np.sin(theta)

        # Angle for the tracing point calculation relative to the rolling circle's center rotation
        trace_angle = ((a + b) / b) * theta

        # Standard formula 
        trace_x = (a + b) * np.cos(theta) - h * np.cos(trace_angle)
        trace_y = (a + b) * np.sin(theta) - h * np.sin(trace_angle)

        return center_x, center_y, trace_x, trace_y


    def _setup_plot(self):
        """Sets up the Matplotlib figure and axes."""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
        # Use calculated limit
        limit = self.max_padding
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(False)
        title = f"Epitrochoids (R={self.R:.2f}, Type: {self.type.capitalize()})"
        self.ax.set_title(title, fontsize=14, pad=10)

        # Stationary circle
        self.fixed_circle = patches.Circle((0, 0), self.a, fill=False, color='gray', lw=1.5, ls='--')
        self.ax.add_patch(self.fixed_circle)

        # Use a colormap for distinct colors
        if len(self.b_values) == 1:
            colors = ['#FFD700'] # Single curve color (e.g., gold)
        elif len(self.b_values) <= 10: # Use perceptually uniform viridis for few colors
             colors = plt.get_cmap('viridis')(np.linspace(0.1, 0.9, len(self.b_values)))
        else: # Use rainbow for many colors, accepting potential perception issues
             colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(self.b_values)))


        # Create plot elements for each epitrochoid
        self.circles_rolling = []
        self.path_lines = []
        self.radius_lines = []
        self.tracing_points = []
        self.center_points = []

        for i, (b, h, n_num, n_typ) in enumerate(zip(self.b_values, self.h_values, self.n_values_numeric, self.n_types)):
            color = colors[i]

            # Rolling circle patch
            circle = patches.Circle((self.a + b, 0), b, fill=False, color=color, lw=1.5, zorder=3)
            self.ax.add_patch(circle)
            self.circles_rolling.append(circle)

            # Path line label generation
            n_input = self.n_values_input_type[i] if i < len(self.n_values_input_type) else n_num # Get original if possible
            if n_typ == "rational":
                 # Prefer original Fraction if input, else format the limited fraction
                frac_n = n_input if isinstance(n_input, Fraction) else Fraction(n_num).limit_denominator(1000)
                n_label = f"{frac_n.numerator}/{frac_n.denominator}"
            elif n_typ == "integer":
                 n_label = f"{int(round(n_num))}" # Display as clean integer
            else: # irrational
                 n_label = f"{n_num:.4f}" # Display float value

            label = f'n={n_label}, r={b:.3f}, d={h:.3f}' 
            path_line, = self.ax.plot([], [], '-', color=color, lw=2, label=label, zorder=2)
            self.path_lines.append(path_line)

            # Radius line (center to tracing point)
            radius_line, = self.ax.plot([], [], '--', color=color, lw=1.5, alpha=0.8, zorder=4) 
            self.radius_lines.append(radius_line)

            # Tracing point marker
            tracing_point, = self.ax.plot([], [], 'o', color=color, ms=7, zorder=5)
            self.tracing_points.append(tracing_point)

            # Rolling circle center marker
            center_point, = self.ax.plot([], [], 'o', color='white', ms=5, alpha=0.6, zorder=4) 
            self.center_points.append(center_point)

        if len(self.b_values) > 0: # Add legend only if there are curves
            self.ax.legend(loc='upper right', fontsize=9, facecolor='#1C1C1C', framealpha=0.7) # Darker legend box

        # Add text annotation for revolution counter if animation > 1 rev
        if self.max_total_revolutions > 1.01: # Add threshold to avoid showing for exactly 1 rev
            self.rev_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                         bbox=dict(facecolor='#2e4053', alpha=0.7, boxstyle='round,pad=0.3'), 
                                         color='white', fontsize=12, ha='left', va='top', zorder=10)


    def _init_animation(self):
        """Initializes animation elements."""
        all_elements = []
        for i in range(len(self.b_values)):
            # Reset lines to empty
            self.path_lines[i].set_data([], [])
            self.radius_lines[i].set_data([], [])
            self.tracing_points[i].set_data([], [])
            self.center_points[i].set_data([], [])
            # Reset circle position and radius line at theta=0
            init_positions = self._epitrochoid_points(0, self.a, self.b_values[i], self.h_values[i])
            init_center_x, init_center_y, init_trace_x, init_trace_y = init_positions
            self.circles_rolling[i].center = (init_center_x, init_center_y)
            self.tracing_points[i].set_data([init_trace_x], [init_trace_y])
            self.center_points[i].set_data([init_center_x], [init_center_y])
            self.radius_lines[i].set_data([init_center_x, init_trace_x], [init_center_y, init_trace_y])

            all_elements.extend([
                self.path_lines[i], self.circles_rolling[i], self.radius_lines[i],
                self.tracing_points[i], self.center_points[i]
            ])

        # Update the text annotation for revolutions
        if self.rev_text is not None:
             # Display total as float if it's not an integer
            total_rev_display = (f"{int(self.max_total_revolutions)}" 
                                 if np.isclose(self.max_total_revolutions, round(self.max_total_revolutions)) 
                                 else f"{self.max_total_revolutions:.2f}")
            self.rev_text.set_text(f'Revs: 0/{total_rev_display}')
            all_elements.append(self.rev_text)

        return all_elements

    def _animate_frame(self, frame):
        """Updates animation for a single frame."""
        # Current angle for this frame
        # Handle potential index out of bounds if total_frames=0 or 1
        if frame >= len(self.theta_vals):
             current_theta = self.theta_vals[-1] if len(self.theta_vals) > 0 else 0
        else:
             current_theta = self.theta_vals[frame]
        # History of angles up to this frame for drawing the path
        theta_history = self.theta_vals[:frame+1]

        plot_elements_to_update = []

        for i, (b, h) in enumerate(zip(self.b_values, self.h_values)):
            # Calculate positions for the entire history for the path line
            (centers_x_hist, centers_y_hist, 
             traces_x_hist, traces_y_hist) = self._epitrochoid_points(theta_history, self.a, b, h)

            # Get position for the current frame for markers and rolling circle
            # Use the last point in the history
            if len(traces_x_hist) > 0:
                current_center_x = centers_x_hist[-1]
                current_center_y = centers_y_hist[-1]
                current_trace_x = traces_x_hist[-1]
                current_trace_y = traces_y_hist[-1]
            else: # Frame 0 or empty history case
                (current_center_x, current_center_y, 
                current_trace_x, current_trace_y) = self._epitrochoid_points(0, self.a, b, h)

            # Update plot elements
            self.circles_rolling[i].center = (current_center_x, current_center_y)
            self.path_lines[i].set_data(traces_x_hist, traces_y_hist)
            self.tracing_points[i].set_data([current_trace_x], [current_trace_y])
            self.center_points[i].set_data([current_center_x], [current_center_y])
            self.radius_lines[i].set_data([current_center_x, current_trace_x],
                                          [current_center_y, current_trace_y])

            plot_elements_to_update.extend([
                self.path_lines[i], self.circles_rolling[i], self.radius_lines[i],
                self.tracing_points[i], self.center_points[i]
            ])

        # Update revolution counter text
        if self.rev_text is not None:
            current_rev = current_theta // (2 * np.pi)
            # Format display based on whether total revs is integer or not
            total_rev_display = (f"{int(self.max_total_revolutions)}" 
                                 if np.isclose(self.max_total_revolutions, round(self.max_total_revolutions)) 
                                 else f"{self.max_total_revolutions:.2f}")
            # Display current revs with precision
            self.rev_text.set_text(f'Revs: {int(current_rev)}/{total_rev_display}')
            plot_elements_to_update.append(self.rev_text)

        return plot_elements_to_update


    def generate_animation(self):
        """Creates and displays or saves the epitrochoid animation."""
        if not self.b_values:
            print("No valid epitrochoids to animate.")
            return None

        self._setup_plot() 
        if self.fig is None:
            raise ValueError("Figure object (self.fig) is not initialized.")


        # Create the animation object
        print(f"Generating animation with {self.total_frames} frames...")
        self.anim = FuncAnimation(self.fig, self._animate_frame, frames=self.total_frames,
                                  init_func=self._init_animation, interval=max(1, int(1000/self.FPS)), 
                                  blit=False, 
                                  repeat=False)

        # Save or show the animation
        if self.save_anim:
            if not self.filename:
                print("Error: filename must be provided when save_anim is True.")
                plt.close(self.fig)
                return None

            # Ensure filename ends with .gif or .mp4 (default to gif)
            if not self.filename.lower().endswith(('.gif', '.mp4')):
                self.filename += '.gif'

            save_dir = Path("ANIMATIONS/EPITROCHOIDS") 
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / self.filename

            print(f"Saving animation to {filepath.resolve()}...")
            writer_choice = 'pillow' if self.filename.lower().endswith('.gif') else 'ffmpeg'

            # Use tqdm for progress bar
            try:
                with tqdm(total=self.total_frames, desc="Saving Animation", unit="frame", ncols=100) as pbar:
                    def progress_update(current_frame, total_frames):
                        pbar.n = current_frame + 1 # Update tqdm progress (callback is 0-based)
                        pbar.refresh() # Force redraw

                    self.anim.save(str(filepath), writer=writer_choice, fps=self.FPS,
                                   dpi=100, progress_callback=progress_update) 
                # Ensure the progress bar finishes at 100%
                if pbar.n < self.total_frames: pbar.update(self.total_frames - pbar.n)

                print("\nAnimation saved successfully!")
            except Exception as e:
                print(f"\nError saving animation: {e}")
                print("Ensure 'ffmpeg' (for MP4) or 'Pillow' (for GIF) is installed.")
                print("If using blit=True, try setting blit=False if errors occur during save.")


            plt.close(self.fig) # Close the plot window after saving
        else:
            plt.show() # Display the animation interactively

        return self.anim # Return the animation object
    


if __name__ == "__main__":
    # --- Random Case Example ---
    print("\nGenerating Random Epitrochoids...")
    random_epi_anim = Epitrochoids(R=4, d=1, n_values=[3, 2.25, 1.75], 
                                   save_anim=False
                                   )
    # random_epi_anim.generate_animation()

    # --- Example specifying r_values directly ---
    # We can insert integer, rational or irrational values for r_values
    print("\nGenerating Epitrochoids with specified r_values...")
    custom_r_epi_anim = Epitrochoids(R=4, r_values=[1.0, 5/3, math.pi], d=[0.5, 1.5, 2.5], type='random',
                                        save_anim=True, 
                                        filename="custom_r_epitrochoids.gif", 
                                        max_revolutions_irrational=5, fps=50) # Limit irrational to 5 revolutions
    # custom_r_epi_anim.generate_animation()

    # --- Epicycloid Example (h = b) ---
    print("\nGenerating Epicycloids (h=b)...")
    # n=1 -> Cardioid, n=2 -> Nephroid
    epicycloid_anim = Epitrochoids(R=3, n_values=[3, 4, 5], type='epicycloid', 
                                   save_anim=False, 
                                   frames_per_rev=200)
    # epicycloid_anim.generate_animation()

    # Epicycloids with rational n values
    print("\nGenerating Epicycloids with rational n values...")
    epicycloid_rational_anim = Epitrochoids(R=3, n_values=[Fraction(2, 3), 3/2, 5/4], type='epicycloid', 
                                            save_anim=False, 
                                            fps=40)
    # epicycloid_rational_anim.generate_animation()

    # Epicycloids with irrational n values
    # Displaying for a single value first
    print("\nGenerating Epicycloid with with irrational n value...")
    epi_irrational_anim = Epitrochoids(R=3, n_values=math.e, type='epicycloid', 
                                    save_anim=True, 
                                    filename='epicycloid_e.mp4',
                                    max_revolutions_irrational=10, fps=50)
    # epi_irrational_anim.generate_animation()

    # Now with multiple irrational values
    print("\nGenerating Epicycloids with irrational n values...")
    epicycloid_irrational_anim = Epitrochoids(R=3, n_values=[math.sqrt(2), math.pi, math.e], type='epicycloid',
                                            save_anim=True, 
                                            max_revolutions_irrational=10,
                                            frames_per_rev=100, fps=50)
    # epicycloid_irrational_anim.generate_animation()

    # Epicycloids with mixed rational and irrational n values
    print("\nGenerating Epicycloids with mixed rational and irrational n values...")
    mixed_epi_anim = Epitrochoids(R=3, n_values=[3, 5/2, math.pi], type='epicycloid',
                                save_anim=True, 
                                filename='epicycloid_3_2p25_pi.mp4',
                                frames_per_rev=200,
                                max_revolutions_irrational=10, fps=100)
    # mixed_epi_anim.generate_animation()

    # --- Limaçon Example (a = b, n=1) ---
    print("\nGenerating Limaçons (a=b)...")
    # Vary d (h) relative to R (a). h=a=R gives Cardioid.
    limacon_anim = Epitrochoids(R=2, d=[0.5, 1, 2, 3, 4], type='limacon', 
                                save_anim=True, 
                                frames_per_rev=200) 
    # limacon_anim.generate_animation()

    # --- Rose Example (h = a + b) ---
    print("\nGenerating Rose from Epitrochoids (h=a+b)...")
    rose_anim = Epitrochoids(R=3, n_values=10, type='rose', 
                            save_anim=True, 
                            frames_per_rev=500, fps=50)
    # rose_anim.generate_animation()

    # Rose with rational n values
    print("\nGenerating Rose with rational n values...")
    rose_rational_anim = Epitrochoids(R=3, n_values=[3/2, 7/4, 11/5], type='rose', 
                                        save_anim=True, 
                                        fps=50)
    # rose_rational_anim.generate_animation()

    # --- Circle Example (h = 0)
    print("\nGenerating Circle from Epitrochoid(h = 0)") # h=0, the tracing point lies at the center of the rolling circle
    circle_anim = Epitrochoids(R=4, n_values=[2, 4, math.e], type='circle', 
                               save_anim=True
                            )
    # circle_anim.generate_animation() 

    # --- Example with R=a=0 (Should degenerate) ---
    # print("\nGenerating Epitrochoid with R=a=0 (Degenerate Case)...")
    # With a=0, formulas become x = b*cos(t)+h*cos(t), y = b*sin(t)-h*sin(t) 
    # Let's try type='random'
    # zero_r_anim = Epitrochoids(R=0, r_values=[1], d=[0.5], type='random', save_anim=False, filename="zero_R_epitrochoid.gif")
    # zero_r_anim.generate_animation()
    # Let's try type='limacon' (a=b forced to 0=0)
    # zero_r_limacon = Epitrochoids(R=0, d=[1], type='limacon', save_anim=True, filename="zero_R_limacon.gif")
    # zero_r_limacon.generate_animation()


