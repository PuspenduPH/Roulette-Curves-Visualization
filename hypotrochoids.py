import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from pathlib import Path
import warnings
from tqdm import tqdm
from fractions import Fraction
import math
import matplotlib.pyplot as plt



class Hypotrochoids:
    """
    Generates and animates multiple hypotrochoids formed by circles rolling
    inside a larger stationary circle.

    Handles integer, rational, and irrational ratios n = R/r, adjusting
    animation duration accordingly. Also handles special cases like
    hypocycloids, rose curves, and ellipses.
    """
    def __init__(self, R=4, d=None, n_values=None, r_values=None,
                 type='random', save_anim=False, filename=None,
                 frames_per_rev=100, fps=25,
                 max_revolutions_irrational=5): 
        """
        Initializes the Hypotrochoids animation setup.

        Parameters:
        -----------
        R : float
            Radius of the stationary circle (default: 4). Also called 'a'.
        d : float or list[float], optional
            Distance(s) of the tracing point from the center of the rolling circle.
            Also called 'h'. If None, defaults are chosen based on type.
            If float, applied to all rolling circles.
            If list, must match the number of rolling circles.
        n_values : int, float, Fraction, or list, optional
            Determines rolling circle radii as r = R/n. Used if r_values is None.
            Accepts integers, floats, Fractions, or lists containing these.
            Affects animation duration:
             - Integer n: Closes in 1 revolution of the center.
             - Rational n=p/q: Closes in q revolutions of the center.
             - Irrational n: Animates for max_revolutions_irrational.
            (default: [3, 4, 5] for random type)
        r_values : int, float, or list, optional
            Explicit radii ('b' or 'r') for the rolling circles. Overrides n_values if provided.
            If int/float, converted to a list.
            Corresponding n values (R/r) will be calculated and used for animation duration.
        type : str, optional
            Type of curve to generate ('random', 'hypocycloid', 'rose', 'ellipse').
            Defaults to 'random'. Affects default parameters and calculations if d, n_values, r_values are None.
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
    def _is_irrational(num, denom_threshold=100):
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
        if num == math.pi or num == math.e: # Known irrationals
            return True
        # Check via Fraction conversion
        try:
            f = Fraction(num).limit_denominator(10000) # Limit complexity
            # Check if denominator is large after limiting (suggests irrational or complex rational)
            if f.denominator > denom_threshold:
                return True
        except (OverflowError, ValueError):
            return True # Treat conversion errors as irrational
        return False # Assume rational otherwise

    def _process_parameters(self, d, n_values_in, r_values_in):
        """
        Determines rolling circle radii (b), tracing distances (h), n values,
        and n types based on inputs and curve type. Handles integer, rational,
        and irrational r_values similarly to n_values.
        """
        default_n = [3, 4, 5] # Default for random if nothing else specified
        processed_n_values = [] # Store numeric n for calculations
        processed_r_values = []
        n_values_input_type = [] # Store original input type for n (esp Fraction)
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
                # Allow r > R for hypotrochoids (rolls internally, n < 1)
                if r > self.a and self.type != 'ellipse': # Allow r=a/2 for ellipse
                    warnings.warn(f"Rolling radius r={r:.3f} > R={self.a:.3f} (n<1). "
                                  "Rolling circle is larger than fixed circle. Ensure this is intended for internal rolling."
                                  )
                if self.type == 'ellipse' and abs(self.a / r - 2.0) > 1e-6:
                    warnings.warn(f"Type is 'ellipse', requires R/r = 2. Adjusting r={r:.3f} to {self.a/2.0:.3f}.")
                    r = self.a / 2.0 # Correct r for ellipse type

                valid_r_values.append(r)
                n_numeric = self.a / r
                temp_n_values.append(n_numeric)
                # Check if the input r itself seems irrational/complex
                r_was_irrational_flags.append(self._is_irrational(r, denom_threshold=1000)) 

            processed_n_values = temp_n_values
            processed_r_values = valid_r_values
            if not processed_n_values:
                raise ValueError("No valid rolling circles could be determined from input r_values.")
            # Since r was input, we don't have an original 'n' input type list yet
            # We will create a placeholder or use numeric n later if needed for display
            n_values_input_type = processed_n_values # Use numeric n as stand-in for 'input type'

        else: # Use n_values_in
            input_n_list = []
            if n_values_in is None:
                # Use defaults based on type if n_values not given
                if self.type == 'ellipse':
                     # Special handling if 'd' is a list but n is not - generate multiple ellipses
                     if d is not None and isinstance(d, (list, tuple)):
                          input_n_list = [2.0] * len(d)
                     else:
                          input_n_list = [2.0] # Ellipse requires n=2 (a=2b)
                     if d is None: d = [self.R * 0.25, self.R * 0.5] # Example distances for ellipse
                elif self.type == 'rose':
                     input_n_list = [3, 5/2, 7/3] # Example values for rose patterns
                elif self.type == 'hypocycloid':
                     input_n_list = default_n
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
                # Allow n < 1 for hypotrochoids (internal rolling)
                if n_numeric < 1 and self.type != 'ellipse': # n=2 for ellipse is >= 1
                    warnings.warn(f"n={n_numeric:.3f} < 1 detected (r > R). "
                                  "Rolling circle is larger than fixed circle. Ensure this is intended for internal rolling."
                                  )

                if self.type == 'ellipse' and abs(n_numeric - 2.0) > 1e-6:
                    warnings.warn(f"Type is 'ellipse', requires n=R/r=2. Adjusting input n={n_numeric:.3f} to 2.0.")
                    n_numeric = 2.0
                    n_raw = 2.0 # Update raw value as well

                r_calc = self.a / n_numeric
                temp_r_values.append(r_calc)
                valid_n_input_type.append(n_raw) # Store original input form
                valid_n_numeric.append(n_numeric)# Store numeric form for calculations

            if not valid_n_numeric:
                raise ValueError("No valid rolling circles could be determined from input n_values.")

            processed_r_values = temp_r_values
            processed_n_values = valid_n_numeric # Store numeric values
            n_values_input_type = valid_n_input_type # Store input types separately

        num_circles = len(processed_r_values)
        self.b_values = processed_r_values
        # Ensure these attributes exist before assigning
        self.n_values_numeric = processed_n_values # Store numeric n for calcs/type checking
        self.n_values_input_type = n_values_input_type # Store input types separately


        # --- Step 2: Determine n_types and revolutions needed ---
        self.n_types = []
        self.n_revs = []
        # Use the numeric n values for determining type and revolutions
        for i, n_numeric in enumerate(self.n_values_numeric):
            n_type = "irrational" # Default assumption
            revolutions = self.max_revolutions_irrational
            is_forced_irrational = False
            # Get original input n if it exists (i.e., if n_values were input)
            n_input_val = self.n_values_input_type[i] if r_values_in is None else n_numeric

            # Check if r was flagged as irrational (only if r_values were the input source)
            if r_values_in is not None and i < len(r_was_irrational_flags) and r_was_irrational_flags[i]:
                 print(f"Input r={self.b_values[i]:.4f} flagged as irrational/complex. "
                       f"Forcing n={n_numeric:.4f} to be treated as irrational."
                       )
                 n_type = "irrational"
                 revolutions = self.max_revolutions_irrational
                 is_forced_irrational = True
            # Check if n was input directly as a float likely representing irrational
            elif r_values_in is None and isinstance(n_input_val, float) and self._is_irrational(n_input_val, denom_threshold=1000):
                 print(f"Input n={n_input_val:.4f} flagged as irrational/complex float. Forcing treatment as irrational.")
                 n_type = "irrational"
                 revolutions = self.max_revolutions_irrational
                 is_forced_irrational = True

            # Only proceed with int/rational checks if not forced to irrational
            if not is_forced_irrational:
                try:
                    # Check if it's essentially an integer
                    if np.isclose(n_numeric, round(n_numeric)):
                        n_int = round(n_numeric)
                        # For hypotrochoid, n=1 means b=a, rolling circle fills fixed circle - edge case.
                        # Let's treat n=1 as integer case needing 1 rev.
                        if n_int >= 1: # Valid integer n (or n=1 edge case)
                            n_type = "integer"
                            revolutions = 1
                        else:
                            # Handle cases like n=0.5 treated as integer 0 or 1 incorrectly
                            # Or n < 1 that are not close to 0 or 1. Treat as rational/irrational.
                            pass # Keep default irrational if rounded n < 1

                    # ONLY check for rational IF it wasn't classified as integer
                    elif not self._is_irrational(n_numeric, denom_threshold=1000):
                        is_rational_simple = False
                        # Check if input was explicitly a Fraction
                        if r_values_in is None and isinstance(n_input_val, Fraction):
                             # Check if it's a non-integer Fraction (denominator > 1)
                             if n_input_val.denominator > 1 and n_input_val.denominator <= 1000:
                                 n_type = "rational"
                                 revolutions = n_input_val.denominator
                                 is_rational_simple = True
                             # If input was Fraction(X,1), treat as integer
                             elif n_input_val.denominator == 1:
                                 # It was already classified as integer above, but confirm revs=1
                                 n_type = "integer"
                                 revolutions = 1
                                 is_rational_simple = True # Mark as handled

                        # If not handled as an input Fraction, check the numeric value
                        if not is_rational_simple:
                            frac_n = Fraction(n_numeric).limit_denominator(1000)
                            p = frac_n.numerator
                            q = frac_n.denominator
                            # Check if it's a non-integer rational (denominator > 1)
                            if q > 1:
                                 n_type = "rational"
                                 revolutions = q
                            # If q == 1, it was already handled by the integer check above.
                            # If q <= 0, something is wrong, keep default irrational.
                    # else: If neither integer nor simple rational, it remains irrational (the default)

                except (ValueError, OverflowError):
                    # Error converting, treat as irrational
                    n_type = "irrational"
                    revolutions = self.max_revolutions_irrational

            self.n_types.append(n_type)
            self.n_revs.append(revolutions)

            # Print info of n interpretation            
            n_display = n_input_val if r_values_in is None else n_numeric # Use input n if available else numeric
    
            if n_type == 'irrational':
                # Display original input if it was float, otherwise show numeric
                input_display_str = f"{float(n_display):.4f}" if isinstance(n_display, (float, np.floating)) else str(n_display)
                print(
                    f"n={n_numeric:.4f} "
                    f"(Input: ~{input_display_str}) "
                    f"interpreted as {n_type}, "
                    f"setting max revolutions to {revolutions:.2f}."
                )
            elif n_type=='rational':
                # For rational numbers, always display the fraction form clearly
                frac_n = Fraction(n_numeric).limit_denominator(1000)
                # Display original input if it was Fraction, otherwise show numeric
                input_display_str = str(n_input_val) if isinstance(n_input_val, Fraction) else f"{float(n_display):.4f}"
                print(
                    f"n={float(n_numeric):.4f} "
                    f"(Input: {input_display_str}, Approx Fraction: {frac_n.numerator}/{frac_n.denominator}) "
                    f"interpreted as {n_type}, "
                    f"setting max revolutions to {int(revolutions)}." # Revolutions for rational are integer
                )
            else: # integer
                # Display original input if it was int/Fraction(X,1), otherwise show numeric
                input_display_str = str(n_input_val) if isinstance(n_input_val, (int, Fraction)) else f"{float(n_display):.4f}"
                print(
                    f"n={int(n_numeric)} "
                    f"(Input: {input_display_str}) "
                    f"interpreted as {n_type}, "
                    f"requires {revolutions} revolution(s)."
                )

        # --- Step 3: Determine d_values (tracing distances 'h') ---
        final_d_values = []
        if self.type == 'hypocycloid':
            final_d_values = list(self.b_values) # h = b
            if d is not None:
                warnings.warn(f"Type is 'hypocycloid', ignoring provided 'd'. Using d=r.")
        elif self.type == 'rose':
            # h = |a - b| creates specific rose-like patterns
            final_d_values = [abs(self.a - b) for b in self.b_values]
            if d is not None:
                warnings.warn(f"Type is 'rose', ignoring provided 'd'. Using d=|R - r|.")
        elif self.type == 'ellipse':
             # n=2 (a=2b) was enforced earlier. Use provided d.
            if d is None:
                # Default 'd' if none provided for ellipse
                default_ellipse_d = [b * 0.5 for b in self.b_values] # e.g., d = r/2 = R/4
                final_d_values = default_ellipse_d
                warnings.warn(f"Type is 'ellipse' but 'd' not provided. Using default d=r/2 (=R/4).")
            elif isinstance(d, (int, float)):
                 final_d_values = [float(d)] * num_circles
            else: # d is a list
                 d_float = [float(val) for val in d]
                 if len(d_float) != num_circles:
                    # If n_values wasn't specified, maybe adjust num_circles based on d
                    if n_values_in is None and r_values_in is None:
                         num_circles = len(d)
                         # Need to resize n_values, b_values etc. assuming n=2
                         self.n_values_numeric = [2.0] * num_circles
                         self.n_values_input_type = [2.0] * num_circles
                         self.b_values = [self.a / 2.0] * num_circles
                         self.n_types = ["integer"] * num_circles
                         self.n_revs = [1] * num_circles
                         # warnings.warn(f"Type is 'ellipse' and 'd' is a list. "
                         # f"Neither n_values nor r_values provided. Adjusted number of curves to match length of 'd' ({num_circles}).")
                         final_d_values = [float(val) for val in d]
                    else:
                        raise ValueError(
                            f"Length of 'd' ({len(d_float)}) must match "
                            f"the number of rolling circles ({num_circles}) "
                            "for ellipse type when 'd' is a list and n/r values were provided."
                        )
                 else:
                     final_d_values = d_float
        else: # Random type
            if d is None:
                # Default d for random: use r/2 for variety
                final_d_values = [b * 0.5 for b in self.b_values]
                # warnings.warn(f"Parameter 'd' not provided for random type. Using default d=r/2.")
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

        # Ensure h (d) is not negative, store as self.h_values
        self.h_values = [abs(val) for val in final_d_values]

        # --- Step 4: Set default filename if needed ---
        if self.save_anim and self.filename is None:
            # Use numeric n for filename consistency
            n_info = "_".join([f"n{n:.3f}".replace('.','p') for n in self.n_values_numeric])
            h_info = "_".join([f"d{h:.2f}".replace('.', 'p') for h in self.h_values])
            # Maybe include r info as well for clarity if r was input
            r_info = "_".join([f"r{r:.3f}".replace('.','p') for r in self.b_values]) if r_values_in is not None else ""
            self.filename = f"{self.type}_R{self.R:.1f}_{n_info}_{h_info}{'_'+r_info if r_info else ''}.gif"

    def _calculate_animation_parameters(self):
        """
        Calculate padding, theta range, and frame count based on n_types.
        The animation runs long enough for the longest closing pattern or
        the max duration for any irrational patterns.
        """
        # Calculate padding for plot limits
        # Max distance from origin = center_dist + tracing_dist = |a-b| + h
        max_reach = max([abs(self.a - b) + h for b, h in zip(self.b_values, self.h_values)], default=self.a)
        # Add padding relative to the larger of fixed radius or max reach
        base_limit = max(self.a, max_reach)
        self.plot_padding = base_limit * 0.15 # Add 15% padding
        self.plot_limit = base_limit + self.plot_padding

        # Determine total angle needed based on the maximum revolutions required by any curve
        if not self.n_revs: # Handle case of no valid circles
             self.max_total_revolutions = 1
        else:
             self.max_total_revolutions = max(self.n_revs)

        print(f"Animation requires max {self.max_total_revolutions:.2f} revolutions of the center.")

        self.theta_max_pattern = 2 * np.pi * self.max_total_revolutions
        self.total_frames = int(self.max_total_revolutions * self.frames_per_revolution)
        if self.total_frames <= 0: self.total_frames = self.frames_per_revolution # Ensure minimum frames

        # Generate theta values for the animation
        self.theta_vals = np.linspace(0, self.theta_max_pattern, self.total_frames)

    @staticmethod
    def _hypotrochoid_points(theta, a, b, h):
        """
        Calculate hypotrochoid points for a given angle theta (or array of thetas).
        a = fixed radius, b = rolling radius, h = tracing distance.
        """
        if b == 0: # Avoid division by zero
             # Point traces circle of radius h around center of rolling circle,
             # which itself is fixed at distance 'a' if we interpret b=0 literally
             # Or point just stays at origin Let's assume it traces a circle around (a,0) - unlikely case.
             # More likely, n=inf -> b=0. Point on circumference (h=b=0) stays at (a,0).
             # If h != 0, it would rotate around (a,0)
             # Let's return a fixed point at origin for simplicity. Requires clearer definition for b=0.
             center_x = a # Center stays at R
             center_y = 0 * theta # Keep shape consistent with theta input
             trace_x = a + h * np.cos(0) # Fixed point relative to center
             trace_y = 0 * theta
             if isinstance(theta, (np.ndarray)): # Match shape for array input
                return np.full_like(theta, center_x), np.full_like(theta, center_y), \
                       np.full_like(theta, trace_x), np.full_like(theta, trace_y)
             else: # Scalar input
                return center_x, center_y, trace_x, trace_y

        # Center of rolling circle
        center_x = (a - b) * np.cos(theta)
        center_y = (a - b) * np.sin(theta)

        trace_angle_term = (a - b) / b * theta # Angle for the tracing point calculation
        # Equivalent to (n - 1) * theta, where n = a/b

        # Tracing point position relative to the fixed circle's center
        trace_x = center_x + h * np.cos(trace_angle_term)
        trace_y = center_y - h * np.sin(trace_angle_term) # Minus sign for internal rolling

        return center_x, center_y, trace_x, trace_y

    def _setup_plot(self):
        """Sets up the Matplotlib figure and axes."""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
        # Use calculated limit
        self.ax.set_xlim(-self.plot_limit, self.plot_limit)
        self.ax.set_ylim(-self.plot_limit, self.plot_limit)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(False)
        title = f"Hypotrochoids (R={self.R:.2f}, Type: {self.type.capitalize()})"
        self.ax.set_title(title, fontsize=14, pad=10)

        # Stationary circle
        self.fixed_circle = patches.Circle((0, 0), self.a, fill=False, color='gray', lw=1.5, ls='--')
        self.ax.add_patch(self.fixed_circle)

        # Use a colormap for distinct colors
        if len(self.b_values) == 1:
             colors = ['yellow'] # Single curve color
        else:
             colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(self.b_values)))

        # Create plot elements for each hypotrochoid
        self.circles_rolling = []
        self.path_lines = []
        self.radius_lines = []
        self.tracing_points = []
        self.center_points = []

        for i, (b, h, n_val, n_typ) in enumerate(zip(self.b_values, self.h_values, self.n_values_numeric, self.n_types)):
            color = colors[i]

            # Rolling circle patch
            circle = patches.Circle((self.a - b, 0), b, fill=False, color=color, lw=1.5, alpha=0.8, zorder=3)
            self.ax.add_patch(circle)
            self.circles_rolling.append(circle)

            # Path line and label generation
            n_input = self.n_values_input_type[i] if i < len(self.n_values_input_type) else n_val 
            if n_typ == "rational":
                 # Prefer original Fraction if input, else format the limited fraction
                frac_n = n_input if isinstance(n_input, Fraction) else Fraction(n_val).limit_denominator(1000)
                n_label = f"{frac_n.numerator}/{frac_n.denominator}"
            elif n_typ == "integer":
                 n_label = f"{int(round(n_val))}" 
            else: # irrational
                 n_label = f"{n_val:.4f}" 

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
            self.ax.legend(loc='upper right', fontsize=9, facecolor='#1C1C1C', framealpha=0.7)

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
            # Reset circle position
            init_position = self._hypotrochoid_points(
                theta=0,
                a=self.a,
                b=self.b_values[i],
                h=self.h_values[i]
            )
            init_center_x, init_center_y, init_trace_x, init_trace_y = init_position
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
        current_theta = self.theta_vals[frame]
        # History of angles up to this frame for drawing the path
        theta_history = self.theta_vals[:frame+1]

        plot_elements_to_update = []

        for i, (b, h) in enumerate(zip(self.b_values, self.h_values)):
            # Calculate positions for the entire history for the path line
            (centers_x_hist, centers_y_hist,
             traces_x_hist, traces_y_hist) = self._hypotrochoid_points(
                theta_history,
                self.a,
                b,
                h)

            # Get position for the current frame for markers and rolling circle
            # The last point in the history corresponds to the current frame
            if len(traces_x_hist) > 0:
                 current_center_x = centers_x_hist[-1]
                 current_center_y = centers_y_hist[-1]
                 current_trace_x = traces_x_hist[-1]
                 current_trace_y = traces_y_hist[-1]
            else: # Frame 0 case
                 (current_center_x, current_center_y, 
                  current_trace_x, current_trace_y) = self._hypotrochoid_points(
                      0, 
                      self.a, 
                      b, 
                      h)

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
        """Creates and displays or saves the hypotrochoid animation."""
        if not self.b_values:
            print("No valid hypotrochoids to animate.")
            return None

        self._setup_plot() 
        if self.fig is None:
            raise ValueError("Figure object (self.fig) is not initialized.")


        # Create the animation object
        print(f"Generating animation with {self.total_frames} frames...")
        self.anim = FuncAnimation(self.fig, self._animate_frame, frames=self.total_frames,
                                  init_func=self._init_animation, interval=int(1000/self.FPS),
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
                 self.filename += ".gif"

            save_dir = Path("ANIMATIONS/HYPOTROCHOIDS")
            save_dir.mkdir(parents=True, exist_ok=True) 
            filepath = save_dir / self.filename

            print(f"Saving animation to {filepath.resolve()}...")
            writer_choice = 'pillow' if self.filename.lower().endswith('.gif') else 'ffmpeg'

            # Use tqdm for progress bar
            try:
                with tqdm(total=self.total_frames, desc="Saving Animation", unit="frame", ncols=100) as pbar:
                     def progress_update(current_frame, total_frames):
                          pbar.n = current_frame # Update tqdm progress
                          pbar.refresh() # Force redraw

                     self.anim.save(str(filepath), writer=writer_choice, fps=self.FPS,
                                    dpi=100, progress_callback=progress_update)
                # Ensure the progress bar finishes at 100%
                if pbar.n < self.total_frames:
                    pbar.update(self.total_frames - pbar.n)
                print("\nAnimation saved successfully!")
            except Exception as e:
                print(f"\nError saving animation: {e}")
                print("Ensure 'ffmpeg' (for MP4) or 'Pillow' (for GIF) is installed.")

            plt.close(self.fig) # Close the plot window after saving
        else:
            plt.show() # Display the animation interactively

        return self.anim # Return the animation object


if __name__ == "__main__":

    # --- Random Case Example (Default behavior if type not specified) ---
    # d is fixed, radii of the rolling circles are varied
    print("\nGenerating Random Hypotrochoids...")
    random_anim = Hypotrochoids(R=5, d=1, n_values=[3, 5, 7], 
                                save_anim=True, # filename automatically generated, if not specified
                                frames_per_rev=400, fps=50) 

    # random_anim.generate_animation() # Integer n values, 1 rev each


    # --- Random Example with float n and custom r ---
    print("\nGenerating Random Hypotrochoids with float n and specified r...")
    custom_random = Hypotrochoids(R=5, r_values=[1.5, 1, math.pi], # n = 5/1.5=10/3, n=5/1=5, n=5/pi=1.5915..
                                    d=[0.5, 1, 1.5], type='random',
                                    save_anim=False, 
                                    filename="hypotrochoids_random_n.gif",
                                    max_revolutions_irrational=5)
    # custom_random.generate_animation() # Should run for 5 revs as the max_revolutions_irrational is set to 5

    # --- Hypocycloid Example ---
    # --- Integer n ---
    print("\nGenerating Hypocycloid (Integer n=3)...")
    hypo_int = Hypotrochoids(R=5, n_values=3, type='hypocycloid', 
                             save_anim=False
                             )
    # hypo_int.generate_animation() # Should complete in 1 revolution

    # Multiple curves with integer n
    print("\nGenerating Hypocycloid (Integer n=3, 4, 5)...")
    hypo_anim = Hypotrochoids(R=4, n_values=[3, 4, 5], type='hypocycloid', 
                              save_anim=False
                              ) 
    # hypo_anim.generate_animation()

    # --- Rational n ---
    print("\nGenerating Hypocycloid (Rational n=3.5 = 7/2)...")
    hypo_rational = Hypotrochoids(R=4, n_values=Fraction(7, 2), type='hypocycloid',
                                    save_anim=False
                                    )
    # hypo_rational.generate_animation() # Should complete in 2 revolutions

    # Multiple curves with rational n
    print("\nGenerating Mixed Hypocycloids with rational n (n=5/2, 11/4, 18/5)...")
    hypo_rational_mixed = Hypotrochoids(R=4, n_values=[5/2, Fraction(11,4), 18/5],
                                        type='hypocycloid',
                                        save_anim=False, 
                                        fps=50)
    # hypo_rational_mixed.generate_animation() # Should complete in 2, 4 and 5 revolutions respectively

    # --- Irrational n ---
    print("\nGenerating Hypocycloids (Irrational n=pi)...")
    hypo_irrational = Hypotrochoids(R=4, n_values=math.pi, type='hypocycloid',
                                    max_revolutions_irrational=4, 
                                    save_anim=False
                                    )
    # hypo_irrational.generate_animation() # Should run for 4 revolutions

    # --- Multiple Curves with Mixed n Types ---
    print("\nGenerating Mixed Hypocycloids (n=3, 7/2, pi)...")
    mixed_anim = Hypotrochoids(R=5, n_values=[3, Fraction(7,2), math.pi], 
                               type='hypocycloid',
                               max_revolutions_irrational=5, 
                               save_anim=True, 
                               fps=40)
    # Revolutions depend on the max_revolutions_irrational number (longest duration needed for pi)
    # mixed_anim.generate_animation() 

    # --- Ellipse Example (n must be 2) with Rational/Irrational d ---
    print("\nGenerating Ellipses (n=2 forced), different d values...")
    # Note: n=2 is integer, duration is 1 rev. 'd' doesn't affect duration.
    ellipse_anim = Hypotrochoids(R=6, d=[0.5, 1.25, np.sqrt(3)], 
                                 type='ellipse',
                                 save_anim=True, 
                                 filename="ellipses_example.gif")
    # ellipse_anim.generate_animation()

    # Single ellipse, specifying n=2 explicitly
    ellipse_anim_single = Hypotrochoids(R=4, d=1, n_values=2, 
                                        type='ellipse', 
                                        save_anim=False
                                        )
    # ellipse_anim_single.generate_animation()

     # --- Rose Example (Integer n) ---
    print("\nGenerating Rose Curves (n=3, 7, 5)...")
    # Note: n=3,4,5 are integers, duration is 1 rev. 'd' doesn't affect duration.
    rose_anim = Hypotrochoids(R=4, n_values=[3, 7, 5], 
                              type='rose', 
                              save_anim=True, 
                              filename="rose_integer_2.gif",
                              frames_per_rev=400, 
                              fps=50)
    # rose_anim.generate_animation() # Should run for 1 revolution

    # (Rational n) 
    print("\nGenerating Rose Curves for n = 4/3, 5/4, 7/5...")
    # Note: n=4/3, 5/4, 7/5 are rational, duration is 3 revs, 4 revs and 5 revs respectively. 
    rose_anim_rational = Hypotrochoids(R=4, n_values=[Fraction(4,3), Fraction(5,4), Fraction(7,5)],
                                        type='rose', frames_per_rev=100, fps=30,
                                        save_anim=False, 
                                        filename="rose_rational.gif")
    # rose_anim_rational.generate_animation() # Should run for 3, 4 and 5 revolutions respectively



