import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit, least_squares

# Median Absolute Deviation scaled to standard deviation by 1.4826
def Median_Absolute_Deviation(data, want_scaled = True):
    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        return np.nan

    mad = np.median(np.abs(data - np.median(data)))
    
    if want_scaled:
        return 1.4826 * mad
    else:
        return mad

def Min_Max_Normalization(data):
    data = np.asarray(data, dtype=float)
    
    if len(data) == 0:
        return np.nan
    
    min_val = np.min(data)
    max_val = np.max(data)
    if min_val == max_val:
        return np.zeros_like(data, dtype=float)
    return (data - min_val) / (max_val - min_val)

def Plot_Ring_Center_And_Radius(img_path, Cx, Cy, mean_radius, std_radius, want_radial_deviation = True, save_path = None):
    donut_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display image
    ax.imshow(donut_img, cmap="gray")

    # Mean radius
    mean_circle = patches.Circle((Cx, Cy),mean_radius,fill=False,linewidth=2,label=f"Mean radius = {mean_radius:.1f} px")
    ax.add_patch(mean_circle)

    if want_radial_deviation:
        # -1 sigma
        inner_circle = patches.Circle((Cx, Cy),mean_radius - std_radius,fill=False,linewidth=2,linestyle="--",label=f"−1σ = {mean_radius - std_radius:.1f} px")
        ax.add_patch(inner_circle)
        
        # +1 sigma
        outer_circle = patches.Circle((Cx, Cy),mean_radius + std_radius,fill=False,linewidth=2,linestyle="--",label=f"+1σ = {mean_radius + std_radius:.1f} px")
        ax.add_patch(outer_circle)

    ax.scatter(Cx, Cy, s = 150, c = 'r', marker = '+')

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend()

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()

def Gaussian_2D(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def Get_Pixel_To_Length_Conversion(pixel_array_width, pixel_array_height, camera_sensor_pitch, array_side_length = 200):
    pixel_side_length = (pixel_array_width + pixel_array_height)/2 # pixel
    pixel_to_length = array_side_length / pixel_side_length # um/pixel
    magnification = camera_sensor_pitch / pixel_to_length
    
    return pixel_to_length, magnification

# Computes the emission angle of the laser, all inputs must be in millimeters
def Get_Theta(f_obj, f_lens, diameter, distance_from_image_plane):
    return np.rad2deg(np.arctan( (f_lens/f_obj) * (diameter/2) * (1/distance_from_image_plane)))

# Fit a 2D Gaussian to an image of a beam, fitting an X and Y beam diameter. 
# Utilize pixel to length conversion to convert the X & Y diameters from pixels to micrometers
# Then compute the beam area with the ellipse area formula: PI * diameter_x/2 * diameter_y/2
def Get_Beam_Profile(beam_path, pixel_to_length_conversion, save_path = None):
    # Load image
    img = cv2.imread(beam_path, cv2.IMREAD_GRAYSCALE) 

    # Find the absolute brightest pixel to center our crop
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
    peak_y, peak_x = max_loc[1], max_loc[0]

    # 3. Crop a Region of Interest (ROI) around the peak
    roi_size = 100  # Adjust based on the estimated size of your spot
    y_min, y_max = max(0, peak_y - roi_size), min(img.shape[0], peak_y + roi_size)
    x_min, x_max = max(0, peak_x - roi_size), min(img.shape[1], peak_x + roi_size)
    roi = img[y_min:y_max, x_min:x_max]

    # 4. Create a coordinate grid for the ROI
    x = np.arange(roi.shape[1])
    y = np.arange(roi.shape[0])
    x, y = np.meshgrid(x, y)

    # 5. Set initial guesses for the optimization [amplitude, xo, yo, sigma_x, sigma_y, theta, offset]
    initial_guess = (max_val - min_val, roi_size, roi_size, 10, 10, 0, min_val)

    # 6. Fit the 2D Gaussian to the data
    popt, pcov = curve_fit(Gaussian_2D, (x, y), roi.ravel(), p0=initial_guess)

    # 7. Extract standard deviations and calculate 1/e² diameter
    # Note: 1/e² radius (w) = 2 * sigma -> 1/e² diameter (D) = 4 * sigma
    amplitude_fit, xo_fit, yo_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit = popt

    d_1e2_x = 4 * abs(sigma_x_fit)
    d_1e2_y = 4 * abs(sigma_y_fit)
    
    gaussian_diameter_x = d_1e2_x * pixel_to_length_conversion
    gaussian_diameter_y = d_1e2_y * pixel_to_length_conversion
    area = np.pi * (gaussian_diameter_x / 2) * (gaussian_diameter_y / 2)

    print(f"Area of Beam: {area:.2f} um^2")
    
    # Generate the 2D fit image grid
    fitted_data = Gaussian_2D((x, y), *popt).reshape(roi.shape)

    if save_path is not None:
        # --- PLOTTING --- #
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Original Laser Spot + Ellipse Overlay
        im1 = ax1.imshow(roi, cmap='viridis')
        ax1.set_title('Original Laser Spot (ROI)')
        fig.colorbar(im1, ax=ax1, label='Intensity')

        # Invert the angle for image coordinate system (Y-down)
        angle_deg = -np.degrees(theta_fit)

        # Create the patch explicitly for the second axis
        beam_ellipse = patches.Ellipse((xo_fit, yo_fit), width=d_1e2_x, height=d_1e2_y, angle=angle_deg, edgecolor='cyan', facecolor='none', linewidth=2, linestyle='--', label='1/e² Boundary')
        ax2.add_patch(beam_ellipse) # Adds the ellipse to the original spot subplot
        ax2.legend(loc='upper right')
    
        # Plot 2: 2D Gaussian Fit
        im2 = ax2.imshow(fitted_data, cmap='viridis')
        ax2.set_title('2D Gaussian Fit')
        fig.colorbar(im2, ax=ax2, label='Intensity')
        ax2.axvline(x=xo_fit, color='red', linestyle='--', alpha=0.6, label='Center')
        ax2.axhline(y=yo_fit, color='red', linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(save_path)
        
    return (gaussian_diameter_x, gaussian_diameter_y), area

def Adaptive_Sobel_Filter(img_path, adaptive_filter_size, gaussian_smoothing_size, sobel_kernel_size = 3):
    # 1. Load in grayscale and apply mild blurring to ignore camera sensor noise
    image = cv2.imread(img_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (gaussian_smoothing_size, gaussian_smoothing_size), 0)
    
    # 3. Apply Local Adaptive Thresholding
    # Parameters explained below
    adaptive_mask = cv2.adaptiveThreshold(
        src=blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, # Inverts so the square outline becomes white
        blockSize=adaptive_filter_size,                        # MUST be an odd number (e.g., 21, 31, 51, 101)
        C=7                                  # Constant subtracted from the calculated local mean
    )

    # 2. Run Sobel on both axes (using 64-bit float to preserve negative transitions)
    sobel_x = cv2.Sobel(adaptive_mask, cv2.CV_64F, 1, 0, ksize = sobel_kernel_size)
    sobel_y = cv2.Sobel(adaptive_mask, cv2.CV_64F, 0, 1, ksize = sobel_kernel_size)

    # 3. Calculate exact gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return gray_img, sobel_normalized

def Determine_Dimensions_Of_Reference_Array(img_path, adaptive_filter_size, gaussian_smoothing_size, sobel_kernel_size = 3, save_path = None):
    # Determine the edges in the image with an adaptive sobel filter
    original_image, sobel_image = Adaptive_Sobel_Filter(img_path, adaptive_filter_size, gaussian_smoothing_size, sobel_kernel_size)
    
    # 3. Find all connected shapes (contours) in the image
    contours, _ = cv2.findContours(sobel_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    width, height = None, None
    if contours:
        # 4. CRUCIAL FILTER: Select ONLY the largest contour by area
        # This completely discards the scattered tiny dust particles/specks
        largest_contour = max(contours, key=cv2.contourArea)
    
        # 5. Extract only the (x, y) coordinates belonging to this largest contour
        # Reshaping makes it a clean list of [[x1, y1], [x2, y2], ...]
        pts = largest_contour.reshape(-1, 2)
    
        # 6. Fit the tightest rotated rectangle around the clean pixel cloud
        rect = cv2.minAreaRect(pts)
    
        # 7. Extract the 4 precise corner vertices
        box_points = cv2.boxPoints(rect)
        box_points = np.int64(box_points) # Convert to integer pixel coordinates
    
        # Create an output canvas for display
        output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
        # 8. Draw the green box wrapping only the main square sample
        cv2.drawContours(output_image, [box_points], 0, (0, 255, 0), 3)
    
        # 9. Print and label the exact corners
        for i, pt in enumerate(box_points):
            # Mark each clean computed corner with a solid red dot
            cv2.circle(output_image, (pt[0], pt[1]), 8, (0, 0, 255), -1)
        
        # Calculate dimensions securely from the box points
        width = np.linalg.norm(box_points[0] - box_points[1])
        height = np.linalg.norm(box_points[1] - box_points[2])
    else:
        print("Error: Could not locate any edges to evaluate.")

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        
    if save_path is not None:
        # Render the final interactive plot layout
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(output_image_rgb)
        ax.set_title("Array Border Recognition")
        ax.axis('on')  # Keeps pixel coordinates visible on borders
        
        plt.tight_layout()
        plt.savefig(save_path)
    return width, height

def Fit_Gaussian_Ring(donut_img_path, downsample=4):
    donut_image = np.asarray(cv2.imread(donut_img_path, cv2.IMREAD_GRAYSCALE), dtype = float)

    # Downsample for faster fitting
    img = donut_image[::downsample, ::downsample]

    h, w = img.shape

    Y, X = np.indices(img.shape)

    # --------------------------------------------------
    # 1. Initial center estimate
    # --------------------------------------------------

    background = gaussian_filter(img, sigma=30)

    corrected = img - background

    weights = np.maximum(corrected, 0)

    if np.sum(weights) == 0:
        raise ValueError("No detectable ring signal.")

    cx0 = np.sum(X * weights) / np.sum(weights)
    cy0 = np.sum(Y * weights) / np.sum(weights)

    # --------------------------------------------------
    # 2. Initial radius estimate
    # --------------------------------------------------

    distances = np.sqrt((X - cx0)**2 + (Y - cy0)**2)

    radial_bins = np.arange(0, int(np.max(distances)) + 2)

    radial_sum, _ = np.histogram(distances, bins = radial_bins, weights = weights)

    radial_count, _ = np.histogram(distances, bins = radial_bins)

    radial_profile = radial_sum / np.maximum(radial_count, 1)

    # Ignore the central region
    radial_profile[:5] = 0

    r0 = np.argmax(radial_profile)

    # Estimate ring thickness
    sigma0 = 5.0

    # --------------------------------------------------
    # 3. Gaussian annulus model
    # --------------------------------------------------

    Xn = (X - w / 2) / w
    Yn = (Y - h / 2) / h

    def model(params):

        cx, cy, radius, sigma, amplitude, b0, bx, by = params

        r = np.sqrt((X - cx)**2 + (Y - cy)**2)

        background = b0 + bx * Xn + by * Yn

        ring = amplitude * np.exp(-0.5 * ((r - radius) / sigma)**2)

        return background + ring

    # Initial background and amplitude
    b0 = np.percentile(img, 20)
    amplitude = max(np.max(img) - b0, 1)

    initial = [cx0, cy0, r0, sigma0, amplitude, b0, 0, 0]

    lower = [0, 0, 1, 1, 0, -np.inf, -np.inf, -np.inf]

    upper = [w, h, np.hypot(w, h), max(w, h), np.inf, np.inf, np.inf, np.inf]

    def residual(params):
        return (model(params) - img).ravel()

    result = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1', f_scale=max(np.std(img) * 0.5, 1), max_nfev=500)

    cx, cy, radius, sigma = result.x[:4]

    # Convert back to original image coordinates
    cx *= downsample
    cy *= downsample
    radius *= downsample
    sigma *= downsample

    return cx, cy, radius, sigma

def Extract_Radial_Intensity_Profiles(gray_img, cx, cy, num_angles=180):
  height, width = gray_img.shape
  max_radius = int(np.hypot(max(cx, width - cx), max(cy, height - cy)))
  s = np.arange(-max_radius, max_radius + 1, 1.0)

  angles = np.linspace(0, 180, num_angles, endpoint=False)
  profiles = {}

  for a in angles:
    theta = np.deg2rad(a)
    x = cx + s * np.cos(theta)
    y = cy + s * np.sin(theta)

    valid = (x >= 0) & (x < width - 1) & (y >= 0) & (y < height - 1)
    x_v, y_v, s_v = x[valid], y[valid], s[valid]

    # Fast bilinear interpolation using cv2.remap or manual indexing
    x0 = np.floor(x_v).astype(int)
    y0 = np.floor(y_v).astype(int)
    dx = x_v - x0
    dy = y_v - y0

    intensity = (
        gray_img[y0, x0] * (1 - dx) * (1 - dy)
        + gray_img[y0, x0 + 1] * dx * (1 - dy)
        + gray_img[y0 + 1, x0] * (1 - dx) * dy
        + gray_img[y0 + 1, x0 + 1] * dx * dy
    )
    profiles[a] = {"position": s_v, "intensity": intensity}

  return profiles


def Refine_Donut_Center_And_Diameter(img_path, cx_init, cy_init, sensor_pitch_um, window_size=51, max_iters=15, tol=0.05):
    donut_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
    
    # Pre-smooth the image once
    smoothed_img = cv2.GaussianBlur(donut_img, (0, 0), 2)
    cx, cy = float(cx_init), float(cy_init)

    for iteration in range(max_iters):
        profiles = Extract_Radial_Intensity_Profiles(smoothed_img, cx, cy, num_angles=180)

        valid_thetas = []
        midpoints = []
        diameters = []

        for a, data in profiles.items():
            pos = data["position"]
            # Normalize profile
            inten = data["intensity"]
            inten = (inten - inten.min()) / (inten.max() - inten.min() + 1e-8)

            smooth_inten = savgol_filter(inten, window_length=window_size, polyorder=3)
            peaks, _ = find_peaks(smooth_inten, prominence=0.05, distance=100)

            # Keep only pairs where one peak is on the negative side and one on the positive side
            neg_peaks = peaks[pos[peaks] < 0]
            pos_peaks = peaks[pos[peaks] > 0]

            if len(neg_peaks) > 0 and len(pos_peaks) > 0:
                # Take the strongest peak on each side
                p_neg = neg_peaks[np.argmax(smooth_inten[neg_peaks])]
                p_pos = pos_peaks[np.argmax(smooth_inten[pos_peaks])]

                s1, s2 = pos[p_neg], pos[p_pos]
                midpoints.append(0.5 * (s1 + s2))
                diameters.append(s2 - s1)
                valid_thetas.append(np.deg2rad(a))

        valid_thetas = np.array(valid_thetas)
        midpoints = np.array(midpoints)
        A = np.column_stack([np.cos(valid_thetas), np.sin(valid_thetas)])

        # Robust linear fit (ignores boundary-clipped outliers)
        res = least_squares(lambda delta: A @ delta - midpoints, x0=[0.0, 0.0], loss="soft_l1")
        dx, dy = res.x

        cx += dx
        cy += dy

        step_size = np.hypot(dx, dy)
        
        if step_size < tol:
            break
        
    # Convert the diameters from pixels to mm
    sensor_pitch_mm = sensor_pitch_um * 1e-3 # mm/px
    
    diameter_px = np.median(diameters) # px
    diameter_px_deviation = Median_Absolute_Deviation(diameters) # px
    
    diameter_mm = diameter_px * sensor_pitch_mm # mm
    diameter_mm_deviation = diameter_px_deviation * sensor_pitch_mm # mm
    
    return cx, cy, diameter_mm, diameter_mm_deviation