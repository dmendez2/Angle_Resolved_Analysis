from wavelength_calibrator import WavelengthCalibrator
from scipy.ndimage import binary_opening, distance_transform_edt
from scipy.ndimage import gaussian_filter1d, maximum_filter, median_filter, uniform_filter1d
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial import distance_matrix
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import h5py 

### HELPER FUNCTIONS ###

# Returns the Median Absolute Deviation of a Data Distribution Scaled by 1.4826 to match the scale of a standard deviation
def Scaled_MAD(data):
    data = np.asarray(data)
    return 1.4826 * np.median(np.abs(data - np.median(data)))

# Perform a min-max normalization
def Min_Max_Normalization(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Returns the most intense pixel along the x-axis
def Get_Brightest_Px(data):
    averaged_data = np.mean(data, axis = 0)
    Px_max = np.argmax(averaged_data)
    return Px_max

def Get_Brightest_Py(input_path):
    wavelength, intensity = Get_Raw_Data(input_path)
    intensity = Min_Max_Normalization(intensity)
    
    averaged_data = np.mean(intensity, axis = 1)
    return np.argmax(averaged_data)

# Opens the hdf5 file and returns the raw data
def Get_Raw_Data(file_name, Px_max = 6248):
    intensity_data = None
    wavelength_data = None
    with h5py.File(file_name, 'r') as f:
        # Get the 2-D Raw Data
        raw_data = np.array(f['raw_data'])
        
        # Check that the background was subtracted
        isBackgroundSubtracted = f.attrs['background_subtraction_applied']
        if not isBackgroundSubtracted:
            print("Background was not subtracted")
            
        # Check if the data is saturated
        if np.max(raw_data >= 65000):
            print("Warning Overflow may Have Occurred")
        
        # Get the Calibration Parameters
        Pc = f.attrs["central_pixel"]
        Dv = f.attrs["deviation_angle (rads)"]
        gamma = f.attrs["tilt_angle (rads)"]
        lambda_c = f.attrs["central_wavelength (nm)"]
        
    # Utilize the wavelength calibrator to convert each x-axis pixel to a wavelength
    pixels = np.arange(0, Px_max, 1)
    calibrator = WavelengthCalibrator()
    calibrator.Set_Free_Parameters({'Pc': Pc, 'Dv': Dv, 'gamma': gamma, 'lambda_c': lambda_c})
    wavelength_data = calibrator.Get_Calibrated_Wavelength_From_Pixel(pixels)
    
    # Convert the pixel intensity from a raw value to one normalized by the maximum value possible for a 16-bit camera
    intensity_data = calibrator.Process_Pixel_Intensity(raw_data)

    return wavelength_data, np.squeeze(intensity_data)

### GENERAL CALIBRATION FUNCTIONS ###

# Determine the band of pixels and corresponding angles we could theoretically expect signal from 
def Determine_Max_Theoretical_Signal_Band(numerical_aperture, slope, normal, Py_Max = 4176):
    # The max angle is given by n_i * sin(theta_max) = NA
    # n_i is the refractive index of the initial medium, theta_max is the maximum angle that passes through the aperture, NA is the numerical aperture
    # n_i = 1 (Air) so theta_max = arcsin(NA)
    max_angle = np.rad2deg(np.arcsin(numerical_aperture))
    
    # Determine the max pixel relative to normal which corresponds to this maximum angle utilizing our fit P = A/m 
    # P = Y-axis pixel, A = angle, m = pixel_per_angle (slope from fit)
    max_pixel_relative_to_normal = int(max_angle/slope)
    
    # Because we can have positive and negative angles, the band of pixels which can theoretically detect signal goes from 
    # The negative of the max pixel to 0 to the positive of the max pixel
    pixel_band = np.arange(-max_pixel_relative_to_normal, max_pixel_relative_to_normal + 1, 1)
    
    # Multiply the pixel band by the pixel_per_angle to determine the angular band, the angle corresponding to each pixel
    angular_band = pixel_band * slope
    
    # Add the normal to the pixel band to go from the normal coordinate being the origin to the natural pixel coordinates of the camera
    pixel_band += normal
    
    # Check for clipping, remove any pixel values lower than 0 or greater than the max pixel of the camera (4176 for the ASI camera)
    overflow_mask = (pixel_band >= 0) & (pixel_band < Py_Max)
    pixel_band = pixel_band[overflow_mask]
    angular_band = angular_band[overflow_mask]
    
    return pixel_band, angular_band

### FILLED NUMERICAL APERTURE CALIBRATION FUNCTIONS ###

# Light is reflected off a diffuse reflector and thus reflected at several different angles
# The light is then captured by an objective and only angles that can be accepted for the given NA can continue to propagate (NA = nsin\theta)
# The light exiting from the objective passes through a tube lens (L1) and a Fourier lens (L2) with focal length F1 and F2 respectively
# The lenses are separated by F1 + F2 and the Fourier lens directs the light to a spectrometer slit F2 away (Creating an image of the Fourier plane)
# As seen in Fourier space the light at the spectrometer slit has a radius of r = fsin\theta
# The light enters the spectrometer slit and is then imaged by a camera (The light is now obviously a thin slit shape)
# Since r_signal < r_camera this means there is signal from the diffuse reflector at the center of the camera surround to the left and right by a dark background
# We must accurately determine at what pixel the background ends and the signal begins 
# This is neccesary to get an accurate angular calibration utilizing the width of the signal from the numerical aperture
# We know the signal is always background at the left and right edges so we compute a background value from these regions
# Then we compute an energy E(p) = 1/(2W + 1) * [SUM((I(p) - B)^2) FROM w_0 TO w_n] by sliding a window of size w across the intensity data
# Above, I is the intensity of the signal, B is the background, p is the pixel value, W is a window size, w_0 is the 1st window and w_n is the final window
# We then compute a log-energy Z(p) = Log(E(p)) to reduce the difference in intensity between peaks and troughs within the signal
# The difference between signal and background is much larger than difference between signal peaks and signal troughs
# Therefore we can utilize variance in the log-energy to determine the boundary points between signal and background
# The variance in the log energy over a set of pixels from P_i to P_j is given by V(Z(p)) = SUM[Z^2(p)] - ( SUM[Z(p)]^2 / n ) FROM P_i TO P_j
# We therefore compute the variance over a left background region (0 --> P_L), a center signal region (P_L --> P_R), and a right background region (P_R --> N)
# We then vary P_L and P_R to minimize the variance in each region which corresponds to the left region being purely background, the center being purely signal, and the right being purely background
# P_L and P_R then become the boundary points for the signal and are returned
def Find_Signal_Boundaries(intensity, window_size = 31, min_segment_size = 100, step = 5, background_fraction = 0.15, want_plot = False):
    # Convert the intensity to a Numpy array if it is not already, then get the size of the array
    intensity = np.asarray(intensity, dtype=float)
    N = len(intensity)

    # --------------------------------------------------
    # 1. Estimate background from both ends
    # --------------------------------------------------

    # We assume that a certain fraction of the beginning and end of the data is background
    # The size of the data is mulitplied by the background fraction to get the explicit number of background pixels (n_b)
    # Then the beginning n_b pixels and ending n_b pixels are considered background and their median is computed to get the background level (B)
    n_background = max(1,int(N * background_fraction))
    background = np.concatenate([intensity[:n_background],intensity[-n_background:]])
    background_median = np.median(background)

    # --------------------------------------------------
    # 2. Calculate local signal energy
    # --------------------------------------------------

    # The residual is the signal minus the background
    # Mathematically, as a function of pixel, p, we have R(p) = I(p) - B
    residual = intensity - background_median
    
    # We now slide a window of size w across the data for each pixel and for each pixel compute the sum of residuals squared across that window 
    # FROM -w to w do SUM(R^2(P + w))
    # Squaring ensures that equal positive and negative deviations from the background are treated the same
    # We then weight the result by the size of the window to compute an energy (E(p))-> E(p) = 1/(2w + 1) * (FROM -w to w do SUM(R^2(P + w)))
    # E(p) acts as an "energy" since it is a measure of how far from background (0 potential energy) that the system at pixel P is
    # This energy is smoothened from noise spikes due to the fact that we do a weight of the energy at pixel P as well as by its w left neighbors and w right neighbors
    # The 'nearest' mode means values beyond the ends of the array are replaced by the nearest endpoint value
    energy = uniform_filter1d(residual**2, size = window_size, mode = "nearest")

    # --------------------------------------------------
    # 3. Log-transform local energy
    # --------------------------------------------------

    # The diffuse reflection is not constant as the beam is scanned across its surface
    # Therefore some angles may have very high intensities while others do not and this relative difference changes per measurement
    # To account for the large relative differences (so dips in the reflection are not assigned as background) we take the logarithm of the energy
    # Since Log(0) is not defined, we must ensure there are no 0 values in the energy
    # To do this we first compute the 10th percentile of the energy distribution
    # If this value is above 1e-12 then we use it as some small value, epsilon
    # Otherwise we use 1e-12 as epsilon to ensure when we add epsilon we are numerically above 0
    # Then we compute Log(E + epsilon) to transform our energy to a logarithm scale
    epsilon = max(0.1 * np.quantile(energy, 0.1), 1e-12)
    log_energy = np.log(energy + epsilon)

    # --------------------------------------------------
    # 4. Calculate cumulative sums
    # --------------------------------------------------

    # Right now we have the log-energy (Z) for each pixel (p_n) --> [Z-p1, Z-p2, .... Z-pn]
    # Now we take the cumulative sum of the log energy for each window --> [0, Z-p1, Z-p1 + Z-wp, ..., Z-p1 + Z-p2 + .... Z-pn]
    cumulative_sum = np.concatenate([[0],np.cumsum(log_energy)])

    # We do a similar cumulative sum for the log-energy squared
    cumulative_sum_squared = np.concatenate([[0],np.cumsum(log_energy**2)])

    # Recall that the formula for variance is proportional to: V(x) = SUM[(x_i - x_bar)^2] where x_bar is the mean
    # We now compute the variance of the log-energy for each segment V(Z(p)) = SUM[(Z(p) - Z_bar)^2] = SUM[Z^2(p)] - 2*Z_barSUM[Z(p)] + n*Z_bar^2
    # But Z_bar = SUM(Z(p))/n so we now have
    # V(Z(p)) = SUM[Z^2(p)] - ( SUM[Z(p)]^2 / n )
    # Therefore, the variance in Z(p) of a given segment can be computed by the cumulative sums of Z(p) and Z^2(p) in that segment
    def Segment_Variance(start, end):
        n = end - start

        Z_total = (cumulative_sum[end] - cumulative_sum[start])

        Z_squared_total = (cumulative_sum_squared[end] - cumulative_sum_squared[start])

        return Z_squared_total - Z_total**2 / n

    # --------------------------------------------------
    # 5. Find optimal two change points
    # --------------------------------------------------

    # We now break up the Py pixels into segments where the change point could occur
    # If min_segment size is 100 and step size is 5 and the max pixel is 4000 then our candidate change points would be [100, 105, 110, .... 3900] --> 4095 is the max Py pixel
    candidates = np.arange(min_segment_size, N - min_segment_size,step)

    # Set starting values for optimizer
    best_variance = np.inf
    left_edge = None
    right_edge = None

    # When each segment is purely background or purely signal then the variance in the log-energy (Z) will be minimized
    # Therefore, we attempt to determine the points that separate the left and right background from the signal to minimize the variance in Z
    # We loop through each candidate and during each iteration the current candidate is assumed to be the left boundary point
    for left in candidates:

        # The right candidates are all the values greater than the current candidate
        # The code below is vectorized so we check the total variance considering all possible right boundary points assuming the current candidate is the left boundary point
        right_candidates = candidates[candidates > left + min_segment_size]

        if len(right_candidates) == 0:
            continue

        # Left background
        # Compute the variance from the 0th pixel to the current left background boundary point (Initialized to min_segment_size)
        left_variance = Segment_Variance(0, left)

        # Middle signal region
        # Compute the variance from the left background boundary point to the right background boundary point (Initialized to N - min_segment_size)
        middle_variance = Segment_Variance(left, right_candidates)

        # Right background
        # Compute the variance from the right background boundary point to the terminal Nth pixel
        right_variance = Segment_Variance(right_candidates, N)
        
        # The total variance is the sum of the variance from the left background, the middle signal region, and the right background
        total_variance = (left_variance + middle_variance + right_variance)

        # Since we tested multiple right candidates when assuming the current candidate is the left boundary point
        # we only look at the minimum total variance of all tests done with the current left candidate to get the best respective right candidate
        # We then compare this minimum total variance with the overall best variance we have found so far
        # If the current total variance is less than the best variance found so far, then we update the best variance found to the current total variance
        # Then we update the left edge to the left candidate and the right edge to the right candidate. 
        index = np.argmin(total_variance)
        if total_variance[index] < best_variance:

            best_variance = total_variance[index]

            left_edge = left

            right_edge = right_candidates[index]

    if left_edge is None:
        return None

    # --------------------------------------------------
    # 6. Plot results
    # --------------------------------------------------

    if want_plot:

        fig, axes = plt.subplots(2, 1,figsize=(12, 8),sharex=True)

        axes[0].plot(intensity, color="tab:blue", label="Intensity")

        axes[0].axvline(left_edge, color="tab:red",linestyle="--", label="Left change point")

        axes[0].axvline(right_edge, color="tab:green", linestyle="--",label="Right change point")

        axes[0].axvspan(left_edge, right_edge, alpha=0.1, color="tab:green")

        axes[0].set_ylabel("Intensity")
        axes[0].legend()

        axes[1].plot(log_energy, color="tab:purple", label="Log local energy")

        axes[1].axvline(left_edge, color="tab:red", linestyle="--")

        axes[1].axvline(right_edge, color="tab:green", linestyle="--")

        axes[1].set_xlabel("Pixel")
        axes[1].set_ylabel("Log energy")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return left_edge, right_edge

def Get_NA_Band_Properties(pixel_axis, intensity_profile, NA, want_plot = False):
    """
    Smooths a cross-section profile, detects the NA edges via a 50% threshold,
    and returns calibration constants and plotting coordinates.
    """
    
    # Step A: Smooth out the speckle noise using a combination of filters
    # Median filter handles sharp spikes; Gaussian smooths the edges
    smoothed = median_filter(intensity_profile, size = 3)
    smoothed = gaussian_filter1d(smoothed, sigma = 3)
    
    # Step B: Determine the boundary between signal and background to determine the width of the NA band
    idx_low, idx_high = Find_Signal_Boundaries(smoothed, want_plot = want_plot, window_size = 15, min_segment_size = 100, step = 5)
    
    y_low = pixel_axis[idx_low]
    y_high = pixel_axis[idx_high]
    
    # Step D: Calculate Center and Scaling Factor
    y_0 = (y_low + y_high) / 2.0
    na_per_pixel = NA / (y_high - y_0)
    
    # Step E: Convert the entire pixel array to NA and Degrees
    na_profile = np.abs(pixel_axis - y_0) * na_per_pixel
    # Clip NA to 1.0 to avoid invalid arcsin values due to noise outside the pupil
    na_profile_clipped = np.clip(na_profile, 0, 1.0)
    angle_profile = np.degrees(np.arcsin(na_profile_clipped))
    
    y_0 = (y_low + y_high) / 2.0
    
    # Calculate the maximum angle in degrees corresponding to the nominal NA
    theta_max = np.degrees(np.arcsin(NA))
    
    # Calculate the linear calibration factor in degrees per pixel
    degrees_per_pixel = theta_max / (y_high - y_0)
    
    # Linear mapping to degrees
    angle_profile = np.abs(pixel_axis - y_0) * degrees_per_pixel
    
    # Calculate the corresponding NA profile back from the angles
    na_profile = np.sin(np.radians(angle_profile))
    
    return {
        "center_pixel": y_0,
        "degrees_per_pixel": degrees_per_pixel, 
        "y_low": y_low,
        "y_high": y_high,
        "smoothed_profile": smoothed,
        "na_profile": na_profile,
        "angle_profile": angle_profile,
    }
    
def NA_Band_Angular_Calibration_Hub(NA, input_directory, statistical_average = 'median', want_plot = False, print_statistics = False):
    # Specify the directory path (use '.' for the current working directory)
    dir_path = Path(input_directory)
    
    # Lists to store the pixel to angle calibration and the center pixel
    center_pixels = []
    pixel_to_angle_slopes = []
    
    # List only files
    files = [f for f in dir_path.iterdir() if f.is_file()]
    for f in files:
        wl, intens = Get_Raw_Data(f)

        max_idx = np.unravel_index(np.argmax(intens), intens.shape)

        y_pixels = np.arange(0, intens.shape[0], 1)
        y_profile = intens[:, max_idx[1]]

        results = Get_NA_Band_Properties(y_pixels, y_profile, NA, want_plot = want_plot)
        center_pixels.append(results['center_pixel'])
        pixel_to_angle_slopes.append(results['degrees_per_pixel'])

        if want_plot:
            # Plot the Raw Image
            #plt.imshow(intens)
            #plt.show()
            
            # Plot the processing steps
            plt.figure(figsize=(10, 6))
            plt.plot(y_pixels, y_profile, label='Raw Saturated Profile', alpha=0.5, color='gray')
            plt.plot(y_pixels, results['smoothed_profile'], label='Filtered & Smoothed', color='blue', linewidth=2.0)

            # Mark detected critical points
            plt.axvline(results['y_low'], color='green', linestyle=':', linewidth=2, label='Detected Edges')
            plt.axvline(results['y_high'], color='green', linestyle=':', linewidth=2)
            plt.axvline(results['center_pixel'], color='purple', linestyle='-', label='Calculated Center ($y_0$)')

            plt.title('Numerical Aperture Band Detection')
            plt.xlabel('Pixel Index')
            plt.ylabel('Normalized Intensity')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
    if statistical_average == 'median':
        Pc = np.median(center_pixels)
        Pc_deviation = Scaled_MAD(center_pixels)
        
        pixel_to_angle_calibration = np.median(pixel_to_angle_slopes)
        pixel_to_angle_calibration_deviation = Scaled_MAD(pixel_to_angle_slopes)
        
        if print_statistics:
            print()
            print("ANGULAR CALIBRATION FOR OBJECTIVE WITH NA: ", NA)
            print("Center Pixel: ", Pc)
            print("Deviation: ", Pc_deviation)
            print("Pixel To Angle Calibration: ", pixel_to_angle_calibration)
            print("Deviation: ", pixel_to_angle_calibration_deviation)
            print()
        
        return Pc, pixel_to_angle_calibration
    elif statistical_average == 'mean':
        Pc = np.mean(center_pixels)
        Pc_deviation = np.std(center_pixels)
        
        pixel_to_angle_calibration = np.mean(pixel_to_angle_slopes)
        pixel_to_angle_calibration_deviation = np.std(pixel_to_angle_slopes)
        
        if print_statistics:
            print("Center Pixel: ", Pc)
            print("Deviation: ", Pc_deviation)
            print("Pixel To Angle Calibration: ", pixel_to_angle_calibration)
            print("Deviation: ", pixel_to_angle_calibration_deviation)
            print()
        
        return Pc, pixel_to_angle_calibration

### DIFFRACTION GRATING CALIBRATION FUNCTIONS ###

def Determine_Grating_Pitch_Fourier_Space(order_positions_px, wavelength_nm = 633e-9, objective_magnification = 20, relay_magnification = 1, pixel_to_um = 3.76):
    wavelength_um = wavelength_nm * 1e6
    
    magnification_to_focal_length_map = {20 : 10, 10 : 20, 5 : 40}
    focal_length_mm = magnification_to_focal_length_map[objective_magnification]

    results = {}

    grating_pitches = []
    for m, r_px in order_positions_px.items():

        # Skip 0th order
        if m == 0:
            continue

        # Distance from 0th order
        r_px = abs(r_px)

        # Convert pixels -> um
        r_um = (r_px * pixel_to_um) / relay_magnification

        # Convert um -> mm
        r_mm = r_um / 1000

        # Calculate sin(theta)
        sin_theta = r_mm / focal_length_mm

        # Check validity
        if sin_theta >= 1:
            print(f"m={m}: invalid r/f")
            continue

        # Angle
        theta_rad = np.arcsin(sin_theta)
        theta_deg = np.degrees(theta_rad)

        # Grating equation
        #
        # d = m lambda / sin(theta)
        #
        d_um = abs(m) * wavelength_um / sin_theta
        grating_pitches.append(d_um)

    print(grating_pitches)
    avg_grating_pitch = np.mean(grating_pitches)
    print("Determined Grating Spacing: ", avg_grating_pitch, " um")

    return avg_grating_pitch

# Compute the real distances from the fourier frequencies along a given axis (x-axis or y-axis) and return the distances by reference
def Convert_Fourier_Frequencies_To_Real_Distances(peaks, axis_mask, max_order, real_space_distances, Cx, Cy, Nx, Ny):
    distances_from_center, distance_to_peak_map = Get_Axis_Filtered_Fourier_Peaks(peaks, axis_mask, Cx, Cy)
    
    for i in range(0, max_order):
        # The order of the peaks start from 1 so it will always be the index shifted up by 1
        m = i + 1
            
        # Get the distance and coordinate information for the current order
        current_distance = distances_from_center[i]
        all_coordinates_for_order = distance_to_peak_map[current_distance]
            
        # The distances are symmetric for a given order along the same axis so we only need one of their coordinates 
        x, y = all_coordinates_for_order[0]
            
        # Compute the coordinates in Fourier space
        fx = (x - Cx) / Nx
        fy = (y - Cy) / Ny
            
        # Compute the distance in Fourier space, accounting for the relay magnification of the system, and append to the results
        f = np.sqrt(fx**2 + fy**2)
            
        # To get the distance in real space we take the inverse of the distance in Fourier space and multiply by the order of the peak
        real_space_distances.append(m*(1/f))
    return
    
def Get_Axis_Filtered_Fourier_Peaks(peaks, axis_mask, Cx, Cy):
    # Get the peak coordinates
    peak_x = peaks[0]
    peak_y = peaks[1]
    
    # Filter peak coordinates
    peak_x = peak_x[axis_mask]
    peak_y = peak_y[axis_mask]

    # Calculate distance from DC
    r = np.sqrt((peak_x - Cx)**2 +(peak_y - Cy)**2)

    # Sort by distance
    peaks_ordered_by_distance = np.argsort(r)
    
    # The peaks for a given order but along different axes in real space will be slightly different
    # This is because to go to fourier space we must compute d/n where d is the distance from the center and n is the index of the pixel
    # Since the camera sensor is asymmetric, the first order points will be stretched differently along different axes
    # Therefore, we create a set of distance keys in order of increasing distance
    # The first order will be the 2 lowest distances, the second order will be the next 2 lowest distances, etc... due to the asymmetry of the axes
    # Then a distance to peak coordinate mapping is created so we can compute the fourier distance 
    distances_from_center = []
    distance_to_peak_map = {}
    for i in peaks_ordered_by_distance:
        print(f"Peak: ({peak_x[i]}, {peak_y[i]})   "f"distance = {r[i]:.1f} pixels")
        
        current_distance = round(r[i], 1)
        if current_distance not in distance_to_peak_map:
            distances_from_center.append(current_distance)
            distance_to_peak_map[current_distance] = []
        distance_to_peak_map[current_distance].append((peak_x[i], peak_y[i]))
        
    return distances_from_center, distance_to_peak_map
        
def Determine_Grating_Pitch_Real_Space(input_path, sensor_pitch = 2.2, objective_magnification = 20, relay_magnification = 1/2, max_order = 1, search_radius = 100, exclusion_radius = 15, want_plot = False):
    img = np.array(Image.open(input_path))
    
    gray_img = None
    if img.ndim == 3:
        gray_img = img.mean(axis=2)
    else:
        gray_img = img

    # Remove the average brightness
    gray_img = gray_img - gray_img.mean()

    # 2D Fourier transform
    fft = np.fft.fftshift(np.fft.fft2(gray_img))
    power = np.abs(fft)**2
    
    # Image dimensions
    Ny, Nx = gray_img.shape

    # FFT center
    Cx = Nx // 2
    Cy = Ny // 2
    
    Y, X = np.ogrid[:Ny, :Nx]
    
    distance_from_center = np.sqrt((X - Cx)**2 + (Y - Cy)**2)
    
    # Suppress DC
    power_search = power.copy()
    power_search[distance_from_center < exclusion_radius] = 0

    # Limit search radius
    power_search[distance_from_center > search_radius] = 0

    # Find local maxima
    local_max = maximum_filter(power_search, size=15) == power_search

    # Threshold to eliminate weak peaks
    threshold = power_search.max() * 0.10

    peak_locations = np.where(local_max & (power_search > threshold))

    peak_y = peak_locations[0]
    peak_x = peak_locations[1]
    
    # Maximum allowed distance (in pixels) from either FFT axis
    axis_tolerance = 5

    # Distance from the vertical and horizontal axes
    dx = np.abs(peak_x - Cx)
    dy = np.abs(peak_y - Cy)

    # Keep peaks near either the horizontal OR vertical axis
    x_axis_mask = dy <= axis_tolerance 
    y_axis_mask = dx <= axis_tolerance
    
    # Only interested in the peaks on the x-axis and y-axis
    xy_axis_mask = x_axis_mask | y_axis_mask
    
    peak_x = peak_x[xy_axis_mask]
    peak_y = peak_y[xy_axis_mask]
    x_axis_mask = x_axis_mask[xy_axis_mask]
    y_axis_mask = y_axis_mask[xy_axis_mask]
    
    filtered_peaks = (peak_x, peak_y)
    
    # Compute the real distances from the Fourier space peaks along the x-axis and y-axis respectively (We pass the distances by reference to the array)
    real_space_distances = []
    Convert_Fourier_Frequencies_To_Real_Distances(filtered_peaks, x_axis_mask, max_order, real_space_distances, Cx, Cy, Nx, Ny)
    Convert_Fourier_Frequencies_To_Real_Distances(filtered_peaks, y_axis_mask, max_order, real_space_distances, Cx, Cy, Nx, Ny)
        
    real_space_distances = np.array(real_space_distances)
    print(real_space_distances*sensor_pitch/objective_magnification)
    avg_real_space_distance = np.mean(real_space_distances)
    grating_spacing = (avg_real_space_distance * sensor_pitch) / (objective_magnification*relay_magnification) # um
    print("Grating Spacing: ", grating_spacing, ' um')
    
    if want_plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(np.log1p(power), cmap="gray")
        plt.scatter(Cx, Cy)
        plt.scatter(peak_x, peak_y)
        plt.xlim(Cx - 200, Cx + 200)
        plt.ylim(Cy - 200, Cy + 200)
        plt.xlabel("Fourier X")
        plt.ylabel("Fourier Y")
        plt.show()
    
    return grating_spacing

def Plot_Diffraction_Centers(input_file_path, final_blob_centers, normal, slope, pixel_band, angular_band, laser_wavelength, G, window_size = 20):
    # Plot the total data
    full_wavelength, full_intensity = Get_Raw_Data(input_file_path)
    full_intensity = Min_Max_Normalization(full_intensity)
    
    Px_Max = Get_Brightest_Px(full_intensity)
    intensity = full_intensity[pixel_band, Px_Max - window_size : Px_Max + window_size]
    wavelength = full_wavelength[Px_Max - window_size : Px_Max + window_size]
    
    fig, ax = plt.subplots(figsize=(15, 12), dpi = 150)
    
    # Use a log-scale or adjust vmin/vmax to see faint higher orders
    im = ax.imshow(intensity, cmap='magma', origin='lower', aspect='auto', extent=[wavelength[0],wavelength[-1],angular_band[0],angular_band[-1]])
    cbar = plt.colorbar(im, ax = ax)
    cbar.set_label('Normalized Intensity',fontsize = 20, labelpad = 15)
    cbar.ax.tick_params(labelsize = 16)
    cbar.outline.set_visible(False)

    i = 0
    for blob in final_blob_centers:
        x, y = blob['x0'], blob['y0']
        wl_x = full_wavelength[round(x)]
        
        angle_y_actual = Diffraction_Grating_Equation(blob['m'], laser_wavelength, G)
        angle_y_fit = (y - normal) * slope
        
        # Draw a green cross at the actual cluster center, a red cross at the fitted center
        if i == 0:
            ax.scatter(wl_x, angle_y_actual, color='green', marker='+', s=150, linewidth=2, label = 'actual')
            ax.scatter(wl_x, angle_y_fit, color='red', marker='+', s=150, linewidth=2, label = 'fit')
        else: 
            ax.scatter(wl_x, angle_y_actual, color='green', marker='+', s=150, linewidth=2)
            ax.scatter(wl_x, angle_y_fit, color='red', marker='+', s=150, linewidth=2)
        i += 1

    ax.set_xlabel('Wavelength (nm)', fontsize = 18)
    ax.set_ylabel('Angle (Deg)', fontsize = 18)
    ax.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 14)
    plt.legend()
    
    #plt.savefig('C:/Users/dm223/OneDrive/Documents/Wagner_Docs/Processed_Data/Reflectance_Measurements/Plots/angular_calibration/py_angle_fits/angular_fit_overlayed_onto_intensity_data.png', dpi = 300)
    
    plt.show()

# G = 0.1 gratings/micrometer
# G = 0.1 gratings/meter * 10^6
# The grating equation assuming incident light is normal
def Diffraction_Grating_Equation(m, wavelength, G):
    beta = np.rad2deg(np.arcsin((m*wavelength*G)))
    return beta

def Compute_R_Squared(y, y_fit):
    y_bar = np.mean(y)
    return 1 - np.sum((y-y_fit)**2) / np.sum((y-y_bar)**2)
    
# Determine the the change in angle as a function of y-axis pixel
def Pixel_Per_Angle_Fit(diffraction_centers, wavelength, diffraction_orders, G, wantPlot):
    # Get the angle corresponding to each diffraction order
    angles = []
    for m in diffraction_orders:
        angles.append(Diffraction_Grating_Equation(m, wavelength, G))
    angles = np.array(angles)
    
    # Fit a linear regression line to the y-axis pixels for each diffraction order centers and their corresponding angle
    slope, intercept = np.polyfit(diffraction_centers, angles, 1) 
    
    # Compute the fit for the angles
    angle_fit = slope * diffraction_centers + intercept
    
    # Compute the R^2 value
    R = Compute_R_Squared(angles, angle_fit)
    
    # If a plot is desired, plot the linear fit
    if wantPlot:
        plt.figure(figsize=(10, 7), dpi=120)
        plt.scatter(diffraction_centers, angles, color = 'red', s = 30)
        plt.plot(diffraction_centers, angle_fit, linewidth=0.7, color = 'k', linestyle = '--')
        
        plt.xlabel(rf"$P_y$", fontsize = 20)
        plt.ylabel(rf"$\theta$ (deg)", fontsize = 20)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        
        ax = plt.gca()
        ax.tick_params(direction = 'in', length = 6, width = 1, top = True, right = True, labeltop = False, labelright = False)
    
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        
        # Add text for the slope intercept and R^2 value respectively
        ax = plt.gca()
        ax.text(0.05, 0.95,rf"$\theta$ = {slope:.5f}$P_y$ + {intercept:.5f}",transform=ax.transAxes,fontsize=16,verticalalignment='top',bbox=dict(facecolor='white', alpha=0.7))
        ax.text(0.05, 0.85,f"$R^2$ = {R:.6f}",transform=ax.transAxes,fontsize=16,verticalalignment='top',bbox=dict(facecolor='white', alpha=0.7))
        
        #plt.savefig('C:/Users/dm223/OneDrive/Documents/Wagner_Docs/Processed_Data/Reflectance_Measurements/Plots/angular_calibration/py_angle_fits/diffraction_grating_angular_fit.png')
    
        plt.show()
        
    return slope, intercept
    
# Compute a watershed transform which works as follows:
# Step 1:   Turn all points which are below the intensity threshold to 0s and the anything above to a 1
# Step 2:   Compute a binary opening which consists of an erosion and dilation.
# Step 2a:  Erosion slides a small structuring element such as a 3 x 3 box all with value 1 over the intensity data. 
#           Let the original intensity matrix be I and let the structuring element be S. Then if the center of S is superimposed onto each
#           pixel of I and the center and surrounding pixels are all value 1 (matching S) then we keep the pixel in I. We denote an erosion as E(I)|S
#           Example:
#               0 0 0 1 1 1 0 0 0       1 1 1            0 0 0 0 0 0 0 0 0
#           I = 0 0 1 1 1 1 1 0 0   S = 1 1 1   E(I)|S = 0 0 0 0 1 0 0 0 0
#               0 0 1 1 1 1 1 0 0       1 1 1            0 0 0 0 1 0 0 0 0
#               0 0 0 1 1 1 0 0 0                        0 0 0 0 0 0 0 0 0
# Step 2b: Dilation slides the same structuring element back over the eroded matrix, E(I)|S. Wherever there is a 1, overlay S onto E(I)|S. 
#          We denote a dilation as D(E(I))|S
#          Example: 
#                  0 0 0 1 1 1 0 0 0
#     D(E(I))|S =  0 0 0 1 1 1 0 0 0    Notice that the binary opening (Erosion + Dilation) removes the fringes of the structure. 
#                  0 0 0 1 1 1 0 0 0    This helps separate clusters which are close together and separated by small bridges
#                  0 0 0 1 1 1 0 0 0
# Step 3: Compute a distance map. To do so, compute how many pixels away from a 0 each 1 is
# Step 4: Determine the seeds which represent each cluster. To do this, find peaks in the distance map and give each peak a label
# Step 5: Invert the distance map so that the seeds become the lowest points
# Step 6: Perform the watershed transform. The 0s (infinities in inverted map) are sources of "water" which flow towards the closest seed.
#         This "water" spreads out towards the seed and the spread of all the "water" heading towards a seed determines each cluster
#         The "water" is represented by the label of the seed. It spreads towards all nearby pixels with a lower inverse distance
#         on the inverse distance map until the seed is reached.
def Watershed_Transform(intensity, intensity_threshold = 0.02, peak_threshold = 0.05, min_distance = 13, num_opens = 3):    
    # First we threshold the data utilizing the binary opening function from scipy
    # This erodes some data with each threshold iteration and also dilates data to help separate the diffraction orders
    mask = intensity > intensity_threshold
    
    clean_mask = binary_opening(mask, iterations = num_opens)
    
    # Compute a distance map from the data above the threshold to the masked region (data below the threshold)
    distance_map = distance_transform_edt(clean_mask)
    
    # Utilize a peak finding algorithms to determine local maxima in the distance map, these will be the seeds for the watershed algorithm 
    coords = peak_local_max(distance_map, min_distance = min_distance, labels = clean_mask, threshold_rel = peak_threshold)
    
    # Each local max becomes a seed, we give each seed a label here
    seeds = np.zeros_like(distance_map, dtype=int)
    for i, (y, x) in enumerate(coords, start=1):
        seeds[y, x] = i
        
    # Create an inverted distance map, the water falls into the valleys, this determines the labels
    inverted_distance_map = -distance_map
    labels = watershed(inverted_distance_map, seeds, mask = clean_mask)
    
    plt.imshow(labels, cmap='magma', origin='lower', aspect='auto')
    plt.show()
    
    return labels

# To determine the cluster centers, compute the COM of the intensity of each cluster
def Fit_Intensity_Clusters_COM(intensity, label_map):
    blob_centers = []
    labels = np.unique(label_map)
    height, width = intensity.shape
    for label in set(labels):
        # Skip noise
        if label == 0: 
            continue 

        # 1. Isolate pixels in this cluster
        mask = label_map == label
        if np.sum(mask) == 1:
            label_map[label_map == label] = 0
            continue
        
        Y, X = np.where(mask)
        I = intensity[mask]
        
        # Background subtraction
        background = np.percentile(I, 10)

        I = I - background
        I[I < 0] = 0

        # Threshold the lowest intensity fringes
        threshold_fraction = 0.1
        keep = I > threshold_fraction * np.max(I)

        X = X[keep]
        Y = Y[keep]
        I = I[keep]
        
        # Compute the center of mass of the intensity
        X_COM = np.sum(X * I)/np.sum(I) 
        Y_COM = np.sum(Y * I)/np.sum(I)
        
        # Ensure that the determined center of mass is within the bounds of the pixel array
        # If it is, append the current cluster label, center positions, and lower and upper bounds to the cluster (blob) information
        if X_COM < 0 or X_COM > width:
            continue
        elif Y_COM < 0 or Y_COM > height:
            continue
        else:
            blob_centers.append(({'label': label,'x0': X_COM, 'y0': Y_COM, 'y_upper': np.max(Y), 'y_lower': np.min(Y)}))
        
    return blob_centers

# If a cluster is located within another cluster, merge the clusters together
def Merge_Internal_Blobs(blob_centers, label_map):
    # Get the center, upper bound, lower bound, and label of each cluster
    y_centers = []
    y_uppers = []
    y_lowers = []
    labels = []
    for b in blob_centers:
        if b['label'] == 0:
            continue
        labels.append(b['label'])
        y_centers.append(b['y0'])
        y_uppers.append(b['y_upper'])
        y_lowers.append(b['y_lower'])
    labels = np.array(labels)
    y_centers = np.array(y_centers)
    y_uppers = np.array(y_uppers)
    y_lowers = np.array(y_lowers)
        
    # Create a set to hold merged cluster labels. If we loop onto a label that has already been merged, skip it
    merged_labels = set()
    for i in range(0, len(labels)):
        current_label = labels[i]
        if current_label in merged_labels:
            continue
        
        # Get the upper and lower bounds of the current cluster
        current_upper_bound = y_uppers[i]
        current_lower_bound = y_lowers[i]
        
        # We are looking for clusters whose center is within the range of the upper and lower bounds of the current cluster
        # We filter out matches of the current cluster by requesting that the label is different from the current label
        mask = (y_centers < current_upper_bound) & (y_centers > current_lower_bound) & (labels != current_label)
        internal_blob_labels = labels[mask]
        
        # If a match is found, set the matching label to the current label to merge the cluster
        # Update the fact that this cluster is now merged so we skip it later if the label is looped over
        if len(internal_blob_labels) > 0:
            label_map[np.isin(label_map, internal_blob_labels)] = current_label
            for internal_label in internal_blob_labels:
                merged_labels.add(internal_label)

    return label_map

def Merge_Close_Blobs(blob_centers, label_map, distance_threshold = 20):
    # Get the labels and the XY-coordinates of each cluster center
    center_coords = []
    labels = []
    for b in blob_centers:
        if b['label'] == 0:
            continue
        labels.append(b['label'])
        center_coords.append((b['x0'], b['y0']))
    center_coords = np.array(center_coords)
    
    # Compute a distance between each cluster
    distance_map = distance_matrix(center_coords, center_coords)

    # If two clusters have a distance between them that is below the distance threshold, they are considered one cluster so merge them
    for i in range(len(distance_map)):
        for j in range(i+1, len(distance_map)):
            if distance_map[i,j] <= distance_threshold:
                label_map[label_map == labels[j]] = labels[i]
    return label_map
    
def Get_Cluster_Centers_Watershed(file_path, intensity_threshold, window_size = 20, distance_threshold = 20, peak_threshold = 0.05, min_distance = 13, num_opens = 3, bounds = None):
    # Get the data and normalize it to make analysis easier
    wavelengths, full_intensity = Get_Raw_Data(file_path)
    full_intensity = Min_Max_Normalization(full_intensity)
    
    plt.imshow(full_intensity, cmap='magma', origin='lower', aspect='auto')
    plt.show()
    
    # Subset the data so we only analyze the x-axis pixels which contain signal from the diffraction measurements
    # We get the brightest pixel along the x-axis and look at a window 15 pixels away in each direction from the brightest pixel
    Px_Max = Get_Brightest_Px(full_intensity)
    intensity = full_intensity[:, Px_Max - window_size : Px_Max + window_size]
    center_wavelength = wavelengths[Px_Max]*1e-9
    
    # Compute a watershed transform to get an initial estimate for the pixel clusters corresponding to each diffraction order
    # Fit the center of mass of the intensity to get the center of each cluster
    # Then merge any clusters which are within another cluster 
    # Also merge any clusters which are close enough within some threshold to be considered 1 cluster
    # After the mergine, compute the cluster centers again through a center of intensity fit
    label_map = Watershed_Transform(intensity, intensity_threshold, peak_threshold, min_distance, num_opens)
    blob_centers = Fit_Intensity_Clusters_COM(intensity, label_map)
    label_map = Merge_Internal_Blobs(blob_centers, label_map)
    label_map = Merge_Close_Blobs(blob_centers, label_map, distance_threshold)      
    final_blob_centers = Fit_Intensity_Clusters_COM(intensity, label_map)
    
    #print(f"Precisely located {len(final_blob_centers)} centers.")
    xy_center_positions = []
    for blob in final_blob_centers:
        # Correct for the fact that we subset the data around the diffraction order
        blob['x0'] += (Px_Max - window_size)
        
        # Append the corrected xy-center positions
        xy_center_positions.append((blob['x0'], blob['y0']))
    
    # Get the coords for non-background labels which 
    y_coords, x_coords = np.where(label_map > 0)
    
    return center_wavelength, xy_center_positions

def Get_Cluster_Centers_Peak_Fitting(file_path, window_size = 20, peak_threshold = 0.03, distance_threshold = 10, smoothing_sigma = 2, wantPlot = False):
    # Get the data and normalize it to make analysis easier
    wavelengths, full_intensity = Get_Raw_Data(file_path)
    full_intensity = Min_Max_Normalization(full_intensity)
    
    #plt.imshow(full_intensity, cmap='magma', origin='lower', aspect='auto')
    #plt.show()
    
    # Subset the data so we only analyze the x-axis pixels which contain signal from the diffraction measurements
    # We get the brightest pixel along the x-axis and look at a window 20 pixels away in each direction from the brightest pixel
    Px_Max = Get_Brightest_Px(full_intensity)
    intensity = full_intensity[:, Px_Max - window_size : Px_Max + window_size]
    center_wavelength = wavelengths[Px_Max]*1e-9
    
    smoothed_averaged_spectrum = gaussian_filter1d(np.mean(intensity, axis = 1), sigma = smoothing_sigma)
    peaks, properties = find_peaks(smoothed_averaged_spectrum, prominence = peak_threshold, distance = distance_threshold)
    normal_pixel = peaks[np.argmax(properties['prominences'])]

    if wantPlot:
        plt.figure(figsize=(10, 6), dpi=120)
        plt.plot(np.arange(0, len(smoothed_averaged_spectrum)) - normal_pixel, smoothed_averaged_spectrum, linewidth = 1.5, color = '#0B3C5D')
        plt.scatter(peaks - normal_pixel, smoothed_averaged_spectrum[peaks], marker = '+', color = 'r')
        plt.xlabel("Py", fontsize = 18)
        plt.ylabel("Intensity (%)", fontsize = 18)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        
        ax = plt.gca()
        ax.tick_params(direction = 'in', length = 6, width = 1, top = True, right = True, labeltop = False, labelright = False)
    
        plt.grid(True, linestyle = "--", alpha = 0.4)
        plt.tight_layout()
        plt.show()
    
    xy_center_positions = []
    for p in peaks:
        xy_center_positions.append((Px_Max, p))
        
    return center_wavelength, xy_center_positions

def Diffraction_Grating_Calibration_Hub(input_directory, grating_spacing_input_file_path, numerical_aperture, fit_type, sensor_pitch, objective_magnification, relay_magnification, window_size = 20, intensity_threshold = 0.03, wantPlot = False):    
    # Turn the input directory into a path object
    input_path = Path(input_directory)
    
    # Get all the files in the directory
    input_files = [f for f in input_path.iterdir() if f.is_file()]
    
    # Create lists to store results
    central_pixels = []
    pixel_to_angle_calibrations = []
    linear_deviations = []
    
    for intensity_input_file_path in input_files:
        center_wavelength = None
        coordinates = None
        
        if fit_type == 'watershed':
            center_wavelength, coordinates = Get_Cluster_Centers_Watershed(intensity_input_file_path, intensity_threshold = 0.04, window_size = window_size, distance_threshold = 50, peak_threshold = 0.1, min_distance = 13, num_opens = 3, bounds = None)
        elif fit_type == 'peak_fitting':
            center_wavelength, coordinates = Get_Cluster_Centers_Peak_Fitting(intensity_input_file_path, peak_threshold = intensity_threshold, window_size = window_size, wantPlot = wantPlot)
            
        # Get the y-axis center coordinates only from the coordinate list, then sort by y-axis pixel
        # The center coordinate becomes the zeroth order. Then we split into the negative angles (Below zeroth order) and positive angles (Above zeroth order)
        centers = []
        for c in coordinates:
            centers.append(c[-1])
            
        # Sort the array, the normal pixel is the pixel which corresponds to the 0th order which is the most intense portion of the diffraction spectrum
        # Subtract all pixels by the normal so that the fit is zeroed. This way the slope-intercept is minimized and easier to analyze quality of fit.
        brightest_Py = Get_Brightest_Py(intensity_input_file_path)
        centers = np.sort(np.array(centers))
        normal_idx = np.argmin(np.abs(centers - brightest_Py))
        normal_pixel = round(centers[normal_idx])
        centers -= normal_pixel
        
        # Get the diffraction orders which correspond to each center
        diffraction_orders = np.zeros(len(centers))
        
        # Assign positive orders
        for i in range(normal_idx + 1, len(centers)):
            diffraction_orders[i] = diffraction_orders[i - 1] + 1

        # Assign negative orders
        for i in range(normal_idx - 1, -1, -1):
            diffraction_orders[i] = diffraction_orders[i + 1] - 1
            
        # Map each y-axis pixel (In natural coordinates not normal centered) to its diffraction order and angle
        center_to_angle_map = {}
        for i in range(0, len(centers)):
            center_to_angle_map[centers[i] + normal_pixel] = diffraction_orders[i]
            
        # Map each order to its corresponding y-axis pixel (Helps with checking what effective magnification is acting on our grating)
        diffraction_order_to_center_map = {}
        for i in range(0, len(centers)):
            diffraction_order_to_center_map[diffraction_orders[i]] = centers[i] 
            
        # Compute the grating pitch and then the grating constant in grating/m
        grating_pitch = Determine_Grating_Pitch_Fourier_Space(order_positions_px = diffraction_order_to_center_map, wavelength_nm = center_wavelength, objective_magnification = objective_magnification, relay_magnification = relay_magnification)
        G = 1/(grating_pitch*1e-6) # grating / m
            
        # Compute the grating spacing and then the grating constant in grating/m
        #grating_spacing = Get_Grating_Spacing(grating_spacing_input_file_path, sensor_pitch, magnification, wantPlot) # um / grating
        #G = 1/(grating_spacing*1e-6) # grating / m 
        
        # Fit both the negative (lower) angles and the positive (upper) angles as a function of y-axis pixel
        # Get the slope (pixel per angle calibration) and the intercept (deviation from a linear fit)
        pixel_per_angle, linear_deviation = np.abs(Pixel_Per_Angle_Fit(centers, center_wavelength, diffraction_orders, G, wantPlot))
        
        # Get the pixels and corresponding angles that correspond to the max possible theoretical band measurable for a given normal and numerical aperture
        pixel_band, angular_band = Determine_Max_Theoretical_Signal_Band(numerical_aperture, pixel_per_angle, normal_pixel)
        
        # Get the X-Y positions for the diffraction order centers for the plotting
        blob_centers = []
        for c in coordinates:
            blob_centers.append({'x0': c[0], 'y0': c[-1], 'm': center_to_angle_map[c[-1]]})
        
        # Plot the fit on top of the raw intensity data if desired
        if wantPlot:
            Plot_Diffraction_Centers(intensity_input_file_path, blob_centers, normal_pixel, pixel_per_angle, pixel_band, angular_band, center_wavelength, G, window_size)
        
        central_pixels.append(normal_pixel)
        pixel_to_angle_calibrations.append(pixel_per_angle)
        linear_deviations.append(linear_deviation)
        
    # Get the result which produced the minimum deviation
    minimum_deviation_idx = np.argmin(np.array(np.abs(linear_deviations)))
    
    # Get the optimal central pixel and pixel to angle calibration and return
    optimal_central_pixel = central_pixels[minimum_deviation_idx]
    optimal_pixel_to_angle_calibration = pixel_to_angle_calibrations[minimum_deviation_idx]
    
    print("Optimal Central Pixel: ", optimal_central_pixel)
    print("Optimal Pixel to Angle Calibration: ", optimal_pixel_to_angle_calibration)
    print("Minimum Linear Deviation: ", linear_deviations[minimum_deviation_idx])
        
    return optimal_central_pixel, optimal_pixel_to_angle_calibration