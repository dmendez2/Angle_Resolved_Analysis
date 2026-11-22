from wavelength_calibrator import WavelengthCalibrator
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import h5py 
import re
import os

C_LIGHT = 299792458 # Speed of light (m/s)
H_PLANCK = 6.62607015e-34 # Plancks constant (J/Hz)
JOULE_TO_EV = 6.242e+18 # J --> eV conversion

GLOBAL_SUBSTRATE_REFLECTANCE_WAVELENGTHS = None
GLOBAL_SUBSTRATE_REFLECTANCE = None
GLOBAL_SUBSTRATE_WAVELENGTHS = None
GLOBAL_SUBSTRATE_INTENSITIES = None
GLOBAL_ARRAY_WAVELENGTHS = None
GLOBAL_ARRAY_INTENSITIES = None

def Set_Raw_Substrate_Parameters(wl, intens):
    global GLOBAL_SUBSTRATE_WAVELENGTHS, GLOBAL_SUBSTRATE_INTENSITIES
    GLOBAL_SUBSTRATE_WAVELENGTHS = wl
    GLOBAL_SUBSTRATE_INTENSITIES = intens
    
def Set_Raw_Array_Parameters(wl, intens):
    global GLOBAL_ARRAY_WAVELENGTHS, GLOBAL_ARRAY_INTENSITIES
    GLOBAL_ARRAY_WAVELENGTHS = wl
    GLOBAL_ARRAY_INTENSITIES = intens
    
def Set_Substrate_Reflectance_Parameters(wl, intens):
    global GLOBAL_SUBSTRATE_REFLECTANCE_WAVELENGTHS, GLOBAL_SUBSTRATE_REFLECTANCE
    GLOBAL_SUBSTRATE_REFLECTANCE_WAVELENGTHS = wl
    GLOBAL_SUBSTRATE_REFLECTANCE = intens
    
def Min_Max_Normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if min_val == max_val:
        return np.zeros_like(data, dtype=float)
    return (data - min_val) / (max_val - min_val)
    
def Plot_1D_Spectra(wavelengths_energies, reflectance, plotEnergy = False, plotReflectance = False, wantSave = False, savePath = None):
    plt.figure(figsize=(15, 8), dpi=120)
    plt.plot(wavelengths_energies, reflectance, linewidth = 1.5, color = '#0B3C5D')
    if plotEnergy:
        plt.xlabel("Energy (eV)", fontsize = 18)
    else:
        plt.xlabel("Wavelength (nm)", fontsize = 18)
        
    if plotReflectance:
        plt.ylabel("Reflectance (%)", fontsize = 18)
    else:
        plt.ylabel("Intensity", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    ax = plt.gca()
    ax.tick_params(direction = 'in', length = 6, width = 1, top = True, right = True, labeltop = False, labelright = False)
    
    plt.grid(True, linestyle = "--", alpha = 0.4)
    plt.tight_layout()
    if wantSave:
        plt.savefig(savePath)
    else:
        plt.show()
    plt.close()
    
def Get_Lasing_Peak_Central_Pixels(INTENSITY, smoothing_window = 121):
    # First we take the average of all the angular measurements so we only look at data along the wavelength axis
    # Then we take the peak to determine at what wavelength the lasing peak occurs at
    avg_intensity = Min_Max_Normalization(np.average(INTENSITY, axis = 0))
    
    # Utilize a peak-finding algorithm to find the peak of the intensity vs wavelength data and determine what wavelength the lasing peak occurs at
    peaks, peak_properties = find_peaks(avg_intensity, prominence = 0.6)
    
    # If there is no peak, return a null value
    if len(peaks) == 0:
        plt.plot(avg_intensity)
        plt.show()
        print("No Peak Found")
        return None, None
    
    lasing_peak_idx = np.argmax(peak_properties['prominences'])
    lasing_peak_Px = peaks[lasing_peak_idx]
    
    # Normalize the angular profile of the peak, then smoothen the data to make peak fitting easier. 
    # Smoothing data tends to kill some intensity and it becomes unnormalized se normalize it again to facilitate finding the central pixel trough
    lasing_peak_profile = Min_Max_Normalization(INTENSITY[:, lasing_peak_Px])
    smoothed_angular_distribution = Min_Max_Normalization(savgol_filter(lasing_peak_profile, window_length = smoothing_window, polyorder = 3))
    
    # Loop through several prominences until we find a single trough
    troughs, trough_properties = None, None
    prominence_values = [0.4, 0.3, 0.2, 0.1, 0.05]
    for p in prominence_values:
        # Negate the distribution to find the location of the minimum and determine the "true zero"
        troughs, trough_properties = find_peaks(1 - smoothed_angular_distribution, prominence = p)
        
        if len(troughs) == 1:
            break
    
    # If there is no peak, return a null value
    if len(troughs) == 0:
        plt.plot(smoothed_angular_distribution)
        plt.show()
        plt.plot(1-smoothed_angular_distribution)
        plt.show()
        print("No Trough Found")
        return None, None
    
    lasing_trough_idx = np.argmax(trough_properties['prominences'])
    lasing_trough_Py = troughs[lasing_trough_idx]
    
    return lasing_peak_Px, lasing_trough_Py
    
def Process_Dominant_Lasing_Mode(angles, INTENSITIES, wavelength_peak_Px, smoothing_window = 121):
    # Get the angular distribution at the wavelength corresponding to the lasing peak
    lasing_angular_distribution = INTENSITIES[:, wavelength_peak_Px]
    
    # Normalize the data to facilitate smoothing and peak finding 
    normalized_angular_distribution = Min_Max_Normalization(lasing_angular_distribution)
    
    # Smooth the data to facilitate peak finding
    smoothed_angular_distribution = Min_Max_Normalization(savgol_filter(normalized_angular_distribution, window_length = smoothing_window, polyorder = 3))
    
    # Determine what angles the lasing peaks at
    peaks, peak_properties = None, None
    prominence_values = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for p in prominence_values:
        peaks, peak_properties = find_peaks(smoothed_angular_distribution, prominence = p)
        
        if len(peaks) == 2:
            break
    angular_peaks = angles[peaks]
    
    return angular_peaks, lasing_angular_distribution, normalized_angular_distribution, smoothed_angular_distribution
    
def Plot_1D_Angular_Distribution(angles, angular_peaks, angular_distribution, smoothed_angular_distribution, wantSave = False, savePath = None):    
    plt.figure(figsize=(12, 6), dpi=120)
    plt.plot(angles, angular_distribution, linewidth = 1.5, alpha = 0.6, color = '#0B3C5D', label = 'Raw Angular Distribution')
    plt.plot(angles, smoothed_angular_distribution, linewidth = 1.5, color = "#ED7C18", label = 'Smoothed Angular Distribution')
 
    ax = plt.gca()
    ax.tick_params(direction = 'in', length = 6, width = 1, top = True, right = True, labeltop = False, labelright = False)
 
    # Plot angular peaks
    for i, a in enumerate(angular_peaks):
        # Determine intensity at angular peak
        peak_intensity = np.interp(a, angles, smoothed_angular_distribution)

        # Vertical dashed line
        #ax.axvline(x = a, color ='#A81313', linestyle = '--', linewidth = 1.5, alpha = 0.7, label = 'Angular Peaks' if i == 0 else None)

        # Marker at peak maximum
        ax.scatter(a, peak_intensity, color ='#A81313', s = 70, zorder = 5)

        # Annotate angular position
        ax.annotate(rf'$\theta = {a:.2f}^\circ$',xy=(a, peak_intensity),xytext=(0, 15),textcoords='offset points',ha='center',fontsize=13,color='#A81313',zorder=10,bbox=dict(facecolor='white',edgecolor='none',alpha=0.9,pad=2))
    
    plt.xlabel(rf"$\theta$ (deg)", fontsize = 18)
    plt.ylabel("Intensity", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    plt.grid(True, linestyle = "--", alpha = 0.4)
    plt.legend()
    plt.tight_layout()
    if wantSave:
        plt.savefig(savePath)
    else:
        plt.show()
    plt.close()
    
def Plot_Multi_1D_Angular_Distribution(angles, labels, angular_distributions, wantSave = False, savePath = None):
    # Generate colors from a colormap
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(labels)))
    plt.figure(figsize=(15, 8), dpi=120)
    for i in range(0, len(labels)):
        plt.plot(angles[i], angular_distributions[i], linewidth = 1.5, color = colors[i], label = f'{labels[i]}')

    plt.xlabel(rf"$\theta$ (deg)", fontsize = 18)
    plt.ylabel("Intensity", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    ax = plt.gca()
    ax.tick_params(direction = 'in', length = 6, width = 1, top = True, right = True, labeltop = False, labelright = False)
    
    plt.grid(True, linestyle = "--", alpha = 0.4)
    plt.legend(fontsize = 16, loc = 'best', frameon = False)
    plt.tight_layout()
    
    if wantSave:
        plt.savefig(savePath)
    else:
        plt.show()
    plt.close()
        
def Plot_Multi_1D_Spectra(wavelengths_energies, labels, spectra_2D, plotEnergy = False, wantSave = False, wantReflectance = False, savePath = None):
    # Generate colors from a colormap
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(labels)))
    plt.figure(figsize=(15, 8), dpi=120)
    if wantReflectance:
        for i in range(0, len(labels)):
            current_reflectance = spectra_2D[i,:]
            plt.plot(wavelengths_energies, current_reflectance, linewidth = 1.5, color = colors[i], label = f"{labels[i]}°")
    else:
        for i in range(0, len(labels)):
            plt.plot(wavelengths_energies[i], spectra_2D[i], label = f'{labels[i]}')
        
    if plotEnergy:
        plt.xlabel("Energy (eV)", fontsize = 18)
    else:
        plt.xlabel("Wavelength (nm)", fontsize = 18)
    if wantReflectance:
        plt.ylabel("Reflectance (%)", fontsize = 18)
    else:
        plt.ylabel("Intensity", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    ax = plt.gca()
    ax.tick_params(direction = 'in', length = 6, width = 1, top = True, right = True, labeltop = False, labelright = False)
    
    plt.grid(True, linestyle = "--", alpha = 0.4)
    plt.legend(fontsize = 16, loc = 'best', frameon = False)
    plt.tight_layout()
    
    if wantSave:
        plt.savefig(savePath)
    else:
        plt.show()
    plt.close()
    
def Plot_2D_Spectra(wavelengths_energies_frequencies, angular_pixel_band, Data_2D, plotEnergy = False, plotFrequency = False, plotReflectance = False, plotRaw = False, wantSave = False, savePath = None):
    # Use a log-scale or adjust vmin/vmax to see faint higher orders
    fig, ax = plt.subplots(figsize = (15, 8), dpi = 150)
    im = None
    if plotEnergy:
        sort_idx = np.argsort(wavelengths_energies_frequencies)
        energies_sorted = wavelengths_energies_frequencies[sort_idx]
        reflectance_2D_sorted = Data_2D[:, sort_idx]
        im = ax.imshow(np.transpose(reflectance_2D_sorted), cmap='magma', origin='lower', aspect='auto', vmin = np.percentile(reflectance_2D_sorted, 2), vmax = np.percentile(reflectance_2D_sorted, 98), extent=[angular_pixel_band[0],angular_pixel_band[-1], energies_sorted[0], energies_sorted[-1]])
    elif plotFrequency:
        sort_idx = np.argsort(wavelengths_energies_frequencies)
        freq_sorted = wavelengths_energies_frequencies[sort_idx]*1e14 # Rescale by multiplying by 1e14
        reflectance_2D_sorted = Data_2D[:, sort_idx]
        im = ax.imshow(np.transpose(reflectance_2D_sorted), cmap='magma', origin='lower', aspect='auto', vmin = np.percentile(reflectance_2D_sorted, 2), vmax = np.percentile(reflectance_2D_sorted, 98), extent=[angular_pixel_band[0],angular_pixel_band[-1], freq_sorted[0], freq_sorted[-1]])
    else:
        im = ax.imshow(Data_2D, cmap='magma', origin='lower', aspect='auto', vmin = np.percentile(Data_2D, 2), vmax = np.percentile(Data_2D, 98), extent=[wavelengths_energies_frequencies[0],wavelengths_energies_frequencies[-1],angular_pixel_band[0],angular_pixel_band[-1]])
    
    cbar = plt.colorbar(im, ax = ax)
    if plotReflectance:
        cbar.set_label('Reflectance (%)',fontsize = 20, labelpad = 15)
    else:
        cbar.set_label('Intensity', fontsize = 20, labelpad = 15)
    cbar.ax.tick_params(labelsize = 16)
    cbar.outline.set_visible(False)
    
    if plotEnergy:
        ax.set_xlabel('Angle (deg)', fontsize = 20)
        ax.set_ylabel('Energy (eV)', fontsize = 20)
    elif plotFrequency:
        ax.set_xlabel('Angle (deg)', fontsize = 20)
        ax.set_ylabel(r'Frequency ($10^{14}$ Hz)', fontsize = 20)
    elif plotRaw:
        ax.set_xlabel('Wavelength (nm)', fontsize = 20)
        ax.set_ylabel('Pixel', fontsize = 20)
    else:
        ax.set_xlabel('Wavelength (nm)', fontsize = 20)
        ax.set_ylabel('Angle (Deg)', fontsize = 20)
    ax.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 16)
    
    if wantSave:
        plt.savefig(savePath, dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()
    plt.close()
    
def Check_Directories_Valid(path):
    dir_path = Path(path)

    # 2. Check if it exists and is a directory
    if dir_path.is_dir():
        # 3. Check if it is empty --> any() checks if the generator returns at least one item
        if any(dir_path.iterdir()):
            return True
        else:
            return False
    else:
        return False

def Save_Reflectance_HDF5(filename, wavelengths, energies, frequencies, angular_band, reflectance_2D):
    with h5py.File(filename, 'w') as f:

        # Main datasets
        f.create_dataset('wavelengths', data = wavelengths)
        f.create_dataset('energies', data = energies)
        f.create_dataset('frequencies', data = frequencies)
        f.create_dataset('angular_band', data = angular_band)
        f.create_dataset('reflectance_2D', data = reflectance_2D)

        # Optional metadata
        f.attrs['description'] = '2D reflectance map from Fourier microscopy'
        f.attrs['units_wavelength'] = 'nm'
        f.attrs['units_energy'] = 'eV'
        f.attrs['units_frequency'] = 'Hz'
        f.attrs['units_angle'] = 'degrees'
        f.attrs['unit_reflectance'] = '%'
        
def Convert_Wavelength_To_Energy(wavelengths):
    return (H_PLANCK * C_LIGHT)/(wavelengths*1e-9) * JOULE_TO_EV

def Convert_Wavelength_To_Frequency(wavelengths):
    return C_LIGHT / (wavelengths*1e-9)
    
def Get_Optical_Axis(input_directory, wavelengths):
    wavelength_map = Parse_Directory_For_Central_Wavelength(input_directory)

    avg_top_row = 0
    avg_center_row = 0
    avg_bottom_row = 0
    N = 0
    for w in wavelengths:
        filename = input_directory + wavelength_map[w][0]
        with h5py.File(filename, 'r') as f:
            data = np.array(f['raw_data'])   
        data = np.squeeze(data)

        vertical_profile = np.mean(data, axis=1)
        threshold = 0.25 * np.max(vertical_profile)  # pick quarter max

        # Mask only rows above threshold
        mask = vertical_profile > threshold
        rows = np.arange(len(vertical_profile))

        # Weighted centroid within bright rows
        center_row = np.sum(rows[mask] * vertical_profile[mask]) / np.sum(vertical_profile[mask])
        center_row = int(round(center_row))
        
        rows = np.where(mask)[0]

        band_top = rows[-1]
        band_bottom = rows[0]
        band = data[band_bottom:band_top + 1, :]
    
        # Add the center row to the average
        avg_top_row += band_top
        avg_center_row += center_row
        avg_bottom_row += band_bottom
        
        N += 1
    
    # Divide by N to compute the average
    avg_top_row //= N
    avg_center_row //= N
    avg_bottom_row //= N
    
    return avg_top_row, avg_center_row, avg_bottom_row

def Parse_File_For_Data(file_name, data_columns = None, attribute_names = None):
    datasets = []
    attributes = []
    with h5py.File(file_name, 'r') as f:
        # Get the data
        for d in data_columns:
            datasets.append(np.array(f[d]))
            
        for a in attribute_names:
            attributes.append(f.attrs[a])
        
    return datasets, attributes

def Parse_Directory_For_Central_Wavelength(directory):
    pattern = re.compile(r'_(\d+)_nm_centered')
    
    wavelength_map = {}
    for fname in os.listdir(directory):
        match = pattern.search(fname)
        if match:
            wl = int(match.group(1))
            wavelength_map.setdefault(wl, []).append(fname)
            
    return wavelength_map

# Opens the hdf5 file and returns the raw data
def Get_Raw_Data(file_name, pixel_band, Px_max = 6248):
    intensity_data = None
    wavelength_data = None
    with h5py.File(file_name, 'r') as f:
        # Get the 2-D Raw Data
        raw_data = np.array(f['raw_data'])
        
        isBackgroundSubtracted = f.attrs['background_subtraction_applied']
        if not isBackgroundSubtracted:
            print("Warning Background Not Subtracted")
            
        if np.max(raw_data >= 65000):
            print("Warning Overflow may Have Occurred for file: ", file_name)
        
        # Get the Calibration Parameters
        Pc = f.attrs["central_pixel"]
        Dv = f.attrs["deviation_angle (rads)"]
        gamma = f.attrs["tilt_angle (rads)"]
        lambda_c = f.attrs["central_wavelength (nm)"]
        
    intensity_data = np.squeeze(raw_data[:, pixel_band])
        
    pixels = np.arange(0, Px_max, 1)
    calibrator = WavelengthCalibrator()
    calibrator.Set_Free_Parameters({'Pc': Pc, 'Dv': Dv, 'gamma': gamma, 'lambda_c': lambda_c})
    wavelength_data = calibrator.Get_Calibrated_Wavelength_From_Pixel(pixels)
        
    return wavelength_data, intensity_data

def Get_Measurements(path_to_data, central_wavelengths, pixel_band, is_directory = True):
    data_map = None
    if is_directory:
        data_map = Parse_Directory_For_Central_Wavelength(path_to_data)
    
    wavelengths = []
    intensities = []
    for w in central_wavelengths:
        wl, intens = None, None
        if is_directory:
            wl, intens = Get_Raw_Data(path_to_data + '/' + data_map[w][0], pixel_band)
        else:
            wl, intens = Get_Raw_Data(path_to_data, pixel_band)
        
        # Average out if multiple frames were taken
        if len(intens.shape) == 3:
            print("Averaging Multi-Frame Measurement")
            intens = np.average(intens, axis = 0)
        
        wavelengths.append(wl)
        intensities.append(intens)
    
    return wavelengths, intensities

# Computing the S-Polarized Reflectance from the Fresnel Equations
# n_i is the index of refraction for the material light was initially traveling through, in this case air
# n_f is the index of refraction for the final material light interacts with, in this case InP
def Get_S_Polarized_Reflectance(theta_i, n_i, n_f):
    theta_i = np.deg2rad(theta_i)
    cos_theta_f = np.sqrt( 1 - ( (n_i / n_f) * np.sin(theta_i))**2)
    return np.abs( (n_i * np.cos(theta_i) - n_f * cos_theta_f) / (n_i * np.cos(theta_i) + n_f * cos_theta_f ))**2

# Computing the P-Polarized Reflectance from the Fresnel Equations
def Get_P_Polarized_Reflectance(theta_i, n_i, n_f):
    theta_i = np.deg2rad(theta_i)
    cos_theta_f = np.sqrt( 1 - ( (n_i / n_f) * np.sin(theta_i))**2)
    return np.abs((n_i * cos_theta_f - n_f * np.cos(theta_i)) / (n_i * cos_theta_f + n_f * np.cos(theta_i)))**2

# Utilize the Fresnel equations to compute the reflectance for s-polarized or p-polarized light at arbitrary incident angle for Indium Phosphide
# The complex indices of refraction utilized are from Aspnes & Studna (1983)
def Get_InP_Reflectance(InP_reflectance_file_path, polarization, theta, wl_min = 600, wl_max = 1000):
    # Read in the data
    df = pd.read_csv(InP_reflectance_file_path)
    wavelengths = df['wl (nm)']
    
    # Get the real and imaginary component of the refractive index. 
    # The complex sum of these two components makes up the total refractive index for InP
    n = df['n'].to_numpy()
    k = df['k'].to_numpy()
    n_InP = n + 1j*k
    
    # Average value for index of refraction in air is 1.0003 for our wavelength range
    n_air = 1.0003
    
    # Compute the reflectance from the Fresnel equations based on the polarization requested
    R = None
    if(polarization == 's' or polarization == 's_pol' or polarization == 's_polarization'):
        R = Get_S_Polarized_Reflectance(theta, n_air, n_InP)
    elif(polarization == 'p' or polarization == 'p_pol' or polarization == 'p_polarization'):
        R = Get_P_Polarized_Reflectance(theta, n_air, n_InP)
    else: 
        return
    
    # Interpolate the reflectance for the range of desired wavelengths
    desired_wavelengths = np.arange(np.min(wavelengths), np.max(wavelengths) + 1, 1)
    R_interp = np.interp(desired_wavelengths, wavelengths, R)
    
    # Mask the wavelengths and reflectance to only return the data in the desired wavelength range
    mask = (desired_wavelengths >= wl_min) & (desired_wavelengths <= wl_max)
    desired_wavelengths = desired_wavelengths[mask]
    R_interp = R_interp[mask]
    
    return desired_wavelengths, R_interp

def Get_Maximum_Intensity(Py):
    max_intensity = 0
    for intens in GLOBAL_SUBSTRATE_INTENSITIES:
        current_max_intensity = np.max(intens[Py, :])
        
        if current_max_intensity > max_intensity:
            max_intensity = current_max_intensity
            
    return max_intensity

# A Hanning window is a cosine blending window that is 1 at the center and transitions to 0 at the edges
def Hanning_Window(x):
    return 0.5 * (1 - np.cos(x))

# Create a blending window which is a set of weights per each measurement 
# The weights determine how much of each measurement that overlap from the left and the right to weight
# Weights are higher for areas near the center of a measurement and lower for areas near the edges of a measurement
# The sum of the weights at a given wavelength should sum to 1
# This function determines the overlaps and creates a set of normalized Hanning windows to achieve the above goals
# Example of Hanning Window Blending (Here we assume overlaps are equivalent per window):
#                n-1 window             nth window          n+1 window         n+2 window
#                                          1                   1
#             (Left Window)  ....      0.5   0.5           0.5   0.5    .... (Right Window)
#                                   0             0     0             0
# Utilizing these weights, we then blend together each measurement
def Blend_Windows(global_wavelength_grid, scans):
    # Create arrays for the weights,w, and for the accumulated intensities computed by Sum(I*w)
    accum = np.zeros(len(global_wavelength_grid))
    global_blending_window = np.zeros(len(global_wavelength_grid))
    
    for i, (wl, intens) in enumerate(scans):
        wl_start = wl[0]
        wl_end = wl[-1]
    
        # 1. Interpolate intensity onto global grid
        interp_intens = np.interp(global_wavelength_grid, wl, intens, left=0, right=0)
    
        # 2. Build a local blending window for this measurement
        # Create a mask for this local window by checking what parts of the global wavelength grid lie in the wavelength range of the measurement
        # The local blending window will be 0 everywhere except for the portion which match the wavelength range of the current measurement
        mask = (global_wavelength_grid >= wl_start) & (global_wavelength_grid <= wl_end)
        local_win = np.zeros_like(global_wavelength_grid)
    
        # If the mask is not empty (Ensures global wavelength grid lies within a measurement window)
        if np.any(mask):
            # Determine the physical wavelength boundaries of the overlap zones
            # Left overlap boundary (First measurement has no left boundary, all others do)
            # The overlap is the difference between the last wavelength in the previous measurement and the first wavelength in the current measurement
            if i > 0:
                prev_wl_end = scans[i-1][0][-1]
                left_overlap_wl = prev_wl_end - wl_start
            else:
                left_overlap_wl = 0
            
            # Right overlap boundary (Last measurement has no right boundary, all others do)
            # The overlap is the difference between the last wavelength in the current measurement and the first wavelength in the next measurement
            if i < len(scans) - 1:
                next_wl_start = scans[i+1][0][0]
                right_overlap_wl = wl_end - next_wl_start
            else:
                right_overlap_wl = 0
            
            # Extract the segment of the global grid where this window is active
            # Initially set the sub blending window to 1. Ensures areas where there is no overlap have a weight of 1 
            # These areas are the left edge of 1st measurement, center of all measurements, and right edge of last measurement
            sub_grid = global_wavelength_grid[mask]
            sub_win = np.ones_like(sub_grid)
        
            # Apply Raised Cosine Left Taper based on start of wavelength grid for current measurement and left overlap position
            if left_overlap_wl > 0:
                taper_mask = (sub_grid >= wl_start) & (sub_grid <= wl_start + left_overlap_wl)
                # Map wavelengths to 0 -> pi
                x = np.pi * (sub_grid[taper_mask] - wl_start) / left_overlap_wl
                sub_win[taper_mask] = Hanning_Window(x)
            
            # Apply Raised Cosine Right Taper based onend of wavelength grid for current measurement and right overlap position
            if right_overlap_wl > 0:
                taper_mask = (sub_grid >= wl_end - right_overlap_wl) & (sub_grid <= wl_end)
                # Map wavelengths to 0 -> pi
                x = np.pi * (wl_end - sub_grid[taper_mask]) / right_overlap_wl
                sub_win[taper_mask] = Hanning_Window(x)
            
            # The local window is 0 everywhere except where it overlaps with the wavelength range of the current measurement
            # Where it overlaps, it matches the values of the sub window computed above
            local_win[mask] = sub_win

        # 3. Accumulate (This is the weighted sum of the intensities) --> Sum(I*w)
        accum += interp_intens * local_win
        global_blending_window += local_win

    # Divide out the sum of the weights --> Sum(I*w)/Sum(w)
    # This is more of a formality since weights are designed to sum to 1. 
    # However, it is a safeguard for numerical inaccuracy
    return accum / global_blending_window

def Process_Individual_Data(wavelength_data, intensity_data, Py):
    scans = []

    wl_min = +np.inf
    wl_max = -np.inf
    min_spacing = np.inf
    
    if len(wavelength_data.shape) == 1:
        return wavelength_data, intensity_data[Py]

    # ===============================
    # ---- Process each window ----
    # ===============================
    for wl, intens in zip(wavelength_data, intensity_data):
        # Subset the data to only analyze a singular pixel/angle
        intens = intens[Py, :]
        
        # ============================================
        # ---- Determine true overlap region ----
        # ============================================
        wl_start = wl.min()
        wl_end   = wl.max()

        if wl_end <= wl_start:
            continue  # no overlap, skip

        # Create local common grid inside overlap
        local_spacing = np.mean(np.diff(wl))

        local_grid = np.arange(wl_start, wl_end, local_spacing)

        # ============================================
        # ---- Interpolate onto common grid ----
        # ============================================
        a_interp = np.interp(local_grid, wl, intens)

        # Append current measurement
        scans.append((local_grid, a_interp))

        wl_min = min(wl_min, local_grid[0])
        wl_max = max(wl_max, local_grid[-1])
        min_spacing = min(min_spacing, local_spacing)

    # ============================================
    # ---- Build final global grid ----
    # ============================================
    global_wavelength_grid = np.arange(wl_min, wl_max, min_spacing)

    # ============================================
    # ---- Blend all windows ----
    # ============================================
    stitched_intens = Blend_Windows(global_wavelength_grid, scans)

    return global_wavelength_grid, stitched_intens

# Process Raw Arrays when the measurements for the substrate and array at a given wavelength occured at the different calibration points through interpolation
def Process_Raw_Arrays_Different_Calibration(Py):
    global_max_intensity = Get_Maximum_Intensity(Py)

    scans = []

    wl_min = +np.inf
    wl_max = -np.inf
    min_spacing = np.inf

    # ===============================
    # ---- Process each window ----
    # ===============================
    for a_wl, s_wl, a_int, s_int in zip(GLOBAL_ARRAY_WAVELENGTHS, GLOBAL_SUBSTRATE_WAVELENGTHS, GLOBAL_ARRAY_INTENSITIES, GLOBAL_SUBSTRATE_INTENSITIES):

        # --- Subset for a certain pixel/angle ---
        a_int = a_int[Py, :]
        s_int = s_int[Py, :]
            
        # ============================================
        # ---- Determine true overlap region ----
        # ============================================
        wl_start = max(s_wl.min(), a_wl.min())
        wl_end   = min(s_wl.max(), a_wl.max())

        if wl_end <= wl_start:
            continue  # no overlap, skip

        # Create local common grid inside overlap
        local_spacing = min(np.mean(np.diff(s_wl)), np.mean(np.diff(a_wl)))

        local_grid = np.arange(wl_start, wl_end, local_spacing)

        # ============================================
        # ---- Interpolate BOTH onto common grid ----
        # ============================================
        a_interp = np.interp(local_grid, a_wl, a_int, left = 0, right = 0)
        s_interp = np.interp(local_grid, s_wl, s_int, left = 0, right = 0)

        # Avoid divide-by-zero and remove data that has very low signal, since noise dominates in these areas
        # This noise can make the sample signal larger than the substrate. We therefore remove and interpolate for these points instead
        threshold = 0.01 * global_max_intensity
        good_indices = s_interp > threshold
        bad_indices = ~good_indices

        intens_ratio = np.zeros_like(s_interp)
        intens_ratio[good_indices] = a_interp[good_indices]/s_interp[good_indices]
        intens_ratio[bad_indices] = np.interp(local_grid[bad_indices], local_grid[good_indices], intens_ratio[good_indices])

        # ============================================
        # ---- Apply windowing for stitching ----
        # ============================================
        scans.append((local_grid, intens_ratio))

        wl_min = min(wl_min, local_grid[0])
        wl_max = max(wl_max, local_grid[-1])
        min_spacing = min(min_spacing, local_spacing)

    # ============================================
    # ---- Build final global grid ----
    # ============================================
    global_wavelength_grid = np.arange(wl_min, wl_max, min_spacing)

    # ============================================
    # ---- Blend all windows ----
    # ============================================
    stitched_intens_ratio = Blend_Windows(global_wavelength_grid, scans)
    
    # Load the known reflectance data for the substrate and interpolate it onto the grid
    # Keep in mind R_Substrate = I_Substrate / I_0
    r_wl = GLOBAL_SUBSTRATE_REFLECTANCE_WAVELENGTHS
    r_int = GLOBAL_SUBSTRATE_REFLECTANCE[Py, :]
    R_substrate = np.interp(global_wavelength_grid, r_wl, r_int)
    
    # Compute the reflectance
    # I_Sample / I_0 =  I_Sample / I_Substrate * I_Substrate / I_0 = I_Sample / I_Substrate * R_Substrate
    reflectance = stitched_intens_ratio * R_substrate

    # Smooth the data
    reflectance = savgol_filter(reflectance, 101, 3, mode = 'nearest')
    
    # ============================================================
    # ---- GLOBAL PERCENTAGE TRIMMING ----
    # ============================================================
    # Calculate total spectrum span
    total_span_nm = wl_max - wl_min
    
    global_left_discard_pct = 0.05
    global_right_discard_pct = 0.05 
    
    # Map percentage to physical nanometers to discard from boundaries
    discard_nm_left = total_span_nm * global_left_discard_pct
    discard_nm_right = total_span_nm * global_right_discard_pct
    
    # Compute new global cutoff bounds
    global_min_cutoff = wl_min + discard_nm_left
    global_max_cutoff = wl_max - discard_nm_right
    
    # Print diagnostics so you know what physical window is kept vs discarded
    #print(f"\n--- Global Trim Diagnostics ---")
    #print(f"Raw range: {wl_min:.2f} nm to {wl_max:.2f} nm (Span: {total_span_nm:.2f} nm)")
    #print(f"Discarding left: {global_left_discard_pct*100}% (~{discard_nm_left:.2f} nm)")
    #print(f"Discarding right: {global_right_discard_pct*100}% (~{discard_nm_right:.2f} nm)")
    #print(f"Clean target output range: {global_min_cutoff:.2f} nm to {global_max_cutoff:.2f} nm\n")

    # Mask and slice the final global arrays
    global_trim_mask = (global_wavelength_grid >= global_min_cutoff) & (global_wavelength_grid <= global_max_cutoff)
    global_wavelength_grid = global_wavelength_grid[global_trim_mask]
    reflectance = reflectance[global_trim_mask]
    
    # Convert to a percentage
    reflectance *= 100
    
    return global_wavelength_grid, reflectance