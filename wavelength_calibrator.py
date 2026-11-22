import tifffile
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, linear_sum_assignment
from lmfit import minimize, Parameters
from scipy.ndimage import gaussian_filter1d

### All Internal Variables Length Scale Held at Millimeters (mm) For Ease of Calculations ###
### Outputted Variables Converted to Nanometers (nm) ###
### "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon is Frequently Cited and Will be Referred to as TOOS in Shorthand ###
class WavelengthCalibrator():

    def __init__(self):
        ### Initial Parameters According to ISA HR-320 Manual ###
        self.k = 1 # First Order Wavelength
        self.n = 1200 # Grating Period (g/mm)
        self.F = 320 # Focal Length of Spectrograph (mm)
        self.Dv = 24 * (np.pi/180) # Deviation Angle (Radians)
        self.gamma = 2.417 * (np.pi/180) # Rotation of grating relative to focal plane (Radians)
        self.width = 68 # Width of the Grating (mm)

        ### Initial Parameters According to ZWO Astro ASI2600 Pro Camera Documentation ###
        self.Pw = 0.00376 # Pitch of Detector --> The spacing between pixels (mm)
        self.P_Min = 0 # Pixel Number Corresponding to Minimum Wavelength
        self.P_Max = 6247 # Pixel Number Corresponding to Maximum Wavelength
        self.Pc = 1593 # Center Pixel Number (Actual value according to documentation is 3124 but according to measurements done with a He-Ne Laser with single line at 632.8 nm, the true center is at about 1590)
        self.max_camera_intensity = 65535 # Maximum intensity which the 16-bit Camera can detect


        ### Wavelength Parameters ###
        self.lambda_c = None # Central Wavelength of Spectrometer (mm)
        self.Wl_Min = None # The Minimum Wavelength Detectable by the Spectrometer (mm)
        self.Wl_Max = None # Maximum Wavelength Detectable by the Spectrometer (mm)

        ### The Reference Ne I spectra from NIST ###
        self.reference_spectra = None # Reference Ne I spectral line wavelength --> From NIST (nm) later converted to (mm)

    # Reset all variables to default manufacturer variables
    def Reset(self):
        self.Pc = 1593 # Central Pixel
        self.Dv = 24 * (np.pi/180) # Deviation Angle (Radians)
        self.gamma = 2.417 * (np.pi/180) # Rotation of grating relative to focal plane (Radians)

    ### Function to Match Computed Lines to Nearest NIST Reference Line ###
    # We match the lines by doing a linear sum assignment which implements the Hungarian Algorithm
    # We first compute a distance matrix between each computed line and the NIST reference lines
    # We then subtract the minimum distance of each row from that row
    # We then look at the zeroes along each column. If there are N columns with a zero for N computed lines, then we have a globally optimal matching
    # We then choose a zero in each column such that the zeroes chosen do not share a column with each other

    ### Example ###
    # Rows are computed lines, columns are reference lines

    # Distance Matrix:
    # 1     9      29      49
    # 20    10     10      30
    # 40    30     10      10

    # Subtract minimum from each row
    # 0     8    28    48
    # 10    0    0     20
    # 30    20   0     0

    # We then choose columns such that the zeroes chosen do not share a column with each other
    # We match row 1 to column 1
    # We match row 2 to column 2 OR 3
    # We match row 3 to column 3 OR 4 (Must be column 4 if row 2 was matched to column 3)
    def Process_Reference_Spectra(self, lambda_0, Wl_Min, Wl_Max, Wl_buffer = 2.0*1e-6):
        # Filter the reference spectrum to only include the wavelengths between the theoretical minimum and maximum wavelengths (plus or minus a buffer) calculated by the calibrator algorithm with the initial parameters as provided by the spectrometer and camera parameters
        Wl_Min -= Wl_buffer
        Wl_Max += Wl_buffer
        spectra = self.reference_spectra[(self.reference_spectra['obs_wl_air(mm)'] > Wl_Min) & (self.reference_spectra['obs_wl_air(mm)'] < Wl_Max)]

        # Filter the spectra so it only contains wavelengths between the minimum and maximum wavelength
        spectra = self.reference_spectra[(self.reference_spectra['obs_wl_air(mm)'] > Wl_Min) & (self.reference_spectra['obs_wl_air(mm)'] < Wl_Max)]

        # Sort the spectra by wavelength and convert the wavelengths and intensities to a numpy array
        spectra = spectra.sort_values(by = 'obs_wl_air(mm)')
        NIST_wavelengths = spectra['obs_wl_air(mm)'].to_numpy()
        intensities = spectra['intens'].to_numpy()

        # Sort the approximate wavelengths and NIST wavelengths to make matching possible through the Numpy searchsorted function
        lambda_0 = np.sort(lambda_0)

        # Create a distance matrix which keeps track of the absolute difference between the computed wavelengths (lambda_0) and the reference wavelenghts (NIST)
        # Then weight the distances by their intensities. For more intense peaks, the weigthed distance become smaller. This ensures we match to the most intense peaks which are the peaks we are detecting
        distances = np.abs(lambda_0[:, None] - NIST_wavelengths[None, :])
        distances = distances * 1/intensities

        # Apply the linear sum assignment algorithm to the distance matrix
        row_ind, col_ind = linear_sum_assignment(distances)

        # Subset the chosen column indices from the linear sum assignment as the matched reference lines for each respective computed line
        matched_wavelengths = NIST_wavelengths[col_ind]

        return matched_wavelengths

    # Compute a Gaussian Around the Detected Pixel
    def Gaussian(self, x, A, P0, sigma, B):
        return A*np.exp(-(x-P0)**2/(2*sigma**2)) + B

    # Function extracts peaks by searching for peaks which have a prominence greater than the minimum
    # Once peaks are determined, we apply a Gaussian fit to get the most accurate peak center
    def Extract_Peak_Centers(self, data):
        # Convert to 32-bit to prevent overflow errors
        data = data.astype(np.float32, copy = False)

        # Collapse 2D CCD data to 1D spectra by taking mean of each column
        # To ensure that we can detect peaks at any gain we normalize such that the integration over the whole spectrum equals 1
        spectrum = np.mean(data, axis = 0)
        spectrum = spectrum/np.max(spectrum)

        # Smooth the Data
        spectrum = gaussian_filter1d(spectrum, sigma = 1)
        
        # Compute the median absolute deviation (MAD)
        # The noise level (standard deviation of data) is defined as 1.4826 * MAD
        mad = np.median(np.abs(spectrum - np.median(spectrum)))
        noise = 1.4826*mad

        # Determine the Peaks, set prominence to 20 * noise
        peaks, properties = find_peaks(spectrum, prominence = 20*noise, distance = 50)
        peak_centers = []

        # Compute a Gaussian Around the Peak
        # We find the Tippy-Top of the Peak, However Due to the Discrete Nature of the Pixel Detection this can Be Choppy
        # The Tippy-Top of the Peak may not be at the Center of the Line so we fit a Gaussian in the Region Around the Peak
        # We Then Take The Gaussian Center as the True Center
        for p in peaks:
            # Choose Window Around Peak
            lo = max(0, p-7)
            hi = min(len(spectrum), p+7)
            x = np.arange(lo, hi)
            y = spectrum[x]

            # Initial Guess for Parameters
            A0 = y.max() - y.min()
            P0 = p
            sigma0 = 1.5
            B0 = y.min()

            # Fit a Gaussian to The Peak
            popt, _ = curve_fit(self.Gaussian, x, y, p0=[A0, P0, sigma0, B0])
            peak_centers.append(popt[1])

        return spectrum, np.array(peak_centers)

    # To get an intensity array, we take the mean along the X-Axis (Recall that wavelengths only change along Y-Axis)
    # Then we divide by the trapezoidal integration of the intensity array to normalize the integration to be 1
    def Process_Pixel_Intensity(self, data):
        # Convert to 32-bit to prevent overflow errors
        # Divide by the max camera intensity, to scale everything between 0 and 1
        data = data.astype(np.float32, copy = False)/self.max_camera_intensity
        return data 

    # Utilizing Equations from "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon to Compute the Wavelength For a Given Pixel Detection
    def Get_Calibrated_Wavelength_Intensity_From_Pixel(self, data):
        # Compute the average, integration normalized intensity along the X-Axis
        intensity = self.Process_Pixel_Intensity(data)

        # Create an array for every single X-axis pixel
        pixels = np.arange(self.P_Min, self.P_Max + 1, 1)

        # Return the wavelengths (nm) and the intensity
        wavelength = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, pixels) * 1e6

        # Convert to 32 bit arrays to save memory (Any additional precision is unphysical anyway)
        wavelength = wavelength.astype(np.float32)
        intensity = intensity.astype(np.float32)

        return  wavelength, intensity
    
    # Utilizing Equations from "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon to Compute the Wavelength For a Given Pixel Detection
    def Get_Calibrated_Wavelength_From_Pixel(self, P_lambda):
        alpha = self.Alpha(self.k, self.n, self.Dv, self.lambda_c)
        beta_lambda_c = self.Beta(self.k, self.n, alpha, self.lambda_c)
        beta_h = self.Beta_H(beta_lambda_c, self.gamma)
        Lh = self.LH(self.F, self.gamma)
        Hb_lambda_c = self.HB_Lambda_C(self.F, self.gamma)
        Hb_lambda_n = self.HB_Lambda_N_From_Pixel(self.Pw, P_lambda, self.Pc, Hb_lambda_c)
        beta_lambda_n = self.Beta_Lambda_N(beta_h, Hb_lambda_n, Lh)
        return self.Get_Wavelength(self.k, self.n, alpha, beta_lambda_n)*1e6

    # Utilizing Equations from "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon to Compute the Wavelength For a Given Pixel Detection
    def Get_Wavelength_From_Pixel(self, k, n, F, Dv, gamma, Pw, Pc, lambda_c, P_lambda):
        alpha = self.Alpha(k, n, Dv, lambda_c)
        beta_lambda_c = self.Beta(k, n, alpha, lambda_c)
        beta_h = self.Beta_H(beta_lambda_c, gamma)
        Lh = self.LH(F, gamma)
        Hb_lambda_c = self.HB_Lambda_C(F, gamma)
        Hb_lambda_n = self.HB_Lambda_N_From_Pixel(Pw, P_lambda, Pc, Hb_lambda_c)
        beta_lambda_n = self.Beta_Lambda_N(beta_h, Hb_lambda_n, Lh)
        return self.Get_Wavelength(k, n, alpha, beta_lambda_n)

    # Utilizing Equations from "The Optics of Spectroscopy" by J.M. Lerner and A. Thevenon to Compute the Pixel Activated For a Given Wavelength
    def P_Lambda(self, k, n, Pw, Pc, F, Dv, gamma, lambda_c, lambda_n):
        alpha = self.Alpha(k, n, Dv, lambda_c)
        beta_lambda_c = self.Beta(k, n, alpha, lambda_c)
        beta_lambda_n = self.Beta(k, n, alpha, lambda_n)
        beta_h = self.Beta_H(beta_lambda_c, gamma)
        Lh = self.LH(F, gamma)
        Hb_lambda_c = self.HB_Lambda_C(F, gamma)
        Hb_lambda_n = self.HB_Lambda_N_From_Beta(Lh, beta_h, beta_lambda_n)
        return (Hb_lambda_c - Hb_lambda_n)/Pw + Pc

    # The Small Wavelength Difference Which Cannot be Resolved by the Diffraction Grating (Equation 5-11 in TOOS)
    def Dlambda(self, n, width, lambda_c):
        return lambda_c/(n*width)

    # The Angle for the Incoming Wavelength (Equation 2-1 in TOOS)
    def Alpha(self, k, n, Dv, lambda_n):
        return np.arcsin((k*n*lambda_n)/(2 * np.cos(Dv/2))) - Dv/2

    # The Angle for the Outgoing Wavelength (Equation 1-1 in TOOS)
    def Beta(self, k, n, alpha, lambda_n):
        return np.arcsin(k*n*lambda_n - np.sin(alpha))

    # The Angle from LH to the Normal to the Grating (Equation 5-4 in TOOS)
    def Beta_H(self, beta_lambda_c, gamma):
        return gamma + beta_lambda_c

    # The Outgoing Angle for Off-Center Wavelengths (Equation 5-7 in TOOS)
    def Beta_Lambda_N(self, beta_h, Hb_lambda_n, Lh):
        return beta_h - np.arctan(Hb_lambda_n/Lh)

    # Perpendicular Distance From Grating or Focusing Mirror to the Focal Plane (Equation 5-3 in TOOS)
    def LH(self, F, gamma):
        return F * np.cos(gamma)

    # Distance From the Intercept of the Normal to the Focal Plane to the Central Wavelength (Equation 5-5 in TOOS)
    def HB_Lambda_C(self, F, gamma):
        return F * np.sin(gamma)

    # Distance From the Intercept of the Normal to the Focal Plane to the Central Wavelength Computed From Outgoing Angle, Beta_Lambda_N (Equation 5-8 in TOOS)
    def HB_Lambda_N_From_Beta(self, Lh, beta_h, beta_lambda_n):
        return Lh * np.tan(beta_h - beta_lambda_n)

    # Distance From the Intercept of the Normal to the Focal Plane to the Central Wavelength Computed From Activated Pixel, P_Lambda (Equation 5-6 in TOOS)
    def HB_Lambda_N_From_Pixel(self, Pw, P_lambda, Pc, Hb_lambda_c):
        return Pw * (Pc - P_lambda) + Hb_lambda_c

    # Compute the Wavelength from the Incoming and Outgoing Angles of the Light Utilizing the Grating Equation (Equation 1-1 in TOOS)
    def Get_Wavelength(self, k, n, alpha, beta):
        return (np.sin(alpha) + np.sin(beta))/(k*n)

    # Get wavelength range in nanometers (nm)
    def Get_Wavelength_Range(self):
        self.Wl_Min = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Min)
        self.Wl_Max = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Max)
        return (self.Wl_Min*1e6, self.Wl_Max*1e6)

    # Get the central wavelength in nanometers (nm)
    def Get_Central_Wavelength(self):
        return self.lambda_c * 1e6

    def Get_Free_Parameters(self):
        return {'Pc': self.Pc, 'Dv': self.Dv, 'gamma': self.gamma, 'lambda_c': self.lambda_c}

    def Set_Free_Parameters(self, parameters):
        self.Pc = parameters['Pc']
        self.Dv = parameters['Dv']
        self.gamma = parameters['gamma']
        self.lambda_c = parameters['lambda_c']
        return

    # Compute the residual between the measured pixel position and the computed pixel position from the reference wavelength utilizing the current parameter combination
    def residual(self, params, P_measured, lambda_ref):
        # Unpack Parameters
        k = params['k']
        n = params['n']
        F = params['F']
        Pw = params['Pw']
        Pc = params['Pc']
        Dv = params['Dv']
        gamma = params['gamma']
        lambda_c = params['lambda_c']

        # Compute the Pixel Position for the Current Parameter Combination and Return the Residual with the Actual Pixel Positions Measured
        P_model = self.P_Lambda(k, n, Pw, Pc, F, Dv, gamma, lambda_c, lambda_ref)
        return (P_measured - P_model)

    # To Calibrate, we First Utilized a Reference He-Ne Laser with a Single Known Wavelength of 632.81646 nm
    # A Peak Finding Algorithm was Used to Find the Pixel Position of the He-Ne Line and Later, here, the Lines for the Ne Spectrum
    # The Spectrometer was Dialed to a Center Wavelength of 633 nm so the He_Ne Line is Roughly at the Center Pixel Position --> ~Pixel 1593
    # With the Center Pixel Position Determined, we can then Estimate the Wavelengths of the Ne I Spectra Lines
    # The Estimated Wavelengths are then Matched to the Closest Wavelengths in the NIST Reference Spectrum
    # The Center Pixel Position, the Deviation Angle, the central wavelength, and the Tilt Angle are then Varied and Utilized to Compute Pixel Positions Using the Matched Reference Wavelengths
    # Residuals Between the Computed Pixel Position and the True Measured Pixel Positions are then Determined
    # A Least Squares Fit is Conducted to Minimize the Residuals and Find the Best Fit Parameter Combination Which Become the Calibrated Values
    def Calibrate(self, Spectral_Data_Path, Nist_Reference_Path, lambda_c):
        # Set new central wavelength
        self.lambda_c = lambda_c*1e-6

        # Preprocess the NIST Reference Spectra
        self.reference_spectra = pd.read_csv(Nist_Reference_Path)[['obs_wl_air(nm)', 'intens']].astype(float).dropna() # Reference Ne I spectral line wavelength --> From NIST (nm)
        self.reference_spectra['obs_wl_air(nm)'] *= 1e-6 # Convert wavelengths to millimeters for later calculations (mm)
        self.reference_spectra.rename(columns = {'obs_wl_air(nm)': 'obs_wl_air(mm)'}, inplace = True) # Rename column to reflect unit change

        # Read in the data
        Spectral_Data = None
        if Spectral_Data_Path.endswith(".npy"):
            Spectral_Data = np.load(Spectral_Data_Path)
        elif Spectral_Data_Path.endswith(".tiff") or Spectral_Data_Path.endswith(".tif"):
            Spectral_Data = tifffile.imread(Spectral_Data_Path)

        # Get Pixel Dimensions
        l,w = Spectral_Data.shape

        # Extract the Peak Centers from the measured Ne Spectrum
        #prominences = [0.0005, 0.00025, 0.0001, 0.000075, 0.00005]
        #spectrum = []
        #P_measured = []
        #for p in prominences:
            #spectrum, P_measured = self.Extract_Peak_Centers(Spectral_Data, prominence = p)
            #if(len(P_measured) >= 4):
            #    break
            
        spectrum, P_measured = self.Extract_Peak_Centers(Spectral_Data)

        # Compute the minimum wavelength that can be resolved from our diffraction grating and ensure the residuals are smaller than it. Set initial residuals to infinity
        # If they are, calibration is complete. Otherwise, if 5 rounds of calibration pass without success, we consider the calibration a failure.
        lambda_ref = None
        calibrated_wavelengths = None
        d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)
        residuals = np.ones(len(P_measured)) * np.inf
        iter = 0
        while(np.any(residuals > d_lambda) and iter < 5):
            # Compute the Minimum and Maximum Wavelengths With the Current Parameters (Default Manufacturer Parameters with Corrected Central Pixel Position)
            self.Wl_Min = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Min)
            self.Wl_Max = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Max)

            # Compute the Wavelengths for the Measured Pixels With the Current Parameters (Default Manufacturer Parameters with Corrected Central Pixel Position)
            lambda_0 = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, P_measured)

            # Utilizing the Minimum and Maximum Wavelength as Bounds, Find the Best Fit NIST Reference Lines to the Current Computed Wavelengths
            lambda_ref = self.Process_Reference_Spectra(lambda_0, self.Wl_Min, self.Wl_Max)

            # Initialize the Parameters to be Utilized in the Least Squares Fit
            # The Parameters to be Fit are the Central Pixel Position, the Deviation Angle, and the Tilt Angle as these are the most likely to be deviated from stated values
            params = Parameters()
            params.add('k', value = self.k, vary = False)
            params.add('n', value = self.n, vary = False)
            params.add('F', value = self.F, vary = False)
            params.add('Pw', value = self.Pw, vary = False)
            params.add('lambda_c', value = self.lambda_c, vary = False, min = self.lambda_c - self.lambda_c*0.01, max = self.lambda_c + self.lambda_c*0.01)
            params.add('Pc', value = self.Pc, vary = True, min = self.Pc - self.Pc*0.1, max = self.Pc + self.Pc*0.1)
            params.add('Dv', value = self.Dv, vary = True, min = self.Dv - self.Dv*0.2, max = self.Dv + self.Dv*0.2)
            params.add('gamma', value = self.gamma, vary = True, min = self.gamma - self.gamma*0.3, max = self.gamma + self.gamma*0.3)

            # Compute the Fit by Miniimizing the Square of the Residual
            least_squares_fitter = minimize(self.residual, params, args=(P_measured, lambda_ref), method = "least_squares")

            # Compute the Wavelengths from the Measured Pixel Positions with the Calibrated Parameters
            calibrated_wavelengths = self.Get_Wavelength_From_Pixel(least_squares_fitter.params['k'], least_squares_fitter.params['n'], least_squares_fitter.params['F'], least_squares_fitter.params['Dv'], least_squares_fitter.params['gamma'], least_squares_fitter.params['Pw'], least_squares_fitter.params['Pc'], least_squares_fitter.params['lambda_c'], P_measured)

            # Set the Internal Parameters to the Calibrated Values
            self.Dv = float(least_squares_fitter.params['Dv'])
            self.gamma = float(least_squares_fitter.params['gamma'])
            self.Pc = int(least_squares_fitter.params['Pc'])
            self.lambda_c = float(least_squares_fitter.params['lambda_c'])

            # Compute the minimum wavelength that can be resolved from our diffraction grating
            d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)

            # Compute the residual between the reference wavelength and the calibrated wavelength
            residuals = np.abs(lambda_ref - calibrated_wavelengths)

            # Update the iteration count
            iter += 1

        # Check if calibration was complete or not by checking if the mean of the residuals are smaller than the resolution of the diffraction grating
        if(np.mean(residuals) < d_lambda):
            return calibrated_wavelengths.tolist(), lambda_ref.tolist()
        else:
            return calibrated_wavelengths.tolist(), lambda_ref.tolist()
        
    # Once the parameters have been calibrated, the computation of wavelengths should be extremely close to the true wavelengths plus/minus an offset
    # This offset is due to the central wavelength on the spectrometer dial not being completely exact
    # We therefore shift the central wavelength while ensuring we fit to Ne I reference lines which ensures we correct for the offset
    def Central_Wavelength_Shift(self, Spectral_Data_Path, Nist_Reference_Path, lambda_c):
    # Set new central wavelength
        self.lambda_c = lambda_c*1e-6
                                                          
        # Preprocess the NIST Reference Spectra
        self.reference_spectra = pd.read_csv(Nist_Reference_Path)[['obs_wl_air(nm)', 'intens']].astype(float).dropna() # Reference Ne I spectral line wavelength --> From NIST (nm)
        self.reference_spectra['obs_wl_air(nm)'] *= 1e-6 # Convert wavelengths to millimeters for later calculations (mm)
        self.reference_spectra.rename(columns = {'obs_wl_air(nm)': 'obs_wl_air(mm)'}, inplace = True) # Rename column to reflect unit change

        # Read in the data
        Spectral_Data = None
        if Spectral_Data_Path.endswith(".npy"):
            Spectral_Data = np.load(Spectral_Data_Path)
        elif Spectral_Data_Path.endswith(".tiff") or Spectral_Data_Path.endswith(".tif"):
            Spectral_Data = tifffile.imread(Spectral_Data_Path)

        # Get Pixel Dimensions
        l,w = Spectral_Data.shape

        # Extract the Peak Centers from the measured Ne Spectrum
        prominences = [0.0005, 0.00025, 0.0001, 0.000075, 0.00005]
        spectrum = []
        P_measured = []
        for p in prominences:
            spectrum, P_measured = self.Extract_Peak_Centers(Spectral_Data, prominence = p)
            if(len(P_measured) >= 2):
                break

        # Compute the minimum wavelength that can be resolved from our diffraction grating and ensure the residuals are smaller than it. Set initial residuals to infinity
        # If they are, calibration is complete. Otherwise, if 5 rounds of calibration pass without success, we consider the calibration a failure.
        lambda_ref = None
        calibrated_wavelengths = None
        d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)
        residuals = np.ones(len(P_measured)) * np.inf
        iter = 0
        while(np.any(residuals > d_lambda) and iter < 5):
            # Compute the Minimum and Maximum Wavelengths With the Current Parameters (Default Manufacturer Parameters with Corrected Central Pixel Position)
            self.Wl_Min = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Min)
            self.Wl_Max = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, self.P_Max)

            # Compute the Wavelengths for the Measured Pixels With the Current Parameters (Default Manufacturer Parameters with Corrected Central Pixel Position)
            lambda_0 = self.Get_Wavelength_From_Pixel(self.k, self.n, self.F, self.Dv, self.gamma, self.Pw, self.Pc, self.lambda_c, P_measured)

            # Utilizing the Minimum and Maximum Wavelength as Bounds, Find the Best Fit NIST Reference Lines to the Current Computed Wavelengths
            lambda_ref = self.Process_Reference_Spectra(lambda_0, self.Wl_Min, self.Wl_Max)

            # Initialize the Parameters to be Utilized in the Least Squares Fit
            # The Parameters to be Fit are the Central Pixel Position, the Deviation Angle, and the Tilt Angle as these are the most likely to be deviated from stated values
            params = Parameters()
            params.add('k', value = self.k, vary = False)
            params.add('n', value = self.n, vary = False)
            params.add('F', value = self.F, vary = False)
            params.add('Pw', value = self.Pw, vary = False)
            params.add('Pc', value = self.Pc, vary = False)
            params.add('Dv', value = self.Dv, vary = False)
            params.add('gamma', value = self.gamma, vary = False)
            params.add('lambda_c', value = self.lambda_c, vary = True, min = self.lambda_c - self.lambda_c*0.01, max = self.lambda_c + self.lambda_c*0.01)

            # Compute the Fit by Miniimizing the Square of the Residual
            least_squares_fitter = minimize(self.residual, params, args=(P_measured, lambda_ref), method = "least_squares")

            # Compute the Wavelengths from the Measured Pixel Positions with the Calibrated Parameters
            calibrated_wavelengths = self.Get_Wavelength_From_Pixel(least_squares_fitter.params['k'], least_squares_fitter.params['n'], least_squares_fitter.params['F'], least_squares_fitter.params['Dv'], least_squares_fitter.params['gamma'], least_squares_fitter.params['Pw'], least_squares_fitter.params['Pc'], least_squares_fitter.params['lambda_c'], P_measured)

            # Set the Internal Parameters to the Calibrated Values
            self.Dv = float(least_squares_fitter.params['Dv'])
            self.gamma = float(least_squares_fitter.params['gamma'])
            self.Pc = int(least_squares_fitter.params['Pc'])
            self.lambda_c = float(least_squares_fitter.params['lambda_c'])

            # Compute the minimum wavelength that can be resolved from our diffraction grating
            d_lambda = self.Dlambda(self.n, self.width, self.lambda_c)

            # Compute the residual between the reference wavelength and the calibrated wavelength
            residuals = np.abs(lambda_ref - calibrated_wavelengths)

            # Update the iteration count
            iter += 1

        # Check if calibration was complete or not by checking if the mean of the residuals are smaller than the resolution of the diffraction grating
        if(np.mean(residuals) < d_lambda):
            return calibrated_wavelengths.tolist(), lambda_ref.tolist()
        else:
            return calibrated_wavelengths.tolist(), lambda_ref.tolist()