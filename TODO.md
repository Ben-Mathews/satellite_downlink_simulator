TODO

 * Update ModulationType enum and add STATIC_CW type.
 * Update Carrier to include support for STSTIC_CW type.  The bandwidth parameter should be ignored for this type.
 * Update generate_psd() to include support for STATIC_CW type.
 * Restructure the satellite_downlink_simulator subfolder.  There should be a subfolder within it for objects, and another subfolder for simulation functionality, such as generation.py.  Split generation.py into one file that includes the PSD functionality, and another file that includes the IQ generation functionality.
 