import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

# Streamlit app configuration
st.set_page_config(
    page_title="Interactive Light Curve Generator",
    initial_sidebar_state='auto',
)

# Sidebar setup
with st.sidebar:
    st.markdown("[![GitHub](https://cdn.iconscout.com/icon/free/png-48/github-1521500-1288242.png)](https://github.com/javiserna)")
    st.write("Need help? Contact us at support@tessextractor.app.")

# Load the FITS file
fits_file = "tess-s0019-1-3_65.489205_28.443210_10x10_astrocut.fits"
try:
    fits_data = fits.open(fits_file)
    flux_data = fits_data[1].data['FLUX']  # Access all FLUX frames
    fits_data.close()
except Exception as e:
    st.error(f"Failed to load FITS file: {e}")
    flux_data = None

# Initialize session state for masks
if "aperture_mask" not in st.session_state:
    st.session_state["aperture_mask"] = np.zeros(flux_data[0].shape, dtype=int)
if "annulus_mask" not in st.session_state:
    st.session_state["annulus_mask"] = np.zeros(flux_data[0].shape, dtype=int)

# Toggle functions for aperture and annulus pixels
def toggle_aperture_pixel(row, col):
    st.session_state["aperture_mask"][row, col] = 1 - st.session_state["aperture_mask"][row, col]

def toggle_annulus_pixel(row, col):
    st.session_state["annulus_mask"][row, col] = 1 - st.session_state["annulus_mask"][row, col]

# Display the interactive image with pixel selection
st.title("Interactive Light Curve Generator")
st.write("Select aperture and annulus pixels to perform photometry on the FITS data.")

# Display the FITS image with pixel selection
fig, ax = plt.subplots(figsize=(5, 5))
norm = Normalize(vmin=np.percentile(flux_data[0], 5), vmax=np.percentile(flux_data[0], 90))
ax.imshow(flux_data[0], cmap="YlGnBu_r", norm=norm)

for row in range(flux_data[0].shape[0]):
    for col in range(flux_data[0].shape[1]):
        if st.session_state["aperture_mask"][row, col] == 1:
            ax.text(col, row, "■", ha="center", va="center", color="orange")
        elif st.session_state["annulus_mask"][row, col] == 1:
            ax.text(col, row, "■", ha="center", va="center", color="cyan")
st.pyplot(fig)

# Interactive pixel selection for aperture
st.write("### Aperture selection")
for row in range(flux_data[0].shape[0]):
    cols = st.columns(flux_data[0].shape[1])
    for col in range(flux_data[0].shape[1]):
        button_label = "⬜" if st.session_state["aperture_mask"][row, col] == 0 else "■"
        cols[col].button(
            button_label, key=f"aperture-{row}-{col}", on_click=toggle_aperture_pixel, args=(row, col)
        )

# Interactive pixel selection for annulus
st.write("### Annulus selection")
for row in range(flux_data[0].shape[0]):
    cols = st.columns(flux_data[0].shape[1])
    for col in range(flux_data[0].shape[1]):
        button_label = "⬜" if st.session_state["annulus_mask"][row, col] == 0 else "■"
        cols[col].button(
            button_label, key=f"annulus-{row}-{col}", on_click=toggle_annulus_pixel, args=(row, col)
        )

# Perform photometry using the selected masks
def compute_light_curve(flux_data, aperture_mask, annulus_mask):
    flux_aperture = []
    flux_annulus = []

    for frame in flux_data:
        aperture_flux = frame[aperture_mask == 1].sum()
        annulus_flux = frame[annulus_mask == 1].sum()
        flux_aperture.append(aperture_flux)
        flux_annulus.append(annulus_flux)

    # Subtract background (mean annulus flux)
    background_flux = np.array(flux_annulus) / np.sum(annulus_mask)
    net_flux = np.array(flux_aperture) - background_flux * np.sum(aperture_mask)

    return net_flux

# Compute the light curve
if st.button("Generate Light Curve"):
    aperture_mask = st.session_state["aperture_mask"]
    annulus_mask = st.session_state["annulus_mask"]
    light_curve = compute_light_curve(flux_data, aperture_mask, annulus_mask)

    # Plot the light curve
    st.write("### Light Curve")
    fig, ax = plt.subplots()
    ax.plot(light_curve, marker="o", linestyle="-", color="blue")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Net Flux")
    ax.grid()
    st.pyplot(fig)

    # Allow download of light curve
    light_curve_df = pd.DataFrame({"Frame": np.arange(len(light_curve)), "Net Flux": light_curve})
    csv = light_curve_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Light Curve",
        data=csv,
        file_name="light_curve.csv",
        mime="text/csv",
    )

