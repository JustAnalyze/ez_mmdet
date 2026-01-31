import warnings

# Suppress noisy library warnings during tests
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated")
