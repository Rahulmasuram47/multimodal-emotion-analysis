"""Streamlit app entrypoint forwarding to streamlit_app.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "streamlit_app.py"), run_name="__main__")

