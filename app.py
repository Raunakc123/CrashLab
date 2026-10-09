"""Streamlit entrypoint for the guided student and advanced views."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("student_app.py")), run_name="__main__")
