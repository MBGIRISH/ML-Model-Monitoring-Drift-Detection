#!/usr/bin/env python3
"""
Streamlit Dashboard Launcher
Run this script to start the Streamlit dashboard

Author: M B GIRISH
Date: January 2026
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "src", "streamlit_dashboard.py")
    
    # Verify dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        sys.exit(0)

