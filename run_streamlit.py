"""
Runner script for Streamlit frontend
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to streamlit app
        streamlit_app = os.path.join(script_dir, "streamlit_app.py")
        
        # Run streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            streamlit_app,
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🚀 Starting Streamlit Frontend...")
        print(f"📍 App will be available at: http://localhost:8501")
        print(f"📁 App directory: {script_dir}")
        print("⚠️  Make sure the API server is running on http://localhost:8000")
        print("-" * 50)
        
        # Run the command
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
