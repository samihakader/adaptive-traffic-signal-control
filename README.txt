ADAPTIVE TRAFFIC LAB — INTERACTIVE VERSION

1. Extract this ZIP.
2. Copy your Colab-generated traffic_dqn.pth into this folder, beside app.py.
3. Open Command Prompt in this folder and run:

   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py

4. If the browser does not open automatically, visit http://localhost:8501.

Features:
- Animated intersection replay with Play/Pause and timeline slider.
- Controller selection: DQN, fixed-time, actuated, or compare all.
- Traffic-intensity, simulation-duration, and random-seed controls.
- Separate North–South and East–West queue chart.
- Performance metrics and CSV export.

Notes:
- Use the exact traffic_dqn.pth file downloaded from Colab.
- The animation is schematic. The underlying model groups opposing directions and mirrors the queue for display.
- Waiting proxy is not measured seconds.
- This is a simulation-based research prototype, not a live road controller.
