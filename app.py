import math
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go

st.set_page_config(page_title="Adaptive Traffic Lab", page_icon="🚦", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PCE = np.array([0.5, 1.0, 1.2, 2.5], dtype=np.float32)
CLASS_NAMES = ["Motorcycle", "Car", "CNG / auto", "Bus / truck"]
COLORS = ["#F4B942", "#3B82F6", "#A78BFA", "#EF8354"]


class TrafficIntersection:
    def __init__(self, steps, seed, demand):
        self.steps, self.rng, self.demand = steps, np.random.default_rng(seed), demand
        self.min_green, self.max_green, self.capacity = 3, 12, 7.0
        self.reset()

    def reset(self):
        self.t, self.phase, self.age, self.clearance = 0, 0, 0, 0
        self.queues = np.zeros((2, 4), dtype=np.float32)
        self.waiting = np.zeros(2, dtype=np.float32)
        self.base = np.array([[1.7, 1.1, .65, .12], [1.4, 1.0, .55, .10]]) * self.demand
        return self.state()

    def state(self):
        q = self.queues @ PCE
        return np.array([q[0]/80, q[1]/80, self.waiting[0]/500,
                         self.waiting[1]/500, float(self.phase),
                         self.age/self.max_green, float(self.clearance),
                         self.demand/1.5], dtype=np.float32)

    def step(self, action):
        switched = int(action) != self.phase
        if switched and self.age < self.min_green:
            action, switched = self.phase, False
        if self.age >= self.max_green:
            action, switched = 1-self.phase, True
        if switched:
            self.phase, self.age, self.clearance = int(action), 0, 1

        x = self.t / max(1, self.steps)
        arrivals = np.zeros((2, 4), dtype=np.int32)
        for m in range(2):
            wave = .75 + .35 * math.sin(2*math.pi*(x + .12*m))
            burst = 1.35 if m == 0 and .25 < x < .48 else (1.4 if m == 1 and .62 < x < .82 else 1)
            arrivals[m] = self.rng.poisson(self.base[m] * wave * burst)
            self.queues[m] += arrivals[m]

        q_before = self.queues @ PCE
        self.waiting += q_before
        departed = 0.0
        departed_by_class = np.zeros(4, dtype=np.float32)
        if self.clearance:
            self.clearance -= 1
        else:
            cap = self.capacity
            for cls in [3, 2, 1, 0]:
                n = min(self.queues[self.phase, cls], cap/PCE[cls])
                self.queues[self.phase, cls] -= n
                departed_by_class[cls] = n
                departed += n*PCE[cls]
                cap -= n*PCE[cls]
                if cap <= 1e-6:
                    break

        self.age += 1
        self.t += 1
        q = self.queues @ PCE
        info = dict(step=self.t, queue=float(q.sum()), ns=float(q[0]), ew=float(q[1]),
                    waiting=float(self.waiting.sum()), departed=float(departed),
                    switches=int(switched), phase=self.phase, clearance=self.clearance,
                    queues=self.queues.copy(), arrivals=arrivals.copy(),
                    departed_by_class=departed_by_class.copy())
        return self.state(), info, self.t >= self.steps


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8,128), nn.ReLU(),
                                 nn.Linear(128,128), nn.ReLU(), nn.Linear(128,2))
    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_model(path):
    model = QNetwork().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def action_for(controller, state, env, model):
    if controller == "DQN (AI)":
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(model(x).argmax(1).item())
    if controller == "Fixed-time":
        return (env.t // 8) % 2
    if env.age < env.min_green:
        return env.phase
    if env.age >= env.max_green:
        return 1-env.phase
    q = env.queues @ PCE
    other = 1-env.phase
    return other if q[other] > q[env.phase]*1.15 else env.phase


def simulate(controller, model, seed, steps, demand):
    env = TrafficIntersection(steps, seed, demand)
    state, rows, done = env.reset(), [], False
    while not done:
        action = action_for(controller, state, env, model)
        state, info, done = env.step(action)
        rows.append(info)
    df = pd.DataFrame(rows)
    return dict(trace=df, avg_queue=df.queue.mean(), avg_wait=df.waiting.mean(),
                discharged=df.departed.sum(), switches=df.switches.sum(),
                final_queue=df.queue.iloc[-1])


def make_animation(df, title, frame_duration_ms=650):
    """Replay queued vehicles and vehicles physically crossing the junction."""
    fig = go.Figure()
    road = "#303943"
    # Green verge and two intersecting carriageways.
    fig.add_shape(type="rect", x0=-10.5, x1=10.5, y0=-10.5, y1=10.5,
                  fillcolor="#263B35", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-2.55, x1=2.55, y0=-10.5, y1=10.5,
                  fillcolor=road, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-10.5, x1=10.5, y0=-2.55, y1=2.55,
                  fillcolor=road, line_width=0, layer="below")
    # Lane separators, dashed centre markings and stop lines.
    for x in [-1.25, 1.25]:
        fig.add_shape(type="line", x0=x, x1=x, y0=-10.5, y1=-2.7,
                      line=dict(color="#D6D9D5", width=1, dash="dash"), layer="below")
        fig.add_shape(type="line", x0=x, x1=x, y0=2.7, y1=10.5,
                      line=dict(color="#D6D9D5", width=1, dash="dash"), layer="below")
    for y in [-1.25, 1.25]:
        fig.add_shape(type="line", x0=-10.5, x1=-2.7, y0=y, y1=y,
                      line=dict(color="#D6D9D5", width=1, dash="dash"), layer="below")
        fig.add_shape(type="line", x0=2.7, x1=10.5, y0=y, y1=y,
                      line=dict(color="#D6D9D5", width=1, dash="dash"), layer="below")
    for y in [-2.75, 2.75]:
        fig.add_shape(type="line", x0=-2.55, x1=2.55, y0=y, y1=y,
                      line=dict(color="#F8FAFC", width=4), layer="below")
    for x in [-2.75, 2.75]:
        fig.add_shape(type="line", x0=x, x1=x, y0=-2.55, y1=2.55,
                      line=dict(color="#F8FAFC", width=4), layer="below")
    # Crosswalk bars.
    for base_y in [-3.15, 3.15]:
        for k in range(6):
            xx = -1.8 + k*.65
            fig.add_shape(type="rect", x0=xx, x1=xx+.32, y0=base_y-.22,
                          y1=base_y+.22, fillcolor="#E8ECEF", line_width=0, layer="below")
    for base_x in [-3.15, 3.15]:
        for k in range(6):
            yy = -1.8 + k*.65
            fig.add_shape(type="rect", x0=base_x-.22, x1=base_x+.22,
                          y0=yy, y1=yy+.32, fillcolor="#E8ECEF", line_width=0, layer="below")

    # Each vehicle is a recognizable top-down emoji sprite, not a coloured dot.
    glyphs = ["🏍️", "🚗", "🛺", "🚌"]
    names = CLASS_NAMES
    fig.add_trace(go.Scatter(x=[], y=[], mode="text", text=[],
        textfont=dict(size=19), hovertemplate="%{customdata}<extra></extra>",
        customdata=[], name="Waiting vehicles"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="text", text=[],
        textfont=dict(size=21), hovertemplate="%{customdata}<extra></extra>",
        customdata=[], name="Crossing vehicles"))
    fig.add_trace(go.Scatter(x=[-1.9, 1.9, -1.9, 1.9], y=[2.0, 2.0, -2.0, -2.0],
        mode="markers", marker=dict(size=18, color=["#ef4444"]*4,
        line=dict(color="white", width=1.5)), hoverinfo="skip", name="Signals"))

    frames = []
    moving = []  # [movement, class, progress, lane, direction]
    def queue_sprites(q):
        xs, ys, glyph_text, hover = [], [], [], []
        ns_slots = [0, 0]
        ew_slots = [0, 0]
        # Keep each approach as one orderly queue; cap the drawing at 9 sprites per lane.
        for cls in range(4):
            count = min(24, int(round(float(q[0, cls]))))
            for j in range(count):
                side = (ns_slots[0] + ns_slots[1]) % 2
                if ns_slots[side] >= 9:
                    continue
                slot = ns_slots[side]
                xs.append(-.72 if side == 0 else .72)
                ys.append((3.8 + slot*.78) * (1 if side == 0 else -1))
                glyph_text.append(glyphs[cls]); hover.append(names[cls] + " · North–South queue")
                ns_slots[side] += 1
            count = min(24, int(round(float(q[1, cls]))))
            for j in range(count):
                side = (ew_slots[0] + ew_slots[1]) % 2
                if ew_slots[side] >= 9:
                    continue
                slot = ew_slots[side]
                xs.append((-3.8 - slot*.78) * (1 if side == 0 else -1))
                ys.append(-.72 if side == 0 else .72)
                glyph_text.append(glyphs[cls]); hover.append(names[cls] + " · East–West queue")
                ew_slots[side] += 1
        return xs, ys, glyph_text, hover


    # Two visual subframes per simulation step make the vehicles visibly travel.
    for _, r in df.iterrows():
        q = r["queues"]
        # Advance vehicles already in the junction.
        for v in moving:
            v[2] += .48
        moving = [v for v in moving if v[2] <= 1.12]
        # Departures create moving sprites on the currently green movement.
        for cls, amount in enumerate(r.get("departed_by_class", np.zeros(4))):
            for j in range(min(3, int(round(float(amount))))):
                moving.append([int(r.phase), cls, .05 + j*.08, j % 2, 1 if j % 2 == 0 else -1])

        for sub in range(2):
            qx, qy, qt, qh = queue_sprites(q)
            mx, my, mt, mh = [], [], [], []
            for movement, cls, progress, lane, direction in moving:
                p = min(1.0, progress + sub*.20)
                if movement == 0:  # North–South movement: travel vertically through junction.
                    mx.append(-.72 if lane == 0 else .72)
                    my.append(direction * (3.0 + p*3.8))
                else:  # East–West movement: travel horizontally through junction.
                    mx.append(direction * (3.0 + p*3.8))
                    my.append(-.72 if lane == 0 else .72)
                mt.append(glyphs[cls]); mh.append(names[cls] + " · passing through junction")
            ns = r.phase == 0 and r.clearance == 0
            ew = r.phase == 1 and r.clearance == 0
            lamps = ["#22c55e" if ns else "#ef4444", "#22c55e" if ew else "#ef4444"]*2
            if r.clearance: lamps = ["#f59e0b"]*4
            step_label = f"Step {int(r.step)} · {'North–South green' if ns else 'East–West green' if ew else 'Changing signal'}"
            frames.append(go.Frame(name=f"{int(r.step)}-{sub}", data=[
                go.Scatter(x=qx, y=qy, text=qt, customdata=qh),
                go.Scatter(x=mx, y=my, text=mt, customdata=mh),
                go.Scatter(x=[-1.9, 1.9, -1.9, 1.9], y=[2.0, 2.0, -2.0, -2.0],
                           marker=dict(size=18, color=lamps, line=dict(color="white", width=1.5)))
            ], traces=[0, 1, 2], layout=go.Layout(title_text=step_label)))

    for x, y, label in [(0, 9.7, "NORTH"), (0, -9.7, "SOUTH"),
                        (-9.5, 0, "WEST"), (9.5, 0, "EAST")]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                           font=dict(size=11, color="#F8FAFC"),
                           bgcolor="rgba(15,23,42,0.82)", borderpad=4)
    fig.add_annotation(x=0, y=0, text="JUNCTION", showarrow=False,
                       font=dict(size=9, color="#FFFFFF"),
                       bgcolor="rgba(15,23,42,0.75)", borderpad=3)
    fig.frames = frames
    if frames:
        for i in range(3):
            for key in ["x", "y"]:
                setattr(fig.data[i], key, getattr(frames[0].data[i], key))
            if i < 2:
                fig.data[i].text = frames[0].data[i].text
                fig.data[i].customdata = frames[0].data[i].customdata
        fig.data[2].marker.color = frames[0].data[2].marker.color
    fig.update_layout(title=title, height=650, margin=dict(l=5, r=5, t=55, b=5),
        xaxis=dict(range=[-10.5, 10.5], visible=False, scaleanchor="y"),
        yaxis=dict(range=[-10.5, 10.5], visible=False),
        plot_bgcolor="#263B35", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        updatemenus=[dict(type="buttons", showactive=False, x=.02, y=1.12, buttons=[
            dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": frame_duration_ms, "redraw": True},
                 "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="Ⅱ Pause", method="animate", args=[[None], {"mode": "immediate",
                 "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}])
        ])],
        sliders=[dict(x=.15, y=1.12, len=.82, currentvalue=dict(prefix="Replay frame: "),
            steps=[dict(method="animate", label=f.name, args=[[f.name], {"mode": "immediate",
            "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}])
            for f in frames[::max(1, len(frames)//24)]])])
    return fig


st.title("🚦 Adaptive Traffic Lab")
st.markdown("**Interactive Deep Q-Learning traffic-signal simulator**")
st.caption("Train in Colab · Experiment locally · Replay the controller's decisions")

with st.sidebar:
    st.header("🎛️ Control panel")
    controller=st.selectbox("Controller",["DQN (AI)","Fixed-time","Actuated","Compare all"])
    demand=st.slider("Traffic intensity",.60,1.50,1.00,.05)
    duration=st.slider("Simulation duration",60,360,240,30)
    playback_speed=st.slider("Vehicle animation speed (slower ← → faster)",350,1200,650,50,
                              help="Controls how long each animation frame stays on screen. Higher values are slower.")
    seed=st.number_input("Traffic scenario seed",0,999999,2026)
    st.caption("Change the settings and rerun to explore different traffic conditions.")

try:
    model=load_model("traffic_dqn.pth")
except Exception as e:
    model=None
    st.error("Couldn't load traffic_dqn.pth. Put it in the same folder as app.py and refresh.")
    with st.expander("Technical details"): st.code(str(e))

if model is not None:
    if st.button("🚦 Start / rerun simulation",type="primary",use_container_width=True):
        chosen=["DQN (AI)","Fixed-time","Actuated"] if controller=="Compare all" else [controller]
        results={}
        progress=st.progress(0)
        for i,name in enumerate(chosen):
            results[name]=simulate(name,model,int(seed),int(duration),float(demand))
            progress.progress((i+1)/len(chosen),text=f"Simulated {name}")
        progress.empty()
        st.session_state["results"]=results
    if "results" not in st.session_state:
        st.info("Choose your settings, then click **Start / rerun simulation**.")
        a,b,c=st.columns(3)
        a.markdown("**🧠 AI controller**\\n\\nWatch the DQN choose a signal phase.")
        b.markdown("**🚗 Traffic intensity**\\n\\nIncrease demand to stress-test the junction.")
        c.markdown("**📊 Policy comparison**\\n\\nCompare all three controllers on the same scenario.")
    else:
        results=st.session_state["results"]
        names=list(results)
        active=st.selectbox("Replay this controller",names)
        r=results[active]; df=r["trace"]
        st.subheader(f"How the {active} controller manages the junction")
        st.info("WHAT YOU ARE SEEING: The controller gives green time to one movement at a time. "
                "North–South (vertical lanes) is one movement; East–West (horizontal lanes) is the other. "
                "Vehicles join queues, wait at a red signal, and leave when their movement receives green. "
                "Yellow markers indicate the short signal-change clearance step. Vehicle icons visibly travel through the junction when traffic is discharged.")
        st.markdown("**Vehicle key:** 🏍️ Motorcycle · 🚗 Car · 🛺 CNG / auto-rickshaw · 🚌 Bus / truck")
        st.caption(f"Playback: {playback_speed} ms per frame. Increase this value to slow the animation. "
                   "This is a simplified two-movement intersection model; opposing lanes are mirrored.")
        st.plotly_chart(make_animation(df,active,playback_speed),use_container_width=True,
                        key=f"anim_{active}_{seed}_{demand}_{duration}_{playback_speed}")
        k1,k2,k3,k4=st.columns(4)
        k1.metric("Average queue",f"{r['avg_queue']:.1f} PCE")
        k2.metric("Waiting proxy",f"{r['avg_wait']:.0f}")
        k3.metric("Discharged traffic",f"{r['discharged']:.1f} PCE")
        k4.metric("Final queue",f"{r['final_queue']:.1f} PCE")
        st.caption("**How to read the metrics:** Average queue = average number of queued passenger-car-equivalent units (PCE). "
                   "Waiting proxy = cumulative queue-based indicator (not seconds). "
                   "Discharged traffic = total PCE that passed through the junction. Lower queue/waiting and higher discharge are generally better.")
        st.subheader("Queue dynamics")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df.step,y=df.ns,name="North–South",mode="lines"))
        fig.add_trace(go.Scatter(x=df.step,y=df.ew,name="East–West",mode="lines"))
        fig.update_layout(height=330,xaxis_title="Simulation step",yaxis_title="Queue (PCE)",
                          margin=dict(l=5,r=5,t=20,b=5))
        st.plotly_chart(fig,use_container_width=True)
        comp=pd.DataFrame([{"Controller":n,"Average queue ↓":x["avg_queue"],
            "Waiting proxy ↓":x["avg_wait"],"Discharged PCE ↑":x["discharged"],
            "Signal switches":x["switches"],"Final queue ↓":x["final_queue"]}
            for n,x in results.items()])
        st.subheader("Controller comparison")
        st.dataframe(comp.set_index("Controller").style.format("{:.2f}"),use_container_width=True)
        st.download_button("⬇ Download results as CSV",comp.to_csv(index=False).encode(),
            "traffic_experiment.csv","text/csv")

st.divider()
with st.expander("How to interpret this prototype"):
    st.markdown("""
    - **DQN:** the trained neural network chooses between North–South and East–West phases.
    - **PCE:** Passenger Car Equivalent, a weighted unit for different vehicle classes.
    - **Waiting proxy:** cumulative queue-based indicator, not measured seconds.
    - **Mirrored queues:** the model groups opposing directions into two movements; the animated view mirrors each grouped queue on both sides. The labels show the four visible approaches.
    - **Animation speed:** the sidebar setting controls playback only; it does not change the simulated traffic or DQN decisions.
    - **Scope:** this is a simplified research simulation, not a live traffic-control system.
    """)
