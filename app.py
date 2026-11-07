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
        for m in range(2):
            wave = .75 + .35 * math.sin(2*math.pi*(x + .12*m))
            burst = 1.35 if m == 0 and .25 < x < .48 else (1.4 if m == 1 and .62 < x < .82 else 1)
            self.queues[m] += self.rng.poisson(self.base[m] * wave * burst)

        q_before = self.queues @ PCE
        self.waiting += q_before
        departed = 0.0
        if self.clearance:
            self.clearance -= 1
        else:
            cap = self.capacity
            for cls in [3, 2, 1, 0]:
                n = min(self.queues[self.phase, cls], cap/PCE[cls])
                self.queues[self.phase, cls] -= n
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
                    queues=self.queues.copy())
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


def make_animation(df, title):
    fig = go.Figure()
    # Schematic road layout.
    for x0,x1,y0,y1,color in [(-2.4,2.4,-10,10,"#343A40"),
                              (-10,10,-2.4,2.4,"#343A40"),
                              (-2.4,2.4,-2.4,2.4,"#59636E")]:
        fig.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,
                      fillcolor=color,line_width=0,layer="below")
    for x0,x1,y0,y1 in [(0,0,-10,-2.5),(0,0,2.5,10),(-10,-2.5,0,0),(2.5,10,0,0)]:
        fig.add_shape(type="line",x0=x0,x1=x1,y0=y0,y1=y1,
                      line=dict(color="white",width=1,dash="dash"))
    fig.add_trace(go.Scatter(x=[],y=[],mode="markers",
        marker=dict(size=11,color=[]),text=[],hovertemplate="%{text}<extra></extra>",
        name="Queued vehicles"))
    fig.add_trace(go.Scatter(x=[-1.6,1.6,-1.6,1.6],y=[2,2,-2,-2],
        mode="markers",marker=dict(size=19,color=["red"]*4,line=dict(color="white",width=1)),
        hoverinfo="skip",name="Traffic signals"))

    frames=[]
    for _,r in df.iterrows():
        q=r["queues"]; xs=[]; ys=[]; cs=[]; labels=[]
        # Opposing directions are mirrored because the model groups each pair.
        for cls in range(4):
            for j in range(min(12,int(q[0,cls]))):
                d=3+j*.48
                xs += [-.9,.9]; ys += [d,-d]; cs += [COLORS[cls]]*2
                labels += [f"North–South · {CLASS_NAMES[cls]}"]*2
            for j in range(min(12,int(q[1,cls]))):
                d=3+j*.48
                xs += [d,-d]; ys += [.9,-.9]; cs += [COLORS[cls]]*2
                labels += [f"East–West · {CLASS_NAMES[cls]}"]*2
        ns = r.phase == 0 and r.clearance == 0
        ew = r.phase == 1 and r.clearance == 0
        lamps = ["#22c55e" if ns else "#ef4444",
                 "#22c55e" if ew else "#ef4444"]*2
        if r.clearance: lamps=["#f59e0b"]*4
        frames.append(go.Frame(name=str(int(r.step)),data=[
            go.Scatter(x=xs,y=ys,mode="markers",marker=dict(size=11,color=cs),
                       text=labels,hovertemplate="%{text}<extra></extra>"),
            go.Scatter(x=[-1.6,1.6,-1.6,1.6],y=[2,2,-2,-2],mode="markers",
                       marker=dict(size=19,color=lamps,line=dict(color="white",width=1)))
        ],traces=[0,1]))
    fig.frames=frames
    if frames:
        fig.data[0].x,fig.data[0].y=frames[0].data[0].x,frames[0].data[0].y
        fig.data[0].marker.color=frames[0].data[0].marker.color
        fig.data[0].text=frames[0].data[0].text
        fig.data[1].marker.color=frames[0].data[1].marker.color
    fig.update_layout(title=title,height=570,margin=dict(l=5,r=5,t=50,b=5),
        xaxis=dict(range=[-10.5,10.5],visible=False,scaleanchor="y"),
        yaxis=dict(range=[-10.5,10.5],visible=False),
        plot_bgcolor="#20262D",paper_bgcolor="rgba(0,0,0,0)",font=dict(color="white"),
        updatemenus=[dict(type="buttons",showactive=False,x=.02,y=1.1,buttons=[
            dict(label="▶ Play",method="animate",args=[None,{"frame":{"duration":90,"redraw":True},
                 "fromcurrent":True,"transition":{"duration":0}}]),
            dict(label="Ⅱ Pause",method="animate",args=[[None],{"mode":"immediate",
                 "frame":{"duration":0,"redraw":False},"transition":{"duration":0}}])
        ])],
        sliders=[dict(x=.15,y=1.1,len=.82,currentvalue=dict(prefix="Step: "),
            steps=[dict(method="animate",label=f.name,args=[[f.name],{"mode":"immediate",
            "frame":{"duration":0,"redraw":True},"transition":{"duration":0}}])
            for f in frames[::max(1,len(frames)//24)]])])
    return fig


st.title("🚦 Adaptive Traffic Lab")
st.markdown("**Interactive Deep Q-Learning traffic-signal simulator**")
st.caption("Train in Colab · Experiment locally · Replay the controller's decisions")

with st.sidebar:
    st.header("🎛️ Control panel")
    controller=st.selectbox("Controller",["DQN (AI)","Fixed-time","Actuated","Compare all"])
    demand=st.slider("Traffic intensity",.60,1.50,1.00,.05)
    duration=st.slider("Simulation duration",60,360,240,30)
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
        st.subheader(f"Animated intersection · {active}")
        st.caption("Use Play/Pause or drag the timeline slider. Vehicle positions are schematic.")
        st.plotly_chart(make_animation(df,active),use_container_width=True,key=f"anim_{active}_{seed}_{demand}_{duration}")
        k1,k2,k3,k4=st.columns(4)
        k1.metric("Average queue",f"{r['avg_queue']:.1f} PCE")
        k2.metric("Waiting proxy",f"{r['avg_wait']:.0f}")
        k3.metric("Discharged traffic",f"{r['discharged']:.1f} PCE")
        k4.metric("Final queue",f"{r['final_queue']:.1f} PCE")
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
    - **Mirrored queues:** the model groups opposing directions into two movements; the animated view mirrors each grouped queue on both sides.
    - **Scope:** this is a simplified research simulation, not a live traffic-control system.
    """)
