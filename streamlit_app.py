import streamlit as st
import requests
import time
import os

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OMEGA Research Grid",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL defaults to localhost, but can be set via environment variable
# for deployment (e.g. to a cloud-hosted FastAPI instance)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #58a6ff, #bc8cff);
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("OMEGA GRID")
    st.markdown("### Autonomous AI Research OS")
    
    st.markdown("---")
    st.markdown("**Configuration**")
    depth = st.slider("Research Depth", min_value=1, max_value=5, value=3, help="Higher depth means more sources and iterations, but costs more.")
    budget = st.slider("Cost Budget ($)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    
    st.markdown("---")
    st.caption("Agent Council Active:\n- 🧠 Planner\n- 🔍 Search\n- 📖 Reader\n- 💾 Memory\n- ✍️ Writer\n- 🔬 Critic\n- ✅ Verifier")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Networked Intelligence Core")
st.markdown("Enter a complex research topic below. OMEGA will orchestrate an autonomous agent council to retrieve, synthesize, and verify the information.")

query = st.text_area("Research Objective", placeholder="What are the latest breakthroughs in large language model reasoning capabilities?", height=100)

if st.button("🚀 Launch Autonomous Research", type="primary", use_container_width=True):
    if len(query) < 10:
        st.error("Please enter a more detailed query (at least 10 characters).")
        st.stop()

    try:
        # 1. Start Session
        with st.spinner("Initializing Agent Council..."):
            res = requests.post(
                f"{API_URL}/api/v1/research/start",
                json={
                    "query": query,
                    "depth": depth,
                    "cost_budget_usd": budget,
                    "max_sources": depth * 5,
                    "token_budget": 80000
                }
            )
            res.raise_for_status()
            session_id = res.json()["session_id"]
        
        st.success(f"Session established: `{session_id}`")
        
        # 2. Monitor Execution
        status_container = st.container()
        metrics_container = st.empty()
        log_container = st.empty()
        
        with status_container:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
        logs = []
        is_complete = False
        
        while not is_complete:
            time.sleep(2)  # Poll every 2 seconds
            
            try:
                status_res = requests.get(f"{API_URL}/api/v1/research/{session_id}/status")
                if status_res.status_code == 200:
                    data = status_res.json()
                    current_state = data.get("state", "UNKNOWN")
                    iteration = data.get("iteration", 0)
                    cost = data.get("cost_usd", 0.0)
                    chunks = data.get("chunks_collected", 0)
                    
                    # Update Progress mapping
                    state_map = {
                        "IDLE": 0, "PLANNING": 10, "SEARCHING": 30, 
                        "READING": 50, "WRITING": 70, "CRITIQUING": 80, 
                        "VERIFYING": 90, "FINALIZING": 95, "COMPLETED": 100, "FAILED": 100
                    }
                    progress_bar.progress(state_map.get(current_state, 50))
                    status_text.markdown(f"**Agent State:** `{current_state}` | **Iteration:** `{iteration}`")
                    
                    # Update Metrics
                    metrics_html = f"""
                    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                        <div class="metric-card" style="flex: 1;"><div class="metric-value">${cost:.4f}</div><div class="metric-label">Cost Incurred</div></div>
                        <div class="metric-card" style="flex: 1;"><div class="metric-value">{chunks}</div><div class="metric-label">Sources Extracted</div></div>
                        <div class="metric-card" style="flex: 1;"><div class="metric-value">{iteration}/3</div><div class="metric-label">Refinement Loops</div></div>
                    </div>
                    """
                    metrics_container.markdown(metrics_html, unsafe_allow_html=True)
                    
                    # Update Logs
                    new_log = f"[{time.strftime('%H:%M:%S')}] Transitioned to {current_state}"
                    if not logs or logs[-1] != new_log:
                        logs.append(new_log)
                        log_text = "\n".join(logs[-5:]) # Show last 5
                        log_container.code(log_text, language="bash")
                    
                    if current_state in ["COMPLETED", "FAILED"]:
                        is_complete = True
                        if current_state == "FAILED":
                            st.error("The research session failed. Check the backend logs for details.")
                            st.stop()
            except Exception as e:
                pass # Ignore polling errors and retry
                
        # 3. Fetch Final Report
        st.success("Research Complete!")
        st.markdown("---")
        
        with st.spinner("Synthesizing final document..."):
            report_res = requests.get(f"{API_URL}/api/v1/research/{session_id}/report")
            if report_res.status_code == 200:
                report_data = report_res.json()
                
                # Display Results
                st.header(report_data.get("title", "Research Report"))
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    st.markdown("### Confidence")
                    conf = report_data.get("confidence_score", 0) * 100
                    st.metric(label="Overall Confidence", value=f"{conf:.1f}%")
                    
                    scores = report_data.get("critique_scores", {})
                    if scores:
                        st.markdown("### Quality Metrics")
                        for k, v in scores.items():
                            st.progress(float(v), text=f"{k.replace('_score', '').title()}: {float(v)*100:.0f}%")
                            
                with col1:
                    tabs = st.tabs(["Executive Summary", "Key Findings", "Full Report"])
                    
                    with tabs[0]:
                        st.markdown(report_data.get("executive_summary", ""))
                        st.markdown("### Conclusion")
                        st.markdown(report_data.get("conclusion", ""))
                        
                    with tabs[1]:
                        findings = report_data.get("key_findings", [])
                        for i, f in enumerate(findings, 1):
                            st.info(f"**{i}.** {f}")
                            
                    with tabs[2]:
                        st.markdown(report_data.get("markdown_content", ""))
                        
            else:
                st.error("Failed to retrieve the final report.")
                
    except Exception as e:
        st.error(f"Failed to connect to the backend. Is Docker running? Error: {str(e)}")
