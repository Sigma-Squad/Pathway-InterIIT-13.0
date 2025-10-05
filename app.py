import streamlit as st
from src.main import compiled_graph

st.set_page_config(page_title="Dynamic RAG Agent", page_icon="🤖", layout="centered")

# Header
st.title("🤖 Dynamic RAG Agent for Legal Policy QA")

# Links below title
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        "📄 [Read Paper](https://drive.google.com/file/d/18Sv8mbk-sqd_uzTOMhvoigH0jMWbzEtS/view?usp=drive_link)",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        "📧 [Contact Us](mailto:sigma.squad@iittp.ac.in)", unsafe_allow_html=True
    )

st.divider()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_step" not in st.session_state:
    st.session_state.current_step = None


def run_agent(user_query: str, step_placeholder):
    """
    Run the LangGraph agent and update UI in real-time.

    Args:
        user_query: The user's input question
        step_placeholder: Streamlit placeholder for showing current step
    """
    try:
        # Initialize the state
        initial_state = {
            "input_prompt": user_query,
            "subtasks": [],
            "rag_response": "",
            "webrag_response": "",
            "eg_response": "",
            "final_response": "",
        }

        # Stream through the graph to show progress
        final_state = None

        # Map technical node names to user-friendly descriptions
        step_names = {
            "CoT": "Chain of Thought - Breaking down the query into subtasks",
            "DBR": "Database Retrieval - Searching local RAG database",
            "WebRAG": "Web Retrieval - Searching online sources",
            "EG": "Evidence Graph - Building knowledge graph from sources",
            "RG": "Response Generation - Drafting final answer",
        }

        for event in compiled_graph.stream(initial_state):
            # Extract node name from event
            node_name = list(event.keys())[0]

            # Update current step and display immediately
            if node_name in step_names:
                step_placeholder.info(f"💭 {step_names[node_name]}")

            # Store the state after this node
            final_state = event[node_name]

        # Clear thinking indicator
        step_placeholder.empty()

        # Return final response
        return final_state.get("final_response", "No response generated")

    except Exception as e:
        # Clear thinking indicator on error
        step_placeholder.empty()
        return f"❌ Error: {str(e)}"


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        # Create placeholder for thinking steps
        step_placeholder = st.empty()

        # Create placeholder for response
        message_placeholder = st.empty()

        # Run agent with real-time step updates
        response = run_agent(prompt, step_placeholder)

        # Display final response
        message_placeholder.write(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Rerun to update UI
    st.rerun()

# Footer section
st.divider()

# Built with section
st.markdown("### 🛠️ Built With")
tech_cols = st.columns(6)
technologies = [
    ("LangGraph", "https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge"),
    ("LangChain", "https://img.shields.io/badge/LangChain-121212?style=for-the-badge"),
    (
        "LlamaIndex",
        "https://img.shields.io/badge/LlamaIndex-8B5CF6?style=for-the-badge",
    ),
    ("ChromaDB", "https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge"),
    (
        "OpenRouter",
        "https://img.shields.io/badge/OpenRouter-00A67E?style=for-the-badge",
    ),
    (
        "HuggingFace",
        "https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge",
    ),
]

for col, (name, badge) in zip(tech_cols, technologies):
    with col:
        st.markdown(f"![{name}]({badge})")

st.divider()

# Organization section
col_logo, col_org = st.columns([1, 4])

with col_logo:
    # Display organization logo
    try:
        st.image("./assets/SigmaSquadLogo.jpg", width=100)
    except:
        st.markdown("### 🏢")  # Fallback if logo not found

with col_org:
    st.markdown("### Built by Sigma Squad - The AI/ML club of IIT Tirupati")

# Team members in columns (2 per column)
st.markdown("**Team Members:**")
team_col1, team_col2, team_col3, team_col4 = st.columns(4)

with team_col1:
    st.markdown("""
    - Niranjan M
    - Chandradithya J
    """)

with team_col2:
    st.markdown("""
    - Adithya Ananth
    - Aniket Johri
    """)

with team_col3:
    st.markdown("""
    - Karthikeya M
    - Sayan Kundu
    """)

with team_col4:
    st.markdown("""
    - Umakant Sahu
    - Deepak Yadav
    """)

st.markdown("### Proposed Architecture")
st.image("./assets/architecture.png", use_container_width=True)
st.caption(
    "Multi-stage retrieval architecture combining Chain-of-Thought reasoning, database retrieval, web search, and evidence graph generation for comprehensive legal policy question answering. The utility module has not yet been implemented."
)

st.markdown("---")
