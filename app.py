import json
import re
from pathlib import Path

import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from crewai import Agent, Crew, Task, Process, LLM
from crewai.tools import BaseTool


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Raabta AI - Customer Support",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 2. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

FAISS_PATH = BASE_DIR / "faiss.index"
CHUNKS_PATH = BASE_DIR / "chunks.json"
ORDERS_PATH = BASE_DIR / "orders.xlsx"


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Current stable Gemini Flash model
GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# 4. STREAMLIT SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_escalations" not in st.session_state:
    st.session_state.pending_escalations = []

if "crew" not in st.session_state:
    st.session_state.crew = None


# ============================================================
# 5. GET GEMINI API KEY FROM STREAMLIT SECRETS
# ============================================================

try:

    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

except KeyError:

    st.error(
        "GEMINI_API_KEY is missing. "
        "Add it to Streamlit Cloud → App Settings → Secrets."
    )

    st.stop()


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# ============================================================
# 7. LOAD FAISS DATABASE + CHUNKS
# ============================================================

@st.cache_resource
def load_knowledge_base():

    if not FAISS_PATH.exists():

        raise FileNotFoundError(
            "faiss.index was not found."
        )

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            "chunks.json was not found."
        )

    index = faiss.read_index(
        str(FAISS_PATH)
    )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks


try:

    faiss_index, chunks = load_knowledge_base()

except Exception as error:

    st.error(
        f"Knowledge Base Error: {error}"
    )

    st.stop()


# ============================================================
# 8. LOAD EXCEL ORDER DATABASE
# ============================================================

@st.cache_data
def load_orders():

    if not ORDERS_PATH.exists():

        raise FileNotFoundError(
            "orders.xlsx was not found."
        )

    df = pd.read_excel(
        ORDERS_PATH,
        dtype=str
    )

    required_columns = [
        "Order ID",
        "Customer Name",
        "Contact Number",
        "Product",
        "Order Date",
        "Shipping Address",
        "Status"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing Excel columns: {missing}"
        )

    return df


try:

    orders_df = load_orders()

except Exception as error:

    st.error(
        f"Order Database Error: {error}"
    )

    st.stop()


# ============================================================
# 9. TOOL #1 — COMPANY KNOWLEDGE BASE
# ============================================================

class CompanyKnowledgeBaseTool(BaseTool):

    name: str = "company_knowledge_base"

    description: str = """
    Search Raabta AI's official company knowledge base.

    Use this tool for questions about:

    - shipping
    - returns
    - refunds
    - cancellations
    - warranty
    - damaged products
    - discounts
    - company information
    - customer service policies
    - security rules
    - escalation policies

    This tool performs semantic similarity search using FAISS.

    Do NOT use this tool to look up customer orders.
    """

    def _run(self, query: str) -> str:

        if not query or not query.strip():

            return "No search query was provided."

        # ----------------------------------------------------
        # Convert query into 384-dimensional embedding
        # ----------------------------------------------------

        query_embedding = embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        # ----------------------------------------------------
        # Search top 5 chunks
        # ----------------------------------------------------

        k = min(5, len(chunks))

        scores, indices = faiss_index.search(
            query_embedding,
            k
        )

        results = []

        # ----------------------------------------------------
        # Retrieve metadata
        # ----------------------------------------------------

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            if idx >= len(chunks):
                continue

            chunk = chunks[idx]

            # Support both requested schema and the
            # older schema from your previous file.

            chunk_id = chunk.get(
                "Chunk ID",
                chunk.get("id", "")
            )

            source_filename = chunk.get(
                "Source filename",
                ""
            )

            source_path = chunk.get(
                "Source path",
                ""
            )

            page_number = chunk.get(
                "Page number",
                ""
            )

            chunk_number = chunk.get(
                "Chunk number",
                ""
            )

            document_type = chunk.get(
                "Document type",
                ""
            )

            text_length = chunk.get(
                "Text length",
                ""
            )

            total_chunks = chunk.get(
                "Total number of chunks",
                ""
            )

            content = chunk.get(
                "content",
                chunk.get("text", "")
            )

            results.append(
                {
                    "similarity_score": round(
                        float(score),
                        4
                    ),

                    "Chunk ID": chunk_id,

                    "Source filename":
                        source_filename,

                    "Source path":
                        source_path,

                    "Page number":
                        page_number,

                    "Chunk number":
                        chunk_number,

                    "Document type":
                        document_type,

                    "Text length":
                        text_length,

                    "Total number of chunks":
                        total_chunks,

                    "content":
                        content
                }
            )

        return json.dumps(
            results,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 10. TOOL #2 — ORDER LOOKUP
# ============================================================

class OrderLookupTool(BaseTool):

    name: str = "order_lookup"

    description: str = """
    Search the Raabta AI Excel order database.

    Use this tool when the customer asks about a specific order.

    The customer normally provides an Order ID such as:

    ORD-2026-1001

    You can retrieve:

    - Order ID
    - Customer Name
    - Contact Number
    - Product
    - Order Date
    - Shipping Address
    - Status

    NEVER invent order information.
    """

    def _run(self, order_id: str) -> str:

        if not order_id:

            return json.dumps(
                {
                    "found": False,
                    "message": "Order ID is required."
                },
                indent=2
            )

        order_id = str(order_id).strip().upper()

        # ----------------------------------------------------
        # Find order
        # ----------------------------------------------------

        matches = orders_df[
            orders_df["Order ID"]
            .astype(str)
            .str.strip()
            .str.upper()
            == order_id
        ]

        if matches.empty:

            return json.dumps(
                {
                    "found": False,
                    "message":
                        f"No order found for {order_id}."
                },
                indent=2
            )

        row = matches.iloc[0]

        result = {

            "found": True,

            "Order ID":
                str(row["Order ID"]),

            "Customer Name":
                str(row["Customer Name"]),

            "Contact Number":
                str(row["Contact Number"]),

            "Product":
                str(row["Product"]),

            "Order Date":
                str(row["Order Date"]),

            "Shipping Address":
                str(row["Shipping Address"]),

            "Status":
                str(row["Status"])
        }

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 11. CREATE TOOL INSTANCES
# ============================================================

knowledge_base_tool = CompanyKnowledgeBaseTool()

order_lookup_tool = OrderLookupTool()


# ============================================================
# 12. EXPLICIT HUMAN REQUEST DETECTION
# ============================================================

def user_requested_human(message: str) -> bool:

    message = message.lower()

    human_phrases = [

        "human agent",

        "human support",

        "real person",

        "talk to a person",

        "speak to a person",

        "talk to an agent",

        "speak to an agent",

        "customer representative",

        "customer service representative",

        "representative",

        "speak with someone",

        "talk with someone",

        "manager",

        "human"
    ]

    return any(
        phrase in message
        for phrase in human_phrases
    )


# ============================================================
# 13. CONVERSATION HISTORY
# ============================================================

def get_conversation_history():

    if not st.session_state.messages:

        return "No previous conversation."

    history = []

    for message in st.session_state.messages:

        role = message["role"].upper()

        content = message["content"]

        history.append(
            f"{role}: {content}"
        )

    return "\n".join(history)


# ============================================================
# 14. ESCALATION RECORD
# ============================================================

def add_escalation(
    user_message: str,
    reason: str
):

    escalation = {

        "id":
            len(
                st.session_state.pending_escalations
            ) + 1,

        "customer_issue":
            user_message,

        "summary":
            reason,

        "status":
            "Pending"
    }

    st.session_state.pending_escalations.append(
        escalation
    )


# ============================================================
# 15. CREATE GEMINI LLM
# ============================================================

def create_gemini_llm():

    return LLM(

        model=f"gemini/{GEMINI_MODEL}",

        api_key=GEMINI_API_KEY,

        temperature=0.2
    )


# ============================================================
# 16. CREATE SINGLE CUSTOMER SUPPORT AGENT
# ============================================================

def create_customer_support_agent():

    llm = create_gemini_llm()

    agent = Agent(

        role="Customer Support Specialist",

        goal="""
        Resolve customer support questions accurately and
        professionally using the company's official knowledge
        base and order database.
        """,

        backstory="""
        You are Raabta AI's Customer Support AI.

        Your job is to help customers with:

        - orders
        - shipping
        - returns
        - refunds
        - cancellations
        - warranties
        - damaged products
        - discounts
        - company policies

        TOOLS:

        You have two tools.

        1. company_knowledge_base

        Use it for company policies and documentation.

        2. order_lookup

        Use it for specific customer order information.

        RULES:

        1. Never invent information.

        2. Never invent an order status.

        3. Never invent a refund.

        4. Never invent a discount.

        5. Never invent a delivery date.

        6. Never invent warranty conditions.

        7. Use the knowledge base for policy questions.

        8. Use the order database for order questions.

        9. Never reveal information belonging to another customer.

        10. Never expose passwords, authentication codes,
            or full payment-card information.

        11. If the information is unavailable, say so clearly.

        12. Do not make promises that are not supported by
            company policy or order data.

        13. Remain polite and professional.

        14. If you cannot solve the issue, mark the response
            for escalation using:

            [ESCALATE]

        15. If the user explicitly requests a human,
            the application will handle the escalation separately.

        ESCALATION:

        When you genuinely cannot resolve an issue with the
        available tools or information, write:

        [ESCALATE]

        followed by a short reason.

        Do not use [ESCALATE] for ordinary questions that
        you can answer from the available tools.
        """,

        tools=[
            knowledge_base_tool,
            order_lookup_tool
        ],

        llm=llm,

        allow_delegation=False,

        verbose=False,

        memory=True
    )

    return agent


# ============================================================
# 17. CREATE CREW
# ============================================================

def create_support_crew():

    agent = create_customer_support_agent()

    task = Task(

        description="""
        Handle the customer's current message.

        CUSTOMER MESSAGE:
        {customer_message}

        CONVERSATION HISTORY:
        {conversation_history}

        Instructions:

        1. Understand the customer's request.

        2. Use the company knowledge base when the question
           concerns company policies.

        3. Use the order lookup tool when the customer asks
           about a specific order.

        4. Use both tools if necessary.

        5. Do not guess.

        6. Give a concise, helpful answer.

        7. If the available information is insufficient to
           safely resolve the issue, use:

           [ESCALATE]

        8. If escalation is required, include a short reason
           after [ESCALATE].
        """,

        expected_output="""
        A professional customer-service response.

        If the issue cannot be resolved:
        [ESCALATE]
        followed by a short reason.

        Otherwise, provide only the customer-facing answer.
        """,

        agent=agent
    )

    crew = Crew(

        agents=[agent],

        tasks=[task],

        process=Process.sequential,

        memory=True,

        verbose=False
    )

    return crew


# ============================================================
# 18. GET / CACHE CREW
# ============================================================

def get_support_crew():

    if st.session_state.crew is None:

        st.session_state.crew = create_support_crew()

    return st.session_state.crew


# ============================================================
# 19. PROCESS CUSTOMER MESSAGE
# ============================================================

def process_customer_message(
    user_message: str
):

    # --------------------------------------------------------
    # Explicit human request
    # --------------------------------------------------------

    if user_requested_human(
        user_message
    ):

        reason = (
            "Customer explicitly requested "
            "human assistance."
        )

        add_escalation(
            user_message,
            reason
        )

        return (
            "Your request has been successfully "
            "escalated to a human agent."
        )

    # --------------------------------------------------------
    # Get previous conversation
    # --------------------------------------------------------

    conversation_history = (
        get_conversation_history()
    )

    # --------------------------------------------------------
    # Get CrewAI crew
    # --------------------------------------------------------

    crew = get_support_crew()

    # --------------------------------------------------------
    # Execute agent
    # --------------------------------------------------------

    try:

        result = crew.kickoff(

            inputs={

                "customer_message":
                    user_message,

                "conversation_history":
                    conversation_history
            }
        )

        response = str(result).strip()

    except Exception as error:

        # ----------------------------------------------------
        # If agent fails, escalate
        # ----------------------------------------------------

        add_escalation(

            user_message,

            "The AI agent could not process "
            f"the request: {str(error)}"
        )

        return (
            "Your request has been successfully "
            "escalated to a human agent."
        )

    # --------------------------------------------------------
    # Check agent escalation signal
    # --------------------------------------------------------

    if "[ESCALATE]" in response:

        parts = response.split(
            "[ESCALATE]",
            1
        )

        customer_response = parts[0].strip()

        reason = (
            parts[1].strip()
            if len(parts) > 1
            else "The AI agent could not resolve the issue."
        )

        add_escalation(
            user_message,
            reason
        )

        return (
            customer_response
            + "\n\n"
            + "Your request has been successfully "
              "escalated to a human agent."
        )

    return response


# ============================================================
# 20. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 Raabta AI")

    st.caption(
        "Single-Agent Customer Support AI"
    )

    st.divider()

    st.subheader("System Status")

    st.success("✓ Gemini connected")

    st.success("✓ FAISS knowledge base")

    st.success("✓ Order database")

    st.success("✓ Agent memory")

    st.divider()

    st.subheader(
        "Pending Escalations"
    )

    if st.session_state.pending_escalations:

        for escalation in (
            st.session_state.pending_escalations
        ):

            with st.expander(
                f"#{escalation['id']} — "
                f"{escalation['status']}"
            ):

                st.write(
                    "**Customer Issue:**"
                )

                st.write(
                    escalation["customer_issue"]
                )

                st.write(
                    "**Summary:**"
                )

                st.write(
                    escalation["summary"]
                )

    else:

        st.info(
            "No pending escalations."
        )


# ============================================================
# 21. MAIN APPLICATION
# ============================================================

st.title(
    "🤖 Raabta AI - Customer Support"
)

st.write(
    "Ask about orders, shipping, returns, "
    "refunds, cancellations, warranties, "
    "and company policies."
)


# ============================================================
# 22. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 23. CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Ask your question..."
)


if user_message:

    # --------------------------------------------------------
    # Show user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_message
        )

    st.session_state.messages.append(

        {
            "role": "user",
            "content": user_message
        }
    )

    # --------------------------------------------------------
    # Generate AI response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Raabta AI AI is thinking..."
        ):

            response = process_customer_message(
                user_message
            )

        st.markdown(
            response
        )

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": response
        }
    )
